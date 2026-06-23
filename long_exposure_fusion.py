"""
long_exposure_fusion.py
"""

from __future__ import annotations

import shutil
import argparse
import numpy
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use("Agg")
import numpy as np
from pathlib import Path
from attr import dataclass

from src.utils.ImageStore import ImageStore
import src.utils.weight_map as weight_map
import src.pipeline.decode_video as decode_video
import src.pipeline.align_images as align_images
import src.pipeline.interpolate_images as interpolate_images
import src.pipeline.fuse_images as fuse_images
import src.pipeline.segment_picker as segment_picker
from src.pipeline.segment_higra import run_higra_segmenter
from src.utils.weight_map import MaskLoader
from src.pipeline.segment_higra import find_sharpest_frame

# ---------------------------------------------------------------------------#
# Constants
# ---------------------------------------------------------------------------#
# Batch size for image fusion
BATCH_SIZE = 4
# Batch size for fusion maps
WEIGHT_MAP_BATCH_SIZE = 32

# Cropping thresholds
INTERSECTION_THRESHOLD = 0.75  # Discard images with intersection below this threshold
DISCARD_RATIO_THRESHOLD = 0.1  # Abort if more than 10% of images are discarded
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
@dataclass
class LongExposureFusionConfig:
    reference_index: int
    weight_map_file: Path
    align: bool = False
    interpolate: int = None
    use_pyramid_decomposition: bool = False
    segment_higra: bool = False

    def __post_init__(self):
        if len(self.weight_maps) == 0:
            raise ValueError("At least one weight map must be provided.")

def run_long_exposure_fusion(
    source: ImageStore,
    config: LongExposureFusionConfig,
) -> ImageStore:
    mask_source = source

    # Optionally align
    if config.align:
        source = align_images.align(source, config.reference_index)
        
        config.reference_index = [
            int(path.stem) for path in source.get_image_filenames()
        ].index(config.reference_index)
    mask_source = source
    # Optionally interpolate

    if config.interpolate is not None:
        source = interpolate_images.interpolate(source, multi=config.interpolate)
        config.reference_index *= config.interpolate

    indexed_filenames = source.get_indexed_image_filenames()
    keys = list(indexed_filenames.keys())
    actual_reference_index = keys[config.reference_index]
    # Define segmentation masks
    run_higra_segmenter(
        source,
        reference_index=actual_reference_index
    )
    
    # Check mask size matches image size
    mask_loader = MaskLoader(mask_source)
    first_image = mask_source.load_image_at(list(mask_source.get_indexed_image_filenames().keys())[0])
    img_h, img_w = first_image.shape[1:3]
    first_masks = mask_loader.load_masks(list(mask_source.get_indexed_image_filenames().keys())[0])
    if first_masks:
        mask_h, mask_w = first_masks[0].shape[-2:]
        if img_h != mask_h or img_w != mask_w:
            raise ValueError(f"Mask size {mask_h}x{mask_w} does not match image size {img_h}x{img_w}. Redo segmentation on pc4302b.")
        else:
            print(f"[INFO] Mask sizes match successfully. Continuing to fusion.")

    if config.weight_map_file is None:
        raise ValueError("A weight map file must be provided. Use --maps <file>.")

    # Prepare dictionary of keys and functions for batch fusion
    # Initialize WeightMapGenerator with mask_dir, reference_index, and frame_count
    print(f"[DEBUG] config.reference_index before WeightMapGenerator: {config.reference_index}")
    decoder = weight_map.WeightMapGenerator(
        source=mask_source,  # use cropped source for masks
        reference_index=config.reference_index,
        frame_count=len(source.get_image_filenames())
    )
    weight_map_items = list(decoder.from_yaml_file(config.weight_map_file).items())

    # Fuse images using exposure fusion
    n_levels = 1
    if config.use_pyramid_decomposition:
        # Calculate pyramid levels based on image dimensions
        height, width = source.load_image_at(0).shape[1:3]
        n_levels = int(numpy.floor(numpy.log2(min(height, width)))) - 1

    for i in range(0, len(weight_map_items), WEIGHT_MAP_BATCH_SIZE):
        print(f"[INFO] Processing weight map batch {i // WEIGHT_MAP_BATCH_SIZE + 1} / {(len(weight_map_items) - 1) // WEIGHT_MAP_BATCH_SIZE + 1}")
        weight_map_batch = dict(weight_map_items[i:i + WEIGHT_MAP_BATCH_SIZE])
        destination = fuse_images.fuse(
            source,
            weight_map_batch,
            n_levels=n_levels,
        )

    indexed_filenames = source.get_indexed_image_filenames()
    keys = list(indexed_filenames.keys())
    source.copy_image_to(indexed_filenames[keys[0]], destination, Path("first.png"))
    source.copy_image_to(indexed_filenames[keys[-1]], destination, Path("last.png"))
    source.copy_image_to(indexed_filenames[keys[config.reference_index]], destination, Path("reference.png"))
    print(f"[DEBUG] reference frame being saved: {keys[config.reference_index]}")

    return destination, len(source.get_image_filenames())

# ---------------------------------------------------------------------------#
# Argument parsing
# ---------------------------------------------------------------------------#
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a burst of photos into a long-exposure image."
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs='?',
        help="Directory of images or path to a video file.",
    )
    parser.add_argument(
        "-m",
        "--maps",
        type=Path,
        help="Path to a YAML file containing weight map definitions to be added to the fusion."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for fused images (default: '.cache/<input_name>_<hash>/fused').",
    )
    parser.add_argument(
        "--reference",
        type=float,
        default=0,
        help="Index or ratio of reference image for alignment. If int >= 1, treated as index. If float in [0,1], treated as ratio of image count (default: 0).",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align frames before fusion.",
    )
    parser.add_argument(
        "--interpolate",
        type=int,
        default=None,
        help="Interpolate intermediate frames (RIFE --multi argument, default: None=no interpolation).",
    )
    parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Enable pyramid decomposition for blending (default: False, uses simple weighted average)."
    )
    parser.add_argument(
        "--segment_higra", 
        action="store_true"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache and re-process all frames.",
    )
    parser.add_argument("--start", type=float, default=0)

    parser.add_argument("--end", type=float, default=None)

    args = parser.parse_args()

    
    # Handle clearing all caches when no input is provided
    if args.input is None and args.clear_cache:
        print("[INFO] Clearing all caches.")
        ImageStore.clear_all_caches()
        return args
    
    # Input is required for normal operation
    if args.input is None:
        parser.error("Input is required unless using --clear-cache to clear all caches.")
    
    if args.reference < 0:
        raise ValueError("Reference must be non-negative (index or ratio).")
    if args.interpolate is not None and args.interpolate < 1:
        raise ValueError("Interpolation must be at least 1.")
    
    if args.clear_cache:
        cache = ImageStore.create_cache(args.input)
        print(f"[INFO] Clearing cache at {cache.path}.")
        cache.clear()

    return args

# ---------------------------------------------------------------------------#
# Main entry‑point
# ---------------------------------------------------------------------------#
def main() -> None:
    """Parse arguments and run the long-exposure pipeline."""
    args = _parse_args()

    if args.input is None:
        return
    

    if args.input.is_file():
        args.input = decode_video.decode(args.input, start_time=args.start, end_time=args.end)
    else:
        args.input = ImageStore(args.input)

    decoded_count = len(args.input.get_image_filenames())  # save before alignment
    first_frame_index = min(args.input.get_indexed_image_filenames().keys())
    if args.reference == 0:
        print("[INFO] Auto-selecting sharpest frame as reference...")
        args.reference = find_sharpest_frame(args.input)
        print(f"[INFO] Sharpest frame selected: {args.reference}")
    else:
        first_frame_index = min(args.input.get_indexed_image_filenames().keys())
    if args.reference == 0:
        args.reference = first_frame_index
    print(f"[DEBUG] first_frame_index: {first_frame_index}, args.reference: {args.reference}")
    image_count = len(args.input.get_image_filenames())
    print(f"[DEBUG] image_count: {image_count}, args.reference: {args.reference}")
    if args.reference == 0:
        args.reference = first_frame_index

    max_frame_index = max(args.input.get_indexed_image_filenames().keys())
    if args.reference >= 1:
        args.reference = int(args.reference)
        if args.reference > max_frame_index:
            raise ValueError(f"Reference index {args.reference} is out of range. Max valid frame index is {max_frame_index}")
    else:
        # Treat as ratio in [0, 1[
        args.reference = int(args.reference * (image_count - 1))
        print(f"[INFO] Using image {args.reference}/{image_count}) as reference")

    config = LongExposureFusionConfig(
        reference_index=args.reference,
        weight_map_file=args.maps,
        align=args.align,
        interpolate=args.interpolate,
        use_pyramid_decomposition=args.pyramid,
        segment_higra=args.segment_higra,
    )

    output_cache, fused_frame_count = run_long_exposure_fusion(
        source=args.input,
        config=config,
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for path in output_cache.path.iterdir():
            if path.is_file():
                shutil.copy2(path, args.output)

    print(f"[INFO] Fused images saved to {args.output or output_cache.path}")
    # Save comparison plot


    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    titles = ['first', 'last', 'reference','constant', 'luminance', 'partial']

    for ax, title in zip(axes, titles):
        img_path = output_cache.path / f'{title}.png'
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(np.array(img))
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.set_visible(False)

    plt.tight_layout(pad=1)
    plt.savefig(output_cache.path / 'comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[INFO] Comparison plot saved to {output_cache.path / 'comparison.png'}")

    #Summary of work
    
    print(f"\n{'='*50}")
    print(f"[SUMMARY] Pipeline Complete!")
    print(f"[SUMMARY] Frames decoded: {decoded_count}")
    print(f"[SUMMARY] Frames cropped/ used in fusion: {fused_frame_count}")
    print(f"[SUMMARY] Frames discarded: {decoded_count - fused_frame_count}")
    if args.interpolate:
        print(f"[SUMMARY] Frames after interpolation (x{args.interpolate}): {len(output_cache.get_image_filenames()) * args.interpolate}")    
    print(f"[SUMMARY] Output saved to: {args.output or output_cache.path}")
    print(f"{'='*50}\n")

    log_path = output_cache.path / 'pipeline.log'
    with open(log_path, 'w') as f:
        f.write(f"Pipeline Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Maps: {args.maps}\n")
        f.write(f"Reference: {args.reference}\n")
        f.write(f"Align: {args.align}\n")
        f.write(f"Interpolate: {args.interpolate}\n")
        f.write(f"Start: {args.start}, End: {args.end}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Frames decoded: {decoded_count}\n")
        f.write(f"Frames used in fusion: {fused_frame_count}\n")
        f.write(f"Frames discarded: {decoded_count - fused_frame_count}\n")
        f.write(f"Output saved to: {args.output or output_cache.path}\n")
    print(f"[INFO] Pipeline log saved to {log_path}")

# ---------------------------------------------------------------------------#
# Script execution
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
