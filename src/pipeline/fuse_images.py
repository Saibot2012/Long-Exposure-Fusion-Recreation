# image_fusion.py

import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed

from src.utils.pyramids import compute_gaussian_pyramid, compute_laplacian_pyramid, collapse_pyramid
from src.utils.weight_map import WeightMap
from src.utils.ImageStore import ImageStore
import matplotlib.pyplot as plt

BATCH_SIZE = 4
OUTPUT_DIRNAME = Path("fused/")

def fuse(source: ImageStore, weight_maps: dict[str, WeightMap], n_levels: int = 1) -> None:
    """
    Exposure fusion for an iterator of images, one at a time, with late normalization.
    image_tensor_iterator: generator of torch tensors [C,H,W]
    weight_map_fn: function(contrast, saturation, well_exposedness, index) -> weight map for each image
    do_pyramid_decomposition: if False, skip pyramid blending and do simple weighted average
    """
    do_pyramid_decomposition = n_levels > 1   #Use pyramid fusion if --pyramid is called, else do simpe average weighting
    cache = source.cache.child(OUTPUT_DIRNAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    levels = list(range(n_levels))
    blended_pyramid = None  # Will be [n_maps][levels][C,h,w]
    weight_sum_pyramid = None    # Will be [n_maps][levels][1,h,w]
    image_index = 0

    image_count = source.get_image_count()
    if image_count == 0:
        raise ValueError("No images provided to image_fusion.")
    
    progress_bar = tqdm(total=image_count, desc="Fusing images", unit="images")
    for batch in source.images(batch_size=BATCH_SIZE):  # [N,C,H,W]
        batch = batch.to(device) # [N,C,H,W]. Loads images in batches of 4 to avoid running out of memory.


        if do_pyramid_decomposition:
            img_gaussian_pyramid = compute_gaussian_pyramid(batch, n_levels=n_levels) # list of [N,C,H,W]
            img_laplacian_pyramid = compute_laplacian_pyramid(img_gaussian_pyramid) # list of [N,C,H,W]
        else:
            # Skip pyramid decomposition - work directly with original images
            img_laplacian_pyramid = [batch]  # Single level containing original images
            n_levels = 1
            levels = [0]

        if blended_pyramid is None:
            assert weight_sum_pyramid is None
            blended_pyramid = [
                [torch.zeros_like(img_laplacian_pyramid[k][0], device=device) for k in levels]
                for _ in range(len(weight_maps))
            ]  # [n_maps][levels][C,h,w]
            weight_sum_pyramid = [
                [torch.zeros_like(img_laplacian_pyramid[k][0], device=device) for k in levels]
                for _ in range(len(weight_maps))
            ]  # [n_maps][levels][1,h,w]



        for weight_map_index, weight_map in enumerate(weight_maps.values()):
            weights = weight_map(batch, image_index)  # [N,1,H,W]
            weights = torch.clamp(weights, min=1e-12, max=1e6)  #Compute per-pixel weights for this batch using the weight map function. Clamp to avoid zeros or infinity.

            weight_gaussian_pyramid = compute_gaussian_pyramid(weights, n_levels) if do_pyramid_decomposition else [weights]  # list of [N,1,H,W]
            for k in levels:
                blended_pyramid[weight_map_index][k] += torch.sum(weight_gaussian_pyramid[k] * img_laplacian_pyramid[k], dim=0)  # [C,h,w]
                weight_sum_pyramid[weight_map_index][k] += torch.sum(weight_gaussian_pyramid[k], dim=0)  # [1,h,w]. Accumulate weighted pixel values and total weights across all batches. This is the core of the fusion: each pixel gets weight * pixel_value summed across all frames.

        image_index += batch.shape[0]  # Increment by batch size
        progress_bar.update(batch.shape[0])
    progress_bar.close()

    frame_count_map = torch.zeros(1, device=device)  #initialize frame count to 0 first. will be reshaped.
    first_batch = True #handle first batch separately as we need to initialize map size.
    for batch in source.images(batch_size=BATCH_SIZE):  #loop through in batches of 4
        batch = batch.to(device) #GPU   
        valid = (batch.sum(dim=1, keepdim=True) > 0.01).float()  # [N,1,H,W]  For each pixel in each frame, sum R+G+B channels. If sum > 0.01 → valid pixel (1.0), otherwise black/warped pixel (0.0). Result shape is one validity value per pixel per frame.
        if first_batch:
            frame_count_map = valid.sum(dim=0)  # [1,H,W]
            first_batch = False
        else:
            frame_count_map += valid.sum(dim=0)


    frame_count_np = frame_count_map.squeeze().cpu().numpy()

    for weight_map_index, weight_map_name in enumerate(weight_maps.keys()):
        weight_sum_np = weight_sum_pyramid[weight_map_index][0][0].squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        im1 = axes[0].imshow(frame_count_np, cmap='hot', )
        plt.colorbar(im1, ax=axes[0], label='Number of frames')
        axes[0].set_title('Frame count per pixel')

        im2 = axes[1].imshow(weight_sum_np, cmap='hot', vmin=weight_sum_np.min(), vmax=weight_sum_np.max())
        plt.colorbar(im2, ax=axes[1], label='Cumulative weight')
        axes[1].set_title(f'Cumulative frame weight per pixel ({weight_map_name})')
        print(f"[DEBUG] {weight_map_name} weight_sum max: {weight_sum_np.max():.2f}, min: {weight_sum_np.min():.2f}")

        plt.tight_layout()
        cache.path.mkdir(parents=True, exist_ok=True)
        plt.savefig(cache.path / f'frame_heatmap_{weight_map_name}.png', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Frame heatmap saved to {cache.path / f'frame_heatmap_{weight_map_name}.png'}")

    

    for weight_map_index, weight_map_name in enumerate(weight_maps.keys()):
        normalized_blended_pyramid = [blended_pyramid[weight_map_index][k] / weight_sum_pyramid[weight_map_index][k] for k in levels]  # [C,h,w]. After processing all frames, normalize by total weight — this gives the weighted average per pixel.
        normalized_blended_pyramid = [level.unsqueeze(0) for level in normalized_blended_pyramid]
        if do_pyramid_decomposition:
            fused_image = collapse_pyramid(normalized_blended_pyramid).squeeze(0)
        else:
            fused_image = normalized_blended_pyramid[0].squeeze(0)

        print(f"[DEBUG] {weight_map_name} - Fused image min: {fused_image.min():.3f}, max: {fused_image.max():.3f}, mean: {fused_image.mean():.3f}")
        print(f"[DEBUG] {weight_map_name} - Any NaN: {torch.isnan(fused_image).any()}, Any Inf: {torch.isinf(fused_image).any()}")
        cache.save_image(fused_image, Path(weight_map_name + ".png"))


    return cache
