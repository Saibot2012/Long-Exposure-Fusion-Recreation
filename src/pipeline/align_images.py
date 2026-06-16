from collections.abc import Iterable
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from tqdm import tqdm
import cv2
import shapely
import largestinteriorrectangle
from lightglue import LightGlue, DISK
import torch
import kornia

from src.utils.ImageStore import ImageStore

ALIGNED_DIRNAME = Path("aligned/")
CROPPED_DIRNAME = Path("cropped/")
REFERENCE_INDEX_FILENAME = "reference.txt"

INTERSECTION_THRESHOLD = 0.70
COVERAGE_THRESHOLD = 0.85
MAX_NUM_KEYPOINTS = 12000
FEATURE_EXTRACTION_METHOD = 'disk'
HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
RANSAC_REPROJECTION_THRESHOLD = 3.0
def _is_sane_homography(H: np.ndarray, max_scale: float = 2.0, max_rotation_deg: float = 15.0) -> bool:
    """Reject homographies with extreme scale or rotation."""
    # Decompose: extract scale and rotation from the upper-left 2x2
    a, b = H[0, 0], H[0, 1]
    c, d = H[1, 0], H[1, 1] #Extracts top left 2x2 homography matrix
    scale_x = np.sqrt(a**2 + c**2)
    scale_y = np.sqrt(b**2 + d**2) #Compute how much the image is being scaled in X and Y directions.
    rotation_deg = abs(np.degrees(np.arctan2(c, a)))  #rotation angle in degrees
    if rotation_deg > 180:
        rotation_deg = 360 - rotation_deg  #normalize betwen 0 and 180
    return (
        1 / max_scale <= scale_x <= max_scale and  #true only if scale is between 0.5-2x and rotation is less than 15deg
        1 / max_scale <= scale_y <= max_scale and
        rotation_deg <= max_rotation_deg
    )

def align(source: ImageStore, reference_index: int) -> ImageStore:
    """
    Aligns images to the reference image and crops them into the given cache under cropped/.
    Returns a directory containing the cropped images. Some images may be discarded.
    """

    aligned_cache = source.cache.child(ALIGNED_DIRNAME) #one cache for aligned frames 
    cropped_cache = source.cache.child(CROPPED_DIRNAME) #one cache for cropped frames

    if cropped_cache.get_entry("reference_index") == reference_index:
        print(f"[INFO] Using cached cropped images from {cropped_cache.path}.")
        return cropped_cache

    # Align images
    indexed_image_paths = source.get_indexed_image_filenames()
    aligned_cache.clear()
    transformed_polygons = _align_images(
        images=(source.load_image(indexed_image_paths[i]) if i in indexed_image_paths else None for i in range(max(indexed_image_paths) + 1)),
        reference_image=source.load_image(indexed_image_paths[reference_index]),
        destination=aligned_cache,
        image_count=len(indexed_image_paths)
    )

    # Crop images
    indexed_image_paths = aligned_cache.get_indexed_image_filenames()
    cropped_cache.clear()
    print("max key:", max(indexed_image_paths))
    print("num keys:", len(indexed_image_paths))

    missing = [
        i for i in range(max(indexed_image_paths) + 1)
        if i not in indexed_image_paths
    ]
    print("missing:", missing[:20])
    _crop_aligned_images(
        
        images=(aligned_cache.load_image(indexed_image_paths[i]) if i in indexed_image_paths else None for i in range(max(indexed_image_paths) + 1)),        polygons=transformed_polygons,
        reference_index=reference_index,
        destination=cropped_cache,
        image_count=len(indexed_image_paths),
    ) #Takes the aligned frames and crops them all to the largest common rectangle, so all frames are the same size with no black borders.
    cropped_cache.save_entry("reference_index", reference_index)

    return cropped_cache #saves which reference frame was used.

def _align_images(
    images: Iterable[torch.Tensor],
    reference_image: torch.Tensor,
    destination: ImageStore,
    image_count: int = None
) -> dict[int, Polygon]:
    print(f"[INFO] Aligning images to reference using DISK+LightGlue (feature matching only). Saving to {destination.path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = DISK(max_num_keypoints=MAX_NUM_KEYPOINTS).eval().to(device) #loads DISK(extracts keypoints)  lightglue(matches keypoints between 2 images)
    matcher = LightGlue(features=FEATURE_EXTRACTION_METHOD).eval().to(device)

    reference_image = reference_image.to(device)
    reference_features = extractor.extract(reference_image) #Extracts keypoints from ref frame once.
    height, width = reference_image.shape[1:]

    reference_corners = torch.tensor(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        device=device,
        dtype=torch.float32
    ) #4 corners of the reference frame, track each frame's boundaries end up after transformation.
    transformed_polygons = {}
    discarded = []
    for i, image in tqdm(
        iterable=((i, image.to(device)) for i, image in enumerate(images) if image is not None),
        total=image_count,
        desc="Aligning images",
        unit="img"
    ):
        # print(f"[DEBUG] image {i}: dtype={image.dtype}, min={image.min():.3f}, max={image.max():.3f}, shape={image.shape}")
        target_features = extractor.extract(image)
        matches = matcher({"image0": reference_features, "image1": target_features})["matches"][0] #for each frame extract keypoints and match against ref using lightglue.
        reference_keypoints = reference_features["keypoints"][0]
        target_keypoints = target_features["keypoints"][0]

        if len(matches) < 4:  #need at least 4 for a homography. discard if <4.
            discarded.append(i)
            continue

        H, _ = cv2.findHomography(  #Compute the homography matrix H: a 3x3 matrix that transforms the current frame's coordinate space to match the reference frame.
            target_keypoints[matches[:, 1]].cpu().numpy(),
            reference_keypoints[matches[:, 0]].cpu().numpy(),
            method=HOMOGRAPHY_METHOD,
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD
        )
        
        if H is None:
            discarded.append(i)
            continue
        # print(f"[DEBUG] H for image {i}:\n{H}")
        # Transform image and corners
        
        # kornia is slightly faster than cv2 here.
        H = torch.from_numpy(H).to(device).float().unsqueeze(0)
        aligned = kornia.geometry.transform.warp_perspective(image.unsqueeze(0), H, dsize=(height, width))[0] #Apply the homography to warp the frame so it aligns with the reference.
        corners = kornia.geometry.transform_points(H, reference_corners.unsqueeze(0))[0] #Transform the 4 corners of the frame using H — this gives the polygon boundary of where valid pixels are in the aligned frame. Stored for later use in cropping.
        transformed_polygons[i] = Polygon(corners.cpu().numpy())
        destination.save_image_at(aligned.cpu(), i)

    if len(discarded):
        print(f"[WARN] Some images could not be aligned: {discarded}.")
        print(f"[WARN] Total discarded: {len(discarded)} out of {image_count} images ({len(discarded) / image_count:.1%}).")
    if len(transformed_polygons) == 0:
        raise ValueError("No frames could be aligned. Check if the video has enough feature points.")
    print(f"[INFO] Alignment complete. {image_count - len(discarded)} aligned images saved to {destination.path}.")

    return transformed_polygons

def _crop_aligned_images(
    images: Iterable[torch.Tensor],
    polygons: dict[int, Polygon],
    reference_index: int,
    destination: ImageStore,
    image_count: int = None
) -> int:
    reference_polygon = polygons[reference_index] #get reference frame's polygon and area
    reference_area = reference_polygon.area

    def is_valid_polygon(polygon):
        if polygon is None or not polygon.is_valid:
            return False
        intersection_ratio = polygon.intersection(reference_polygon).area / reference_area
        if intersection_ratio < INTERSECTION_THRESHOLD:
            return False
        # Check if polygon is approximately rectangular
        bbox_area = polygon.minimum_rotated_rectangle.area
        rectangularity = polygon.area / bbox_area
        return rectangularity >= 0.80     # Checks whether frame is 80% rectangular and whether it intersects 75% with reference frame

    discarded = [i for i, polygon in polygons.items() if not is_valid_polygon(polygon)]  #build a list of bad frames and remove them from polygon.
    if len(discarded):
        print(f"[WARN] Some images could not be cropped: {discarded}.")
        print(f"[WARN] Total discarded: {len(discarded)} out of {len(polygons)} images ({len(discarded) / len(polygons):.1%}).")
    polygons = {i: polygon for i, polygon in polygons.items() if i not in discarded}
    print(f"[DEBUG] Polygons after filtering: {len(polygons)}")
    if len(polygons) == 0:
        raise ValueError("All frames were discarded during alignment. Try lowering INTERSECTION_THRESHOLD or COVERAGE_THRESHOLD.")

    # Build a coverage map: how many frames contain each pixel
    bounds = reference_polygon.bounds  # (minx, miny, maxx, maxy)
    canvas_h = int(np.ceil(bounds[3]))
    canvas_w = int(np.ceil(bounds[2]))
    coverage = np.zeros((canvas_h, canvas_w), dtype=np.float32) #Build a coverage map: for each pixel, count how many frames contain it.
    for poly in polygons.values():
        coords = np.array(poly.exterior.coords, dtype=np.int32)
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(mask, [coords], 1)
        coverage += mask

    # Keep pixels present in at least COVERAGE_THRESHOLD fraction of frames
    binary = ((coverage / len(polygons)) >= COVERAGE_THRESHOLD).astype(np.uint8) #Keep only pixels that appear in at least 70% of frames (COVERAGE_THRESHOLD = 0.70).
    print(f"[INFO] Coverage mask: {binary.mean():.1%} of reference area retained at threshold={COVERAGE_THRESHOLD}.")

    x, y, w, h = largestinteriorrectangle.lir(binary.astype(bool)).astype(int) #Find the largest interior rectangle that fits inside coverage mask
    print(f"[INFO] Cropping region from LIR: x={x}, y={y}, w={w}, h={h}.")

    for i, image in tqdm(
        ((i, image) for i, image in enumerate(images) if image is not None and i in polygons),
        total=image_count - len(discarded),
        desc="Cropping images",
        unit="img" 
    ):
        cropped = image[:, y:y+h, x:x+w] #Apply same crop every surviving frame, so all frames are the same size.
        destination.save_image_at(cropped, i)
    print(f"[INFO] Cropping complete. {len(polygons)} cropped images saved to {destination.path}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align and crop a sequence of images.")
    parser.add_argument("source", type=ImageStore, help="Directory containing images to align.")
    parser.add_argument("--reference", type=int, default=0, help="Index of reference image. (default: 0)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and re-align frames.")
    args = parser.parse_args()
    
    if args.clear_cache:
        aligned_cache = args.source.cache.child(ALIGNED_DIRNAME)
        cropped_cache = args.source.cache.child(CROPPED_DIRNAME)
        print(f"[INFO] Clearing cache at {aligned_cache.path}.")
        aligned_cache.clear()
        print(f"[INFO] Clearing cache at {cropped_cache.path}.")
        cropped_cache.clear()

    align(args.image_dir, args.reference)

if __name__ == "__main__":
    main()
