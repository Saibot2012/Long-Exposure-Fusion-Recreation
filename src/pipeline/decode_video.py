import cv2
from pathlib import Path
import src.utils.utils as utils
from src.utils.ImageStore import ImageStore

DECODED_DIRNAME = Path("decoded/")

def decode(video_path: Path, start_time=0, end_time=None) -> ImageStore:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    if not any(str(video_path).lower().endswith(ext) for ext in utils.VALID_VIDEO_EXTENSIONS):
        raise ValueError(f"Input file {video_path} is not a supported video format.")

    cache = ImageStore.create_cache(video_path).child(DECODED_DIRNAME)

    if not cache.is_empty():
        print(f"[INFO] Using cached frames in {cache.path}.")
        return cache

    cache.path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback safety

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    i = start_frame

    while True:
        if end_frame is not None and i >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(str(cache.path / f"{i:06d}.jpg"), frame)
        i += 1

    cap.release()
    extracted = i - start_frame
    if extracted == 0:
        raise ValueError(f"No frames extracted between start={start_time}s and end={end_time}s. Check your --start and --end values.")

    print(f"[INFO] Extracted {i - start_frame} frames to {cache.path}.")
    return cache

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decode a video into PNG frames.")
    parser.add_argument("video_path", type=Path, help="Path to the input video file.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and re-decode video.")
    args = parser.parse_args()
    if args.clear_cache:
        cache = ImageStore.create_cache(args.video_path).child(DECODED_DIRNAME)
        print(f"[INFO] Clearing cache at {cache.path}.")
        cache.clear()
    output = decode(args.video_path)
    print(f"Frames saved to: {output.path}.")

if __name__ == "__main__":
    main()