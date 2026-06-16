import os
import sys
import shutil
import subprocess
from pathlib import Path

from src.utils.ImageStore import ImageStore

RIFE_SCRIPT = Path("./external/Practical-RIFE/inference_video.py")
TRAIN_LOG_PATH = Path("./external/Practical-RIFE/train_log/")
# Hardcoded output directory for Practical-RIFE interpolation. Do not change this.
RIFE_OUTPUT_DIR = Path("./vid_out/")

DEFAULT_MULTI = 2
DEFAULT_SCALE = 0.5

INTERPOLATED_DIRNAME = Path("interpolated/")

def _clear_directory(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def interpolate(source: ImageStore, multi: int = DEFAULT_MULTI, scale: float = DEFAULT_SCALE) -> ImageStore:
    """Add intermediate frames using Practical-RIFE, reading images from a directory."""
    if multi < 1:
        raise ValueError("Interpolation factor must be at least 1.")
    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise ValueError("Scale must be one of [0.25, 0.5, 1.0, 2.0, 4.0].")

    cache = source.cache.child(INTERPOLATED_DIRNAME)
    cached_entries = cache.get_entries()
    if cached_entries.get("multi") == multi and cached_entries.get("scale") == scale:
        print(f"[INFO] Using cached interpolated images from {cache.path}.")
        return cache
    
    _clear_directory(RIFE_OUTPUT_DIR)
    cmd = [ str(x) for x in [
        sys.executable, RIFE_SCRIPT,
        "--img", source.path,
        "--png",
        "--UHD",
        "--model", TRAIN_LOG_PATH,
        "--scale", scale,
        "--multi", multi
    ]]
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.dirname(__file__))
    env["PYTHONPATH"] = project_root + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    print(f"[INFO] Running Practical-RIFE: {' '.join(cmd)} with PYTHONPATH={env['PYTHONPATH']}")
    subprocess.run(cmd, check=True, env=env)

    cache.clear()
    shutil.move(RIFE_OUTPUT_DIR, cache.path)
    cache.save_entries({"multi": multi, "scale": scale})
    print(f"[INFO] Interpolation complete. {len(cache.get_image_filenames())} interpolated images saved to {cache.path}.")

    return cache

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interpolate frames using Practical-RIFE.")
    parser.add_argument("source", type=ImageStore, help="Directory containing images to interpolate.")
    parser.add_argument("--multi", type=int, default=DEFAULT_MULTI, help="Interpolation factor (default: 2)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale factor for output images (default: 0.5)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache and re-interpolate frames.")
    args = parser.parse_args()
    interpolate(args.source, multi=args.multi, scale=args.scale)

if __name__ == "__main__":
    main()
