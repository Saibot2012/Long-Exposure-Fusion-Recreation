import shutil
from pathlib import Path
import torch
import torchvision
import numpy as np
import cv2
import hashlib

VALID_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
HASH_LENGTH = 8

def is_valid_video(file: Path) -> bool:
    return file.exists() and file.is_file() and \
        any(file.suffix.lower() == extension for extension in VALID_VIDEO_EXTENSIONS)

def is_valid_image(file: Path) -> bool:
    return file.exists() and file.is_file() and \
        any(file.suffix.lower() == extension for extension in VALID_IMAGE_EXTENSIONS)

def clear_directory(directory: Path):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

def append_hash_to_name(path: Path) -> str:
    """Returns the name of a given path with a hash appended to ensure uniqueness."""
    hash_string = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:HASH_LENGTH]
    return f"{path.stem}_{hash_string}"

def get_image_paths(directory: Path) -> list[Path]:
    if not any_valid_images(directory):
        return []

    return sorted([path for path in directory.iterdir() if is_valid_image(path)])

def load_image(image: Path) -> torch.Tensor:
    return torchvision.io.decode_image(image).float() / 255.0

def save_image(image: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image, path)

def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0).contiguous()

def any_valid_images(directory: Path) -> bool:
    return directory.exists() and directory.is_dir() \
        and any(is_valid_image(file) for file in directory.iterdir())
