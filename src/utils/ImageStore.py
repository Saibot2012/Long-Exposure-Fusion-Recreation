import torch
from pathlib import Path
import shutil
import yaml
from collections.abc import Iterable

import src.utils.utils as utils

CACHE_ROOT = Path(".cache/").resolve()

class ImageStore:
    """
    Manages a directory of images and associated metadata.
    Provides methods for saving, loading, batching, and caching images.
    """
    def __init__(self, directory: str | Path):
        self.path = Path(directory).resolve()

    @staticmethod
    def create_cache(source_path: Path) -> 'ImageStore':
        source_path = source_path.resolve()
        if CACHE_ROOT in source_path.parents:
            cache_path = source_path.parents[source_path.parents.index(CACHE_ROOT) - 1]
            return ImageStore(cache_path)

        return ImageStore(CACHE_ROOT / utils.append_hash_to_name(source_path))
    
    @staticmethod
    def clear_all_caches():
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)

    @staticmethod
    def _check_valid_filename(filename: Path):
        if filename.parent != Path('.'):
            raise ValueError("Filename must not contain parent directories. Use ImageStore.child() instead.")

    @property
    def cache(self) -> 'ImageStore':
        return ImageStore.create_cache(self.path)

    def child(self, dirname: Path) -> 'ImageStore':
        child_path = self.path / dirname
        if not self.path in child_path.parents:
            raise ValueError("Child path must be within the cache directory")
        return ImageStore(child_path)
    
    def save_image(self, image: torch.Tensor, filename: Path):
        ImageStore._check_valid_filename(filename)
        print(f"[DEBUG] Saving image at {self.path / filename}")
        utils.save_image(image, self.path / filename)

    def save_image_at(self, image: torch.Tensor, index: int):
        utils.save_image(image, self.path / f"{index:06d}.png")

    def load_image(self, filename: Path) -> torch.Tensor:
        if filename is None:
            return None
        ImageStore._check_valid_filename(filename)
        return utils.load_image(self.path / filename)
    
    def load_image_at(self, index: int) -> torch.Tensor:
        return self.load_image(self.get_indexed_image_filenames().get(index))

    def images(self, batch_size: int = 1) -> Iterable[torch.Tensor]:
        batch = []
        for filename in self.get_image_filenames():
            image = self.load_image(filename)
            batch.append(image)
            if len(batch) == batch_size:
                yield torch.stack(batch)  # [batch_size, C, H, W]
                batch = []
        if batch:
            yield torch.stack(batch)

    def get_image_filenames(self) -> list[Path]:
        return [Path(path.name) for path in utils.get_image_paths(self.path)]

    def get_indexed_image_filenames(self) -> dict[int, Path]:
        return {int(path.stem): path for path in self.get_image_filenames() if path.stem.isdigit()}

    def get_image_count(self) -> int:
        return len(self.get_image_filenames())

    def save_entries(self, data: dict[str, any]):
        if not len(data):
            return
        self.path.mkdir(parents=True, exist_ok=True)

        yaml_path = self.path / ".yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as file:
                existing_data = yaml.safe_load(file)
                if not isinstance(existing_data, dict):
                    raise ValueError("Existing YAML data is not a dictionary!")
                existing_data.update(data)
                data = existing_data
                
        with open(yaml_path, "w") as file:
            yaml.dump(data, file)

    def save_entry(self, key: str, value: any):
        self.save_entries({key: value})

    def get_entries(self) -> dict[str, any]:
        yaml_path = self.path / ".yaml"
        if not yaml_path.exists():
            return {}
        
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError("YAML data is not a dictionary!")
        return data

    def get_entry(self, key: str, default: any = None) -> any:
        return self.get_entries().get(key, default)

    def is_empty(self) -> bool:
        return  not self.path.exists() or not any(self.path.iterdir())

    def clear(self):
        if self.path.exists():
            assert CACHE_ROOT in self.path.parents # Failsafe
            shutil.rmtree(self.path)

    def copy_image_to(self, filename: Path, other: 'ImageStore', destination_filename: Path):
        if filename.suffix == destination_filename.suffix:
            ImageStore._check_valid_filename(filename)
            ImageStore._check_valid_filename(destination_filename)
            other.path.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.path / filename, other.path / destination_filename)
            return

        image = self.load_image(filename)
        other.save_image(image, destination_filename)
