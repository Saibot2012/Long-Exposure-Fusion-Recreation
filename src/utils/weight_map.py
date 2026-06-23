# weight_map.py

import torch
import torchvision
import os
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from guided_filter_pytorch.guided_filter import GuidedFilter
import matplotlib.pyplot as plt
import numpy as np
from src.utils.ImageStore import ImageStore

# Guided Filter parameters
DEFAULT_GUIDED_FILTER_RADIUS = 120
DEFAULT_GUIDED_FILTER_EPSILON = 1e-6
DEFAULT_GUIDED_FILTER_K = 0.9

DEFAULT_TIME_LAPSE_TAU = 1 # Standard deviation of time lapse decay scaled by frame count

MASKS_DIRNAME = Path("masks/")
BLENDED_DIRNAME = Path("blended/")
class MaskLoader:
    def __init__(self, source: ImageStore):
        self.source = source
        self.masks_cache = source.child(MASKS_DIRNAME)

    def load_masks(self, frame_index: int) -> list[np.ndarray]:
        """
        Load an 8-bit-per-pixel PNG mask for the given frame index and return (masks, object_ids).
        Returns:
            masks: batch of binary torch tensors (H,W), one per object_id (including background 0)
        """
        try:
            mask_id_map = torch.load(self.masks_cache.path / f"{frame_index:06d}.pt", map_location='cpu')
        except FileNotFoundError:
            return []

        # Find object count
        object_count = mask_id_map.max().item()
        # Convert to PyTorch tensors
        masks = [(mask_id_map == object_id).unsqueeze(0) for object_id in range(1, object_count + 1)]
        return masks

class MaskProcessor:
    def __init__(
        self,
        source: ImageStore,
        radius: int = DEFAULT_GUIDED_FILTER_RADIUS,
        epsilon: float = DEFAULT_GUIDED_FILTER_EPSILON,
        sharpness: float = DEFAULT_GUIDED_FILTER_K
    ):
        self.mask_loader = MaskLoader(source)
        self.indexed_keys = list(source.get_indexed_image_filenames().keys())
        self.cache = {}
        self.radius = radius
        self.epsilon = epsilon
        self.sharpness = sharpness

    def get_masks_for_batch(self, image_batch: torch.Tensor, i: int):
        if i in self.cache:
            return self.cache[i]

        N, _, H, W = image_batch.shape
        device = image_batch.device
        
        batch_masks = []
        for frame_idx in range(i, i + N):
            actual_frame_key = self.indexed_keys[frame_idx]
            frame_masks = self.mask_loader.load_masks(actual_frame_key)
            frame_masks = [mask.to(device).float() for mask in frame_masks]
            batch_masks.append(frame_masks) #Load the .pt mask file for each frame in the batch. Each mask is a binary tensor — 1 where the object is, 0 elsewhere.

        object_count = max(len(frame_masks) for frame_masks in batch_masks)
        if object_count == 0:
            return [torch.ones((N, 1, H, W), device=device)] #If no masks exist (segmentation was skipped), return all ones — treat entire image as one region. 
        print(f"[DEBUG] batch {i}: object_count={object_count}")
        mask_batches = []
        for object_id in range(object_count):
            mask_batch = []
            for frame_masks in batch_masks:
                if object_id < len(frame_masks):
                    mask_batch.append(frame_masks[object_id])
                else:
                    mask_batch.append(torch.zeros((1, H, W), device=device))
            mask_batches.append(torch.stack(mask_batch, dim=0)) #Reorganize masks by object ID across the batch — mask_batches[0] has object 0's mask for all frames in the batch, mask_batches[1] has object 1's mask, etc.

        guided_filter = GuidedFilter(self.radius, self.epsilon)
        # Perform guided filtering on masks
        for mask_index, mask_batch in enumerate(mask_batches):
            filtered = guided_filter(rgb_to_gray(image_batch), mask_batch)

            # Push mask towards boolean values
            filtered = tunable_sigmoid(filtered, self.sharpness)

            mask_batches[mask_index] = filtered

        background = torch.ones((N, 1, H, W), device=device) - torch.sum(torch.stack(mask_batches, dim=0), dim=0)
        mask_batches = [background] + mask_batches #Computes the background mask as 1 - sum(all object masks). So if sky mask = 0.8 at a pixel, background = 0.2 at that pixel. Then prepends background to the list so:

        self.cache = { i: mask_batches }
        return mask_batches
        #raw higra mask has hard binary edges (0 or 1). The guided filter smooths these edges using the image itself as a guide, so mask transitions follow natural image boundaries (e.g. the skyline edge) instead of being a hard cut.

class WeightMap(ABC):
    """Abstract base class for weight map implementations."""
    
    @abstractmethod
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        """
        Compute weight map for a batch of images.
        
        Args:
        - batch: [N,C,H,W] image tensor
        - i: Index of the image in the batch
        
        Returns:
        - Computed weight map tensor
        """
        raise NotImplementedError

class MaskedWeightMap(WeightMap):
    """
    Hybrid masked weight map that computes a weighted combination of weight maps
    using masks for each frame from a given MaskLoader.
    """
    
    def __init__(self, weight_maps: list[WeightMap], mask_processor: MaskProcessor):
        self.weight_maps = weight_maps
        self.mask_processor = mask_processor

    def __call__(self, batch, i):
        mask_batches = self.mask_processor.get_masks_for_batch(batch, i)
        weight_outputs = [
            weight_map(batch, i) for weight_map in self.weight_maps[:len(mask_batches)]
        ]
        masked_weights = [mask * weight for mask, weight in zip(mask_batches, weight_outputs)]
        return torch.sum(torch.stack(masked_weights, dim=0), dim=0)
    
# class for partial output:

# Gets the segmentation masks (background + sky regions)
# Computes weights for each region using its own weight map
# Multiplies each weight map by its corresponding mask
# Sums them all together

# So background pixels use ReferenceWeightMap, segmented pixels use LinearValueWeightMap, and the guided filter smooths the mask edges so there's no hard boundary between regions.
    
class TimeLapsedWeightMap(WeightMap):
    """Time-lapse fusion weight map."""
    
    def __init__(self, reference_index: int, tau: float, weight_map: WeightMap, decay_map: WeightMap):
        """
        Args:
        - reference_index: Reference frame index
        - tau: Decay rate factor
        - weight_map: Base weight map
        - decay_map: Decay weight map, determines pixel decay rate
        """
        if tau <= 0:
            raise ValueError("tau must be greater than 0")
        if reference_index < 0:
            raise ValueError("reference_index must be non-negative")
        
        self.reference_index = reference_index
        self.tau = tau
        self.weight_map = weight_map
        self.decay_map = decay_map

    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        batch_size = batch.shape[0]
        indexes = i + torch.arange(batch_size, device=batch.device)

        # Sigma: either pixelwise or scalar
        time_decay_map = self.decay_map(batch, i)  # [N,1,H,W]
        sigma = self.tau * (time_decay_map + 1e-12)  # [N,1,H,W]

        # Contribution: Gaussian temporal weight
        # (idxs - reference_index) shape: [N,1,1,1] for broadcasting
        delta = (indexes - self.reference_index).view(batch_size, 1, 1, 1)
        contribution = torch.exp(-(delta ** 2) / (2 * (sigma ** 2)))
        
        # Compute all base weights in batch
        base_weights = self.weight_map(batch, i)  # [N,1,H,W]
        return contribution * base_weights
        
#Applies a Gaussian temporal decay around the reference frame:

#Frames close to the reference frame get high weight
#Frames far from the reference frame get low weight
#sigma (tau) controls how quickly the weight drops off with distance
#The base weight_map (e.g. contrast) is then multiplied by this temporal decay
    
class WeightedWeightMap(WeightMap):
    """Weighted scaling of a weight map."""
    
    def __init__(self, weight_map, weight):
        self.weight_map = weight_map
        self.weight = weight
    
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        return self.weight * self.weight_map(batch, i)
    
class InverseWeightMap(WeightMap):
    """Inverse of a weight map function or instance."""
    
    def __init__(self, weight_map: WeightMap):
        self.weight_map = weight_map
    
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        return 1.0 / (self.weight_map(batch, i) + 1e-12)

class ExposureWeightMap(WeightMap):
    """Weight map for exposure fusion."""

    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        contrast = ContrastWeightMap()(batch, i)
        saturation = SaturationWeightMap()(batch, i)
        well_exposedness = WellExposednessWeightMap()(batch, i)

        return (contrast ** 1) * (saturation ** 1) * (well_exposedness ** 1)

class ContrastWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        gray_batch = rgb_to_gray(batch)
        kernel = torch.tensor(
            [[0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]],
            device=gray_batch.device,
            dtype=gray_batch.dtype,
        ).unsqueeze(0).unsqueeze(0)
        
        return torch.abs(
            torch.nn.functional.conv2d(gray_batch, kernel, padding="same")
        ) #Adds a laplacian kernel to grayscale image to detect edges, high contrast pixels gets a high weight, so sharper frames contribute more to fusion.

class SaturationWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        return torch.std(batch, dim=1, keepdim=True) #Computes standard deviation across R, G, B channels per pixel. High std = vivid colors (saturated). Low std = grey/desaturated. Colorful pixels get higher weight.

class WellExposednessWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        sigma = 0.2
        return torch.exp(
            -torch.sum(
                (batch - 0.5) ** 2, dim=1, keepdim=True
            ) / (2 * sigma ** 2)
        )   #Gaussian centered at 0.5(mid brightness). Pixels near 0.5 gets highest weight.

class LuminanceWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        batch = srgb_to_linear(batch)
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=batch.device)
        return torch.tensordot(batch, weights, dims=([1], [0])).unsqueeze(1) #Computes perceived brightness using standard luminance coefficients — green contributes most (0.7152), red second (0.2126), blue least (0.0722). This matches how human eyes perceive brightness.

class LinearValueWeightMap(WeightMap):
    """Weight map for linear RGB."""
    
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        """
        Args:
        - batch: [N,C,H,W] image tensor
        - i: Index of the image in the batch

        Returns:
        - Computed weight map for linear RGB.
        """
        return ValueWeightMap()(srgb_to_linear(batch), i) #Same as ValueWeightMap but first converts from sRGB to linear RGB before taking the max. sRGB is gamma-corrected (how screens display), linear is the actual physical light intensity. More physically accurate weighting.

class ConstantWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        return torch.ones_like(batch[:, 0:1, :, :])  #Returns all 1s for every pixel in the frame.  

class ValueWeightMap(WeightMap):
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        return torch.max(batch, dim=1, keepdim=True)[0] #For each pixel, returns the maximum value across R, G, B channels. Brighter pixels get higher weight. So overexposed/bright regions contribute more to the fusion.

class ReferenceWeightMap(WeightMap):
    """Weight map that returns ones for reference frame, zeros otherwise."""

    def __init__(self, reference_index: int, frame_count: int):
        self.reference_index = reference_index
        self.frame_count = frame_count
    
    def __call__(self, batch: torch.Tensor, i: int) -> torch.Tensor:
        """
        Args:
        - batch: [N,C,H,W] image tensor
        - i: Index of the first image in the batch

        Returns:
        - Weight map with ones for reference frame, zeros otherwise.
        """
        N = batch.shape[0]
        device = batch.device
        
        # Create frame indices for the batch
        frame_indices = torch.arange(i, i + N, device=device)
        
        # Create mask for reference frame
        is_reference = (frame_indices == self.reference_index).float()
        
        # Reshape to [N, 1, 1, 1] for broadcasting to [N, 1, H, W]
        weight_mask = is_reference.view(N, 1, 1, 1) * self.frame_count

        # Create weight map with same spatial dimensions as input
        return weight_mask.expand_as(batch[:, 0:1, :, :])
    
    # Creates a weight of frame_count (e.g. 1101) for the reference frame and 0 for all other frames. This effectively says "use only the reference frame for these pixels". The multiplication by frame_count ensures the reference frame dominates over other weight maps that sum weights across all frames.

def srgb_to_linear(batch: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB to linear RGB.
    """
    return torch.where(
        batch <= 0.04045,
        batch / 12.92,
        ((batch + 0.055) / 1.055) ** 2.4
    )

def rgb_to_gray(batch: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale.
    """
    return torch.mean(batch, dim=1, keepdim=True)

class WeightMapGenerator:
    """
    Weight map generator class that handles JSON decoding and weight map creation.
    Stores mask_loader and reference_index for future use.
    """
    
    # Static dictionary of available weight map classes
    basic_weight_map_classes = {
        'exposure': ExposureWeightMap,
        'contrast': ContrastWeightMap,
        'saturation': SaturationWeightMap,
        'wellExposedness': WellExposednessWeightMap,
        'luminance': LuminanceWeightMap,
        'LinearValue': LinearValueWeightMap,
        'constant': ConstantWeightMap,
        'value': ValueWeightMap,
    }
    
    def __init__(self, source: ImageStore, reference_index: int, frame_count: int):
        """
        Initialize the weight map generator.
        
        Args:
            source: ImageStore instance containing mask images
            reference_index: Index of the reference frame
            frame_count: Total number of frames (used for tau scaling in time-lapse)
        """
        self.mask_processor = MaskProcessor(source)
        self.reference_index = reference_index
        self.frame_count = frame_count
    
    @staticmethod
    def from_string(name: str) -> WeightMap:
        """
        Create a weight map instance from a string name.
        
        Args:
            name: String name of the weight map (e.g., 'exposure', 'contrast', etc.)
        Returns:
            WeightMap instance (possibly wrapped in InverseWeightMap)
        """
        
        if name not in WeightMapGenerator.basic_weight_map_classes:
            raise ValueError(f"Unknown weight map: {name}. Must be one of: {', '.join(WeightMapGenerator.basic_weight_map_classes.keys())}")

        return WeightMapGenerator.basic_weight_map_classes[name]()
    
    def from_yaml_file(self, yaml_path: Path) -> dict[str, WeightMap]:
        """
        Load a YAML file and decode it as weight maps using from_json.
        Always returns a dictionary mapping unique labels to individual weight map functions.
        
        Args:
            yaml_path: Path to YAML file

        Returns:
            Dictionary mapping unique labels to individual weight map functions
        """
        with open(yaml_path, 'r') as f:
            obj = yaml.safe_load(f)

        print(f"[INFO] Loaded weight map configuration from {yaml_path}")

        if not isinstance(obj, dict):
            raise ValueError("Top-level YAML structure must be a dictionary")
        if not obj.keys() <= {'filter', 'maps'}:
            raise ValueError("YAML can only contain 'filter' and 'maps' keys at the top level")
            
        if 'filter' in obj:
            if not isinstance(obj['filter'], dict):
                raise ValueError("'filter' key must contain a dictionary of filter parameters")
            if not obj['filter'].keys() <= {'radius', 'epsilon', 'sharpness'}:
                raise ValueError("'filter' dictionary can only contain 'radius', 'epsilon', and 'sharpness' keys")
        
            filter_config = obj['filter']
            self.mask_processor.set_parameters(
                radius=filter_config.get('radius'),
                epsilon=filter_config.get('epsilon'),
                sharpness=filter_config.get('sharpness'),
            )

        if 'maps' not in obj:
            raise ValueError("YAML must contain 'maps' key with weight map definitions")
        if not isinstance(obj['maps'], list):
            raise ValueError("'maps' key must contain a list of map definitions")
        
        maps = {}
        for map_index, map_obj in enumerate(obj['maps']):
            if not isinstance(map_obj, dict):
                raise ValueError("Each map definition must be a dictionary")
            if 'name' in map_obj and not isinstance(map_obj['name'], str):
                raise ValueError("Map 'name' must be a string if provided")
            
            maps[map_obj.get('name') or str(map_index)] = self.from_dict(
                {k: v for k, v in map_obj.items() if k != 'name'}
            )
        return maps

    def from_dict(self, map_dict: dict) -> WeightMap:
        """
        Create a weight map from a dictionary specification.
        """

        if 'type' not in map_dict:
            raise ValueError("Weight map dictionary must contain a 'type' key")
        
        if 'weight' in map_dict and not isinstance(map_dict['weight'], (int, float)):
            raise ValueError("Weight map 'weight' must be a number")
        if map_dict.get('weight', 1) != 1:
            return WeightedWeightMap(
                weight_map=self.from_dict({k: v for k, v in map_dict.items() if k != 'weight'}),
                weight=map_dict['weight']
            )

        if 'inverse' in map_dict and not isinstance(map_dict['inverse'], bool):
            raise ValueError("Weight map 'inverse' must be a boolean")
        if map_dict.get('inverse', False):
            return InverseWeightMap(
                weight_map=self.from_dict({k: v for k, v in map_dict.items() if k != 'inverse'})
            )

        type = map_dict['type']
        map_dict = {k: v for k, v in map_dict.items() if k != 'type'}
        match type:
            case 'masked':
                if not map_dict.keys() <= {'maps'}:
                    raise ValueError("Masked weight map can only contain 'maps' key")
                if 'maps' not in map_dict or not isinstance(map_dict['maps'], list):
                    raise ValueError("Masked weight map must contain a 'maps' key with a list of map specifications")
                
                weight_maps = [self.from_dict(m) for m in map_dict['maps']]
                return MaskedWeightMap(weight_maps, self.mask_processor)
        
            case 'time_lapse':
                if not map_dict.keys() <= {'tau', 'weight', 'decay'}:
                    raise ValueError("Time-lapse weight map can only contain 'tau', 'weight', and 'decay' keys")
                if 'tau' in map_dict and not isinstance(map_dict['tau'], (int, float)):
                    raise ValueError("Time-lapse 'tau' must be a number")
                if 'weight' in map_dict and not isinstance(map_dict['weight'], dict):
                    raise ValueError("Time-lapse 'weight' must be a weight map dictionary")
                if 'decay' in map_dict and not isinstance(map_dict['decay'], dict):
                    raise ValueError("Time-lapse 'decay' must be a weight map dictionary")

                return TimeLapsedWeightMap(
                    reference_index=self.reference_index,
                    tau=map_dict.get('tau', DEFAULT_TIME_LAPSE_TAU) * self.frame_count,
                    weight_map=self.from_dict(map_dict.get('weight', {'type': 'constant'})),
                    decay_map=self.from_dict(map_dict.get('decay', {'type': 'constant'}))
                )
            
            case 'reference':
                if len(map_dict.keys()) > 0:
                    raise ValueError("Reference weight map cannot contain any extra keys")
                return ReferenceWeightMap(
                    reference_index=self.reference_index,
                    frame_count=self.frame_count
                )
            
            case _:
                if len(map_dict.keys()) > 0:
                    raise ValueError(f"Basic weight map of type '{type}' cannot contain any extra keys")
                if not type in self.basic_weight_map_classes:
                    raise ValueError(f"Unknown basic weight map type: {type}. Must be one of: {', '.join(self.basic_weight_map_classes.keys())}")
                return WeightMapGenerator.from_string(type)

def tunable_sigmoid(mask, sharpness):
    """
    Apply tunable sigmoid function to the mask with given k.
    https://math.stackexchange.com/questions/459872/adjustable-sigmoid-curve-s-curve-from-0-0-to-1-1
    """
    if abs(sharpness) > 1:
        raise ValueError("Invalid k value")
    sharpness = -sharpness * (1 - 1e-12)
    numerator = (sharpness - 1) * (2 * mask - 1)
    denominator = 2 * (4 * sharpness * torch.abs(mask - 0.5) - sharpness - 1)
    return numerator / denominator + 0.5