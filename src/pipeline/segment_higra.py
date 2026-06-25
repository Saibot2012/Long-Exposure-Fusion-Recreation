'''
segment_higra.py - A lightweight GUI tool for annotating a single reference frame
with a segmentation mask using Higra watershed hierarchy.

Left-click: mark as object (region to fuse)
Right-click: mark as background (region to use reference frame)
Ctrl+Z: undo
Close window to confirm and save masks.
'''

import argparse
import urllib.request as request
import numpy as np
import torch
import higra as hg
import cv2 as cv
import matplotlib
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.widgets
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm
import shutil

from src.utils.ImageStore import ImageStore

MASKS_DIRNAME = Path("masks/")


def compute_alpha_frame(frame):
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    G = np.abs(img[:, 1:] - img[:, :-1])

    counts, bin_edges = np.histogram(G.flatten(), bins=50, range=(0, 0.1))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = counts > 0
    x = x[mask]
    counts = counts[mask]

    y = counts / counts.sum()

    slope, intercept = np.polyfit(x, np.log(y), 1)

    return -slope


def find_sharpest_frame(source: ImageStore) -> int:
    best_index = None
    best_alpha = -1
    for index, filename in source.get_indexed_image_filenames().items():
        image = source.load_image(filename)
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
        alpha = compute_alpha_frame(img_np)
        if alpha > best_alpha:
            best_alpha = alpha
            best_index = index
    print(f"[DEBUG] source path: {source.path}")
    print(f"[DEBUG] loaded image shape: {image.shape}, filename: {filename}")
    print(f"[INFO] Sharpest frame: {best_index:06d} with alpha={best_alpha:.2f}")
    return best_index


class SkySegmenter:
    def __init__(self, image: np.ndarray):
        """
        Args:
            image: (H, W, 3) uint8 numpy array in BGR format
        """
        exec(request.urlopen("https://github.com/higra/Higra-Notebooks/raw/master/utils.py").read(), globals())

        size = image.shape
        image = resize(image, (int(size[0] * 0.65), int(size[1] * 0.65)), mode="reflect")
        self.image = image.astype(np.float32)
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.size = self.image.shape[:2]
        print(f"[DEBUG] SkySegmenter self.size: {self.size}")
        self.history = []

        detector = cv.ximgproc.createStructuredEdgeDetection(get_sed_model_file())
        gradient_image = detector.detectEdges(self.image)
        gradient_image = cv.GaussianBlur(gradient_image, (5, 5), 1.0)

        graph = hg.get_4_adjacency_graph(self.size)
        edge_weights = hg.weight_graph(graph, gradient_image, hg.WeightFunction.mean)
        self.tree, self.altitudes = hg.watershed_hierarchy_by_volume(graph, edge_weights)

        image_alpha = np.pad(self.image, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=1)
        self.markers = np.zeros_like(image_alpha)

        sm = hg.graph_4_adjacency_2_khalimsky(graph, hg.saliency(self.tree, self.altitudes)) ** 0.5
        sm = sm[1::2, 1::2]
        sm = np.pad(sm, ((0, 1), (0, 1)), mode="edge")
        sm = 1 - sm / np.max(sm)
        sm = np.dstack([sm] * 3)
        sm = np.pad(sm, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=1)

        self.base_image = np.hstack((image_alpha, sm))

    def get_mask(self) -> np.ndarray:
        """Returns a (H, W) binary numpy array — True where object (region to fuse)."""
        return hg.binary_labelisation_from_markers(self.tree, self.markers[:, :, 1], self.markers[:, :, 0])

    def _redraw(self):
        self.ax.clear()
        result = self.get_mask()
        self.ax.imshow(self.base_image, interpolation="none")
        self.ax.imshow(np.hstack((self.markers, np.dstack((np.copy(self.image), result)))), interpolation="none")
        self.ax.set_title("Left-click: fuse region | Right-click: use reference | Ctrl+Z: undo")
        self.fig.canvas.draw()

    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata) % self.size[1]
        y = int(event.ydata)
        self.history.append(self.markers.copy())
        r = int(self.slider.val)
        if event.button == 1:
            self.markers[max(0, y - r):y + r, max(0, x - r):x + r, :] = (0, 1, 0, 1)
        elif event.button == 3:
            self.markers[max(0, y - r):y + r, max(0, x - r):x + r, :] = (1, 0, 0, 1)
        self._redraw()

    def _onkey(self, event):
        if event.key == "ctrl+z" and self.history:
            self.markers[:] = self.history.pop()
            self._redraw()

    def run(self) -> np.ndarray:
        """Opens the GUI and blocks until the window is closed. Returns the binary mask."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.subplots_adjust(bottom=0.2)

        ax_slider = self.fig.add_axes([0.2, 0.0, 0.6, 0.03])
        self.slider = matplotlib.widgets.Slider(ax_slider, "Brush size", 1, 50, valinit=10)
        self.ax.imshow(self.base_image, interpolation="none")
        self.ax.set_title("Left-click: fuse region | Right-click: use reference | Ctrl+Z: undo")
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        print("[INFO] Window open — left-click to mark fusion regions, right-click for reference regions. Close to confirm.")
        plt.ioff()
        plt.show()
        return self.get_mask()


def run_higra_segmenter(source: ImageStore, reference_index: int = None) -> ImageStore:
    """
    Opens the Higra segmentation GUI on the reference frame, then saves the
    resulting mask as per-frame .pt files compatible with MaskLoader.

    Args:
        source: ImageStore containing the aligned/cropped frames
        reference_index: index of the reference frame to annotate on.
                         If None, the sharpest frame is used automatically.

    Returns:
        ImageStore pointing to the masks cache directory
    """
    masks_cache = source.child(MASKS_DIRNAME)
    print(f"[DEBUG] indexed filenames: {list(source.get_indexed_image_filenames().items())[:5]}")
    if reference_index is None:
        reference_index = find_sharpest_frame(source)

    # Load reference frame and convert to (H, W, 3) uint8 numpy BGR
    ref_tensor = source.load_image_at(reference_index)
    print(f"[DEBUG] ref_tensor shape: {ref_tensor.shape}")
    ref_np = (ref_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ref_np_bgr = cv.cvtColor(ref_np, cv.COLOR_RGB2BGR)

    # Run GUI
    print(f"[DEBUG] ref_np_bgr shape: {ref_np_bgr.shape}")
    segmenter = SkySegmenter(ref_np_bgr)
    mask = segmenter.run()

    # Resize back to original resolution
    original_h, original_w = ref_np.shape[:2]
    print(f"mask shape before resize: {mask.shape}")
    mask = resize(mask.astype(np.float32), (original_h, original_w), mode="reflect")
    mask = mask > 0.5

    # Build mask_id_map: 0=background, 1=object (region to fuse)
    mask_id_map = torch.zeros((original_h, original_w), dtype=torch.uint8)
    mask_id_map[torch.from_numpy(mask)] = 1

    # Save same mask for every frame
    # Clear old masks first
    if masks_cache.path.exists():
        shutil.rmtree(masks_cache.path)
    masks_cache.path.mkdir(parents=True, exist_ok=True)

    # Save mask only for frames that exist
    indexed_filenames = source.get_indexed_image_filenames()
    for i in tqdm(indexed_filenames.keys(), desc="Saving Higra masks"):
        torch.save(mask_id_map, masks_cache.path / f"{i:06d}.pt")

    print(f"[INFO] Higra masks saved to {masks_cache.path} for {len(indexed_filenames)} frames.")
    return masks_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Higra segmenter — annotate a reference frame to select fusion regions.")
    parser.add_argument("source", type=ImageStore, help="Directory containing aligned/cropped images.")
    parser.add_argument("--reference", type=int, default=None, help="Index of reference frame. If not provided, sharpest frame is used automatically.")
    args = parser.parse_args()

    run_higra_segmenter(args.source, args.reference)