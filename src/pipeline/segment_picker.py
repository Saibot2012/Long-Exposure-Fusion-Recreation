'''
segment_picker.py - A simple GUI tool for annotating images with segmentation masks.
This script allows users to click on images to create positive and negative points,
and uses a SAM model to predict masks based on these points.
'''
import sys

import argparse
import time
import sys
import numpy as np
import torch
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor
from PySide6.QtCore import Qt, Slot, QThreadPool, Signal, QTimer, QEvent
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QAction, QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QStyle

from src.utils.ImageStore import ImageStore
import src.utils.utils as utils

MASKS_DIRNAME = Path("masks/")
BLENDED_DIRNAME = Path("blended/")

SAM2_CHECKPOINT_PATH = Path("external/sam2-sequential/checkpoints/sam2.1_hiera_tiny.pt")
MODEL_CONFIG_PATH = Path("configs/sam2.1/sam2.1_hiera_t.yaml")

# Point class for storing annotation info
class Point:
    def __init__(self, frame_index, object_id, x, y, is_positive):
        self.frame_index = frame_index
        self.object_id = object_id
        self.x = x
        self.y = y
        self.is_positive = is_positive

    def to_dict(self):
        return {
            'frame_index': self.frame_index,
            'object_id': self.object_id,
            'x': self.x,
            'y': self.y,
            'is_positive': self.is_positive
        }

    @staticmethod
    def from_dict(d):
        return Point(
            frame_index=d['frame_index'],
            object_id=d['object_id'],
            x=d['x'],
            y=d['y'],
            is_positive=d['is_positive']
        )

    def __repr__(self):
        return f"Point(frame={self.frame_index}, object={self.object_id}, pos=({self.x},{self.y}), positive={self.is_positive})"
    
def run_segment_picker(source: ImageStore) -> ImageStore:
    masks_cache = source.child(MASKS_DIRNAME)
    
    # Clear old masks to avoid stale size mismatches
    if masks_cache.path.exists():
        shutil.rmtree(masks_cache.path)
    
    app = QApplication(sys.argv)
    window = MainWindow(source)
    window.setWindowTitle("Segment Picker")
    window.show()
    app.exec()

    return source.child(MASKS_DIRNAME)

class MaskPredictor:
    def __init__(self, source: ImageStore):
        self.source = source
        self.masks_cache = source.child(MASKS_DIRNAME)
        self.blended_cache = source.cache.child(BLENDED_DIRNAME)

        self.points = None
        self._load_points()

        self.predictor = None
        self.inference_state = None
        self.valid_mask_index = None
        self._initialize_inference_state()

    def add_point(self, point):
        self.points.append(point)

        frame_points = [p for p in self.points if p.frame_index == point.frame_index and p.object_id == point.object_id]
        _, output_object_ids, output_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=point.frame_index,
            obj_id=point.object_id,
            points=np.array([[p.x, p.y] for p in frame_points], dtype=np.float32),
            labels=np.array([p.is_positive for p in frame_points], dtype=np.int32),
        )
        print(f"[DEBUG] Added point, {self.valid_mask_index}")

        masks = MaskPredictor._masks_from_logits(output_mask_logits, output_object_ids)

        self._save_masks(masks, point.frame_index)
        blended = self._blend_masks(
            masks,
            point.frame_index
        )
        self.blended_cache.save_image_at(blended, point.frame_index)

        self._reset_propagate_iterator(point.frame_index)

        self._save_points()

    def clear_points(self, frame_index: int = None, object_id: int = None):
        """Clear all points for the specified frame and object, then update the mask."""
        cleared = {
            (point.frame_index, point.object_id) for point in self.points if
            (frame_index is None or point.frame_index == frame_index) and
            (object_id is None or point.object_id == object_id)
        }
        if not len(cleared):
            return

        new_points = [
            point for point in self.points if
            (point.frame_index, point.object_id) not in cleared
        ]
        cleared_objects = {point.object_id for point in self.points} - {point.object_id for point in new_points}
        self.points = new_points

        for frame_index, object_id in reversed(sorted(cleared)):
            _, output_object_ids, output_mask_logits = self.predictor.clear_all_prompts_in_frame(
                inference_state=self.inference_state,
                frame_idx=frame_index,
                obj_id=object_id
            )
        frame_index = min(frame_index for frame_index, _ in cleared)

        # SAM2 behaves really weird when removing object. I couldn't find a better solution than this.
        cleared_object_indexes = []
        for i, object_id in enumerate(output_object_ids):
            if object_id in cleared_objects:
                cleared_object_indexes.append(i)
        output_object_ids = [output_object_ids[i] for i in range(len(output_object_ids)) if i not in cleared_object_indexes]
        output_mask_logits = [output_mask_logits[i] for i in range(len(output_mask_logits)) if i not in cleared_object_indexes]

        for object_id in cleared_objects:
            self.predictor.remove_object(
                inference_state=self.inference_state,
                obj_id=object_id
            )

        masks = MaskPredictor._masks_from_logits(output_mask_logits, output_object_ids)
        self._save_masks(masks, frame_index)
        blended = self._blend_masks(
            masks,
            frame_index
        )
        self.blended_cache.save_image_at(blended, frame_index)

        self._reset_propagate_iterator(frame_index)

        self._save_points()

    def get_frame(self, index: int) -> torch.Tensor:
        if index <= self.valid_mask_index:
            return self.blended_cache.load_image_at(index)

        return self._generate_frame(index)

    @staticmethod
    def _masks_from_logits(output_mask_logits, output_object_ids) -> list[torch.Tensor]:
        """Convert SAM2 output logits and object IDs to a list of binary masks."""
        if output_mask_logits is None or len(output_mask_logits) == 0:
            return []
        mask_count = max(output_object_ids) + 1
        masks = [None] * mask_count
        for i, obj_id in enumerate(output_object_ids):
            masks[obj_id] = (output_mask_logits[i] > 0.0).cpu().bool()
        return masks

    def _blend_masks(self, masks: list[torch.Tensor], frame_index: int) -> torch.Tensor:
        """Render masks with colors and blend them with the input image using PyTorch methods."""
        image = self.source.load_image_at(frame_index)

        # image: (H, W, 3) or (3, H, W) torch tensor, uint8 or float32
        # masks: list of (H, W) torch tensors (bool or 0/1)
        tab10 = torch.tensor([
            [255, 127, 14], [31, 119, 180], [44, 160, 44], [214, 39, 40],
            [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
            [188, 189, 34], [23, 190, 207]
        ], dtype=torch.float32, device=image.device) / 255.0  # Normalize to [0,1]

        # Ensure image is (H, W, 3) and float32
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        height, width = image.shape[:2]
        mask_image = torch.zeros((height, width, 3), dtype=torch.float32, device=image.device)

        for object_id, mask in enumerate(masks):
            if mask is None:
                continue

            mask = (mask > 0)
            color = tab10[object_id % len(tab10)]
            for channel in range(3):
                mask_image[..., channel] = torch.where(mask, color[channel], mask_image[..., channel])

        # Blend mask with image
        mask_binary = (mask_image != 0).any(dim=2)
        blended = image.clone()
        alpha = 0.6
        blended_mask = image * (1 - alpha) + mask_image * alpha
        # Only blend where mask is present
        blended[mask_binary] = blended_mask[mask_binary]

        return blended.permute(2, 0, 1)  # Return to (3, H, W)
    
    def _initialize_inference_state(self) -> None:
        self.predictor = build_sam2_video_predictor(str(MODEL_CONFIG_PATH), str(SAM2_CHECKPOINT_PATH), device='cuda')
        time.sleep(1)
        self.inference_state = self.predictor.init_state(video_path=str(self.source.path), async_loading_frames=True)

        points_per_frame = {}
        for point in self.points:
            points_per_frame.setdefault(point.frame_index, []).append(point)

        for points in points_per_frame.values():
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=points[0].frame_index,
                obj_id=points[0].object_id,
                points=np.array([[p.x, p.y] for p in points], dtype=np.float32),
                labels=np.array([p.is_positive for p in points], dtype=np.int32),
            )

        self._reset_propagate_iterator(self.masks_cache.get_entry("valid_mask_index", default=-1))
    
    def _generate_frame(self, index: int) -> np.ndarray:
        # Traverse propagate iterator until the desired index
        while (self.valid_mask_index < index):
            frame_index, output_object_ids, output_mask_logits = next(self.propagate_iterator)

            masks = MaskPredictor._masks_from_logits(output_mask_logits, output_object_ids)
            self._save_masks(masks, frame_index)
            blended = self._blend_masks(
                masks,
                frame_index
            )
            self.blended_cache.save_image_at(blended, frame_index)

            self.valid_mask_index = frame_index
            self.masks_cache.save_entry("valid_mask_index", self.valid_mask_index)
        return blended

    def _reset_propagate_iterator(self, valid_mask_index: int) -> None:
        self.valid_mask_index = valid_mask_index
        self.masks_cache.save_entry("valid_mask_index", self.valid_mask_index)

        # If no points have been added, return a dummy iterator of empty masks
        if not len(self.points):
            self.propagate_iterator = (
                (frame_index, [], []) for frame_index in range(valid_mask_index + 1, self.source.get_image_count())
            )
            return
        
        # Restart propagation from the first dirty frame
        self.propagate_iterator = self.predictor.propagate_in_video(self.inference_state, start_frame_idx=valid_mask_index + 1)

    def _save_masks(self, masks: list[torch.Tensor], frame_index: int) -> None:
        """
        Save a single 8-bit-per-pixel PNG mask for the given frame, where each pixel value is the object id (background=0).
        Args:
            masks: list of binary numpy arrays (1,H,W), one per object, as returned by sam2
            frame_index: int, frame number for filename
            destination: ImageStore to save the mask PNG into
        """
        if not len(masks):
            return

        # Each mask is (1,H,W), so squeeze to (H,W)
        height, width = masks[0].shape[-2:]
        mask_id_map = torch.zeros((height, width), dtype=torch.uint8) # background is 0
        for obj_id, mask in enumerate(masks, start=1):
            if mask is None:
                continue
            mask2d = torch.squeeze(mask) # (H,W)
            # Overlay: assign obj_id where mask is True
            mask_id_map[mask2d] = obj_id

        torch.save(mask_id_map, self.masks_cache.path / f"{frame_index:06d}.pt")
    
    def _save_points(self):
        self.masks_cache.save_entry("points", [p.to_dict() for p in self.points])

    def _load_points(self):
        self.points = [Point.from_dict(d) for d in self.masks_cache.get_entry("points", default=[])]

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

class ImageView(QGraphicsView):
    clicked = Signal(float, float, Qt.MouseButton)
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

    def mousePressEvent(self, ev: QMouseEvent):
        # Map click to scene coordinates
        pos = self.mapToScene(ev.position().toPoint())
        self.clicked.emit(pos.x(), pos.y(), ev.button())


class MainWindow(QMainWindow):
    def __init__(self, source: ImageStore):
        super().__init__()

        self.frame_index = 0
        self.current_frame = None
        self.image_count = source.get_image_count()
        self.thread_pool = QThreadPool()
        self.is_playing = False
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(1000 / 10)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.current_object_id = 0
        # --- SAM2 video predictor setup ---
        self.mask_predictor = MaskPredictor(source)

        self.scene = QGraphicsScene()
        self.view = ImageView(self.scene)
        self.setCentralWidget(self.view)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.height, self.width = source.load_image_at(0).shape[1:3] # (3, H, W)

        # --- Object selection buttons ---
        self.object_buttons = []
        self.object_count = 0
        self.object_button_layout = QVBoxLayout()
        self.object_button_layout.setAlignment(Qt.AlignTop)
        # Add "+" button to add objects
        self.plus_btn = QPushButton("+")
        self.plus_btn.setCheckable(False)
        self.plus_btn.clicked.connect(self.add_object_button)
        self.object_button_layout.addWidget(self.plus_btn)
        # Add starting object buttons
        for i in range(max(map(lambda p: p.object_id, self.mask_predictor.points), default=0) + 1):
            self.add_object_button()

        # Layout: image and frame label on the left, object buttons to the right
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.frame_label)
        left_layout.addWidget(self.view)
        # --- Playback controls below the image ---
        playback_layout = QHBoxLayout()
        style = QApplication.style()
        prev_btn = QPushButton()
        prev_btn.setIcon(style.standardIcon(QStyle.SP_MediaSkipBackward))
        play_btn = QPushButton()
        play_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        next_btn = QPushButton()
        next_btn.setIcon(style.standardIcon(QStyle.SP_MediaSkipForward))
        prev_btn.clicked.connect(lambda: self.previous_frame())
        play_btn.clicked.connect(self.toggle_play_pause)
        next_btn.clicked.connect(lambda: self.next_frame())
        self._play_btn = play_btn  # Store for toggling icon
        playback_layout.addWidget(prev_btn)
        playback_layout.addWidget(play_btn)
        playback_layout.addWidget(next_btn)
        left_layout.addLayout(playback_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(self.object_button_layout)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set a reasonable default window size
        self.default_width = 800
        self.default_height = 600
        self.resize(self.default_width, self.default_height)

        next_action = QAction("Next", self); next_action.setShortcut(Qt.Key_Right); next_action.setObjectName("next")
        previous_action = QAction("Prev", self); previous_action.setShortcut(Qt.Key_Left); previous_action.setObjectName("prev")
        clear_action = QAction("Clear", self); clear_action.setShortcut("C"); clear_action.setObjectName("clear")
        quit_action = QAction("Quit", self); quit_action.setShortcut(Qt.Key_Escape); quit_action.setObjectName("quit")
        for action in (next_action, previous_action, clear_action, quit_action):
            action.triggered.connect(self._on_action)
            self.addAction(action)

        self.view.clicked.connect(self.on_click)
        self._show_frame()
        # Defer fitInView until after window is shown
        QTimer.singleShot(0, lambda: self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio))

        # Install event filter for key events
        self.installEventFilter(self)

    def wheelEvent(self, event):
        # Scroll up: previous frame, Scroll down: next frame
        if event.angleDelta().y() > 0:
            self.previous_frame()
        elif event.angleDelta().y() < 0:
            self.next_frame()
        # Stop play mode on scroll
        if self.is_playing:
            self._set_is_playing(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def eventFilter(self, obj, event):
        # Toggle play mode on space, stop on any other key
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self._set_is_playing(not self.is_playing)
                return True
            elif self.is_playing:
                self._set_is_playing(False)
                return True
        return super().eventFilter(obj, event)

    def next_frame(self):
        if self.frame_index == self.image_count - 1:
            return False
        
        self.frame_index += 1
        self._show_frame()
        return True

    def previous_frame(self):
        if self.frame_index == 0:
            return False
        self.frame_index -= 1
        self._show_frame()
        return True

    def toggle_play_pause(self):
        self._set_is_playing(not self.is_playing)

    def add_object_button(self):
        index = len(self.object_buttons)
        btn = QPushButton(str(index))
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked, idx=index: self.set_current_object(idx))
        btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #3399ff;
                color: white;
            }
        """)
        # Insert before the "+" button (which is always last)
        self.object_button_layout.insertWidget(self.object_button_layout.count() - 1, btn)
        self.object_buttons.append(btn)

    def set_current_object(self, index: int):
        self.current_object_id = index
        # Update button checked state
        for i, button in enumerate(self.object_buttons):
            button.setChecked((i) == index)
        # Refresh points display for new object
        self._show_frame()

    @Slot(int, int, Qt.MouseButton)
    def on_click(self, x, y, button):
        self._set_is_playing(False)

        # Ensure coordinates are integers
        x = int(round(x))
        y = int(round(y))

        # Only allow points inside the image bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        point = Point(
            frame_index=self.frame_index,
            object_id=self.current_object_id,
            x=x,
            y=y,
            is_positive=(button == Qt.LeftButton)
        )
        self.mask_predictor.add_point(point)
        self._show_frame()

    def _set_is_playing(self, state: bool):
        if state == self.is_playing:
            return

        self.is_playing = state

        style = QApplication.style()
        if self.is_playing:
            self.play_timer.start()
            self._play_btn.setIcon(style.standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_timer.stop()
            self._play_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))

    def _on_play_tick(self):
        if not self.next_frame():
            self._set_is_playing(False)

    def _show_frame(self):
        self.frame_label.setText(f"Frame {self.frame_index+1} / {self.image_count}")

        # Only display the given image (already blended if needed)
        frame = self.mask_predictor.get_frame(self.frame_index)
        frame = utils.torch_image_to_numpy(frame)
        # Convert to QImage and QPixmap
        image_qt = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            3 * frame.shape[1],
            QImage.Format_BGR888
        )
        pixmap = QPixmap.fromImage(image_qt)
        self.scene.clear()
        self.scene.setSceneRect(0, 0, frame.shape[1], frame.shape[0])
        self.scene.addPixmap(pixmap)
        # Draw points as overlays, only for current object
        # Make point size relative to image size
        min_dim = min(frame.shape[0], frame.shape[1])
        point_radius = int(0.015 * min_dim)
        for point in self.mask_predictor.points:
            if point.frame_index == self.frame_index:
                if point.object_id == self.current_object_id:
                    color = QColor(0, 255, 0) if point.is_positive else QColor(255, 0, 0)
                else:
                    color = QColor(128, 128, 128)  # Dim color for other objects

                # Draw cross
                h_line = self.scene.addLine(
                    point.x - point_radius, point.y, 
                    point.x + point_radius, point.y, 
                    QPen(color, 0.003 * min_dim)
                )
                h_line.setZValue(1)
                v_line = self.scene.addLine(
                    point.x, point.y - point_radius, 
                    point.x, point.y + point_radius, 
                    QPen(color, 0.003 * min_dim)
                )
                v_line.setZValue(1)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _on_action(self):
        action = self.sender().objectName()
        if action == 'next':
            self.next_frame()
        elif action == 'prev':
            self.previous_frame()
        elif action == 'clear':
            self.mask_predictor.clear_points(self.frame_index, self.current_object_id)
            self._show_frame()
        elif action == 'quit':
            self.close()
        # Stop play mode on any action
        if self.is_playing:
            self._set_is_playing(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment Picker - Annotate images with segmentation masks.")
    parser.add_argument('folder', help='Folder containing images')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.folder, args.output)
    window.setWindowTitle("Segment Picker")
    window.show()
    sys.exit(app.exec())
