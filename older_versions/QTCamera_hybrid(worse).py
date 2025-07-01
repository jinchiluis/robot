import sys
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox,
                               QCheckBox, QGroupBox)


class QTCamera(QWidget):
    """USB Camera control widget with hybrid motion + static object detection."""
    
    object_detected = Signal(list)  # Emits list of detected regions
    
    def __init__(self, camera_index=0, area=None, parent=None):
        """
        Initialize the camera widget.
        
        Args:
            camera_index: Camera device index (default: 0)
            area: Tuple of (x, y, width, height) to define the area of interest
            parent: Parent widget
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self.area = area  # (x, y, width, height)
        self.cap = None
        self.timer = None
        self.reference_frame = None
        self.current_frame = None
        self.previous_frame = None
        self.detected_regions = []
        
        # Display size settings
        self.fixed_display_size = None  # (width, height) or None for automatic
        
        # Detection mode
        self.detection_mode = 'hybrid'  # 'static', 'motion', 'hybrid'
        
        # Static detection parameters
        self.color_threshold = 30  # Threshold for color differences
        self.min_area = 500
        self.blur_size = 5
        self.min_dimension = 10
        self.color_space = 'LAB'
        
        # Motion detection parameters
        self.motion_threshold = 25
        self.motion_min_area = 500
        self.motion_blur_size = 21
        self.motion_history_frames = 3
        self.motion_history = []
        
        # Hybrid mode parameters
        self.motion_weight = 0.6  # How much to weight motion vs static
        self.edge_detection_enabled = True
        self.edge_threshold = 50
        
        # Common parameters
        self.object_detection_enabled = False
        self.use_morphology = True
        self.morph_kernel_size = 3
        
        # UI setup
        self.setup_ui()
        self.init_camera()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black")
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)
        
        # Detection mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detection Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Hybrid', 'Static Only', 'Motion Only'])
        self.mode_combo.setCurrentText('Hybrid')
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # Static detection controls
        static_group = QGroupBox("Static Detection Settings")
        static_layout = QHBoxLayout()
        
        static_layout.addWidget(QLabel("Color Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(5, 100)
        self.threshold_spin.setValue(self.color_threshold)
        self.threshold_spin.valueChanged.connect(lambda v: setattr(self, 'color_threshold', v))
        static_layout.addWidget(self.threshold_spin)
        
        static_layout.addWidget(QLabel("Color Space:"))
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(['LAB', 'RGB', 'HSV'])
        self.color_space_combo.setCurrentText(self.color_space)
        self.color_space_combo.currentTextChanged.connect(lambda s: setattr(self, 'color_space', s))
        static_layout.addWidget(self.color_space_combo)
        
        static_group.setLayout(static_layout)
        layout.addWidget(static_group)
        
        # Motion detection controls
        motion_group = QGroupBox("Motion Detection Settings")
        motion_layout = QHBoxLayout()
        
        motion_layout.addWidget(QLabel("Motion Threshold:"))
        self.motion_threshold_spin = QSpinBox()
        self.motion_threshold_spin.setRange(5, 100)
        self.motion_threshold_spin.setValue(self.motion_threshold)
        self.motion_threshold_spin.valueChanged.connect(lambda v: setattr(self, 'motion_threshold', v))
        motion_layout.addWidget(self.motion_threshold_spin)
        
        motion_layout.addWidget(QLabel("History Frames:"))
        self.history_spin = QSpinBox()
        self.history_spin.setRange(1, 10)
        self.history_spin.setValue(self.motion_history_frames)
        self.history_spin.valueChanged.connect(lambda v: setattr(self, 'motion_history_frames', v))
        motion_layout.addWidget(self.history_spin)
        
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)
        
        # Hybrid mode controls
        hybrid_group = QGroupBox("Hybrid Mode Settings")
        hybrid_layout = QHBoxLayout()
        
        hybrid_layout.addWidget(QLabel("Motion Weight:"))
        self.motion_weight_spin = QSpinBox()
        self.motion_weight_spin.setRange(0, 100)
        self.motion_weight_spin.setValue(int(self.motion_weight * 100))
        self.motion_weight_spin.valueChanged.connect(lambda v: setattr(self, 'motion_weight', v / 100.0))
        self.motion_weight_spin.setSuffix("%")
        hybrid_layout.addWidget(self.motion_weight_spin)
        
        self.edge_checkbox = QCheckBox("Use Edge Detection")
        self.edge_checkbox.setChecked(self.edge_detection_enabled)
        self.edge_checkbox.toggled.connect(lambda c: setattr(self, 'edge_detection_enabled', c))
        hybrid_layout.addWidget(self.edge_checkbox)
        
        hybrid_layout.addWidget(QLabel("Edge Threshold:"))
        self.edge_threshold_spin = QSpinBox()
        self.edge_threshold_spin.setRange(10, 200)
        self.edge_threshold_spin.setValue(self.edge_threshold)
        self.edge_threshold_spin.valueChanged.connect(lambda v: setattr(self, 'edge_threshold', v))
        hybrid_layout.addWidget(self.edge_threshold_spin)
        
        hybrid_group.setLayout(hybrid_layout)
        layout.addWidget(hybrid_group)
        
        # Common controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Min Area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 5000)
        self.min_area_spin.setValue(self.min_area)
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.valueChanged.connect(self.on_min_area_changed)
        controls_layout.addWidget(self.min_area_spin)
        
        # Reset button
        self.reset_btn = QPushButton("Reset Reference")
        self.reset_btn.clicked.connect(self.reset_reference_frame)
        controls_layout.addWidget(self.reset_btn)
        
        # Toggle detection button
        self.toggle_btn = QPushButton("Enable Detection")
        self.toggle_btn.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.toggle_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
    def on_mode_changed(self, mode_text):
        """Handle detection mode change."""
        mode_map = {
            'Hybrid': 'hybrid',
            'Static Only': 'static',
            'Motion Only': 'motion'
        }
        self.detection_mode = mode_map.get(mode_text, 'hybrid')
        self.reset_reference_frame()
        print(f"Detection mode changed to: {self.detection_mode}")
        
    def on_min_area_changed(self, value):
        """Update min area for both detection types."""
        self.min_area = value
        self.motion_min_area = value
        
    def set_fixed_display_size(self, width, height):
        """Set a fixed display size for the camera widget."""
        self.fixed_display_size = (width, height)
        self.video_label.setFixedSize(width, height)
        self.video_label.setMinimumSize(width, height)
        self.video_label.setScaledContents(False)
        
    def init_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms for ~33 FPS
        
    def toggle_detection(self):
        """Toggle object detection on/off."""
        self.set_object_detection_enabled(not self.object_detection_enabled)
        
    def update_frame(self):
        """Capture and process a new frame."""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Crop to area of interest if specified
        if self.area:
            x, y, w, h = self.area
            frame = frame[y:y+h, x:x+w]
        
        self.current_frame = frame.copy()
        
        # Initialize reference frame
        if self.reference_frame is None:
            self.reset_reference_frame()
        
        # Initialize previous frame for motion detection
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.previous_frame = cv2.GaussianBlur(self.previous_frame, 
                                                  (self.motion_blur_size, self.motion_blur_size), 0)
        
        # Detect objects if enabled
        display_frame = frame.copy()
        if self.reference_frame is not None and self.object_detection_enabled:
            if self.detection_mode == 'static':
                self.detected_regions = self.detect_static_objects(frame)
            elif self.detection_mode == 'motion':
                self.detected_regions = self.detect_motion_objects(frame)
            else:  # hybrid
                self.detected_regions = self.detect_hybrid_objects(frame)
            
            # Draw detection boxes with mode-specific colors
            box_color = (0, 255, 0)  # Green for static
            if self.detection_mode == 'motion':
                box_color = (255, 0, 0)  # Blue for motion
            elif self.detection_mode == 'hybrid':
                box_color = (255, 255, 0)  # Cyan for hybrid
                
            for (ox, oy, ow, oh) in self.detected_regions:
                cv2.rectangle(display_frame, (ox, oy), (ox + ow, oy + oh), box_color, 2)
            
            # Emit signal if objects detected
            if self.detected_regions:
                self.object_detected.emit(self.detected_regions)
        
        # Update previous frame for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_frame = cv2.GaussianBlur(gray, (self.motion_blur_size, self.motion_blur_size), 0)
        
        # Add status text
        status_text = f"Mode: {self.detection_mode.upper()} | Detection: {'ON' if self.object_detection_enabled else 'OFF'}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display frame
        self.display_frame(display_frame)
        
    def detect_hybrid_objects(self, frame):
        """Detect objects using hybrid approach combining motion and static detection."""
        # Get masks from both detection methods
        static_mask = self.get_static_mask(frame)
        motion_mask = self.get_motion_mask(frame)
        
        # Combine masks with weighting
        combined_mask = cv2.addWeighted(motion_mask.astype(np.float32), self.motion_weight,
                                       static_mask.astype(np.float32), 1.0 - self.motion_weight,
                                       0).astype(np.uint8)
        
        # Add edge detection for better boundary detection
        if self.edge_detection_enabled:
            edges = self.get_edge_mask(frame)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
        
        # Apply threshold
        _, thresh = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (self.morph_kernel_size, self.morph_kernel_size))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        return self.filter_contours(contours)
        
    def get_static_mask(self, frame):
        """Get mask from static object detection."""
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        current_color = self.convert_color_space(blurred_frame)
        reference_color = self.convert_color_space(self.reference_frame)
        color_diff = self.calculate_color_difference(current_color, reference_color)
        _, mask = cv2.threshold(color_diff, self.color_threshold, 255, cv2.THRESH_BINARY)
        return mask
        
    def get_motion_mask(self, frame):
        """Get mask from motion detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.motion_blur_size, self.motion_blur_size), 0)
        
        # Calculate frame difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        
        # Update motion history
        self.motion_history.append(frame_delta)
        if len(self.motion_history) > self.motion_history_frames:
            self.motion_history.pop(0)
        
        # Accumulate motion over history
        if len(self.motion_history) > 1:
            accumulated_motion = np.zeros_like(frame_delta, dtype=np.float32)
            for i, delta in enumerate(self.motion_history):
                weight = (i + 1) / len(self.motion_history)  # Recent frames have more weight
                accumulated_motion += delta.astype(np.float32) * weight
            accumulated_motion = (accumulated_motion / len(self.motion_history)).astype(np.uint8)
        else:
            accumulated_motion = frame_delta
        
        _, mask = cv2.threshold(accumulated_motion, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return mask
        
    def get_edge_mask(self, frame):
        """Get mask from edge detection for better boundary detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Dilate edges to make them thicker
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
        
    def detect_static_objects(self, frame):
        """Original static object detection."""
        mask = self.get_static_mask(frame)
        
        # Apply morphological operations
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (self.morph_kernel_size, self.morph_kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect nearby regions
        dilate_kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        return self.filter_contours(contours)
        
    def detect_motion_objects(self, frame):
        """Pure motion-based object detection."""
        mask = self.get_motion_mask(frame)
        
        # Apply morphological operations
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (self.morph_kernel_size * 2, self.morph_kernel_size * 2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        return self.filter_contours(contours, self.motion_min_area)
        
    def filter_contours(self, contours, min_area=None):
        """Filter contours based on area and dimensions."""
        if min_area is None:
            min_area = self.min_area
            
        detected_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Filter by minimum dimensions
            if w < self.min_dimension or h < self.min_dimension:
                continue
            
            # Filter extreme aspect ratios (likely noise)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 15:
                continue
            
            detected_regions.append((x, y, w, h))
        
        return detected_regions
        
    def convert_color_space(self, frame):
        """Convert frame to the selected color space."""
        if self.color_space == 'LAB':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'HSV':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:  # RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    def calculate_color_difference(self, frame1, frame2):
        """Calculate color difference between two frames."""
        if self.color_space == 'LAB':
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            weights = np.array([0.5, 1.0, 1.0])
            diff = diff * weights
            distance = np.sqrt(np.sum(diff**2, axis=2))
        elif self.color_space == 'HSV':
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            h_diff = diff[:,:,0]
            h_diff = np.minimum(h_diff, 180 - h_diff)
            weights = np.array([2.0, 1.5, 0.5])
            diff[:,:,0] = h_diff
            diff = diff * weights
            distance = np.sqrt(np.sum(diff**2, axis=2))
        else:  # RGB
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            distance = np.sqrt(np.sum(diff**2, axis=2))
            
        return distance.astype(np.uint8)
    
    def reset_reference_frame(self):
        """Reset the reference frame to current frame."""
        if self.current_frame is not None:
            self.reference_frame = cv2.GaussianBlur(self.current_frame, 
                                                   (self.blur_size, self.blur_size), 0)
            self.motion_history.clear()  # Clear motion history
            print(f"Reference frame reset at {datetime.now().strftime('%H:%M:%S')}")
    
    def get_current_frame(self, include_boxes=True):
        """Get the current frame with optional detection boxes."""
        if self.current_frame is None:
            return None
        
        frame = self.current_frame.copy()
        
        if include_boxes and self.detected_regions:
            for (x, y, w, h) in self.detected_regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame
    
    def set_object_detection_enabled(self, enabled, reset_ref_frame=True):
        """Enable or disable object detection."""
        self.object_detection_enabled = enabled
        self.toggle_btn.setText("Disable Detection" if enabled else "Enable Detection")
        print(f"Object detection {'enabled' if enabled else 'disabled'}")
        
        if enabled and reset_ref_frame:
            self.reset_reference_frame()
    
    def capture_snapshot(self, include_boxes=True):
        """Capture a snapshot of the current frame."""
        if self.current_frame is None:
            return None
        
        snapshot = self.current_frame.copy()
        
        if include_boxes and self.detected_regions:
            for (x, y, w, h) in self.detected_regions:
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        cv2.imwrite(filename, snapshot)
        
        return filename
    
    def display_frame(self, frame):
        """Convert and display frame in Qt with aspect ratio preservation."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        if self.fixed_display_size:
            target_width, target_height = self.fixed_display_size
            
            scale_x = target_width / w
            scale_y = target_height / h
            scale = min(scale_x, scale_y)
            
            scaled_width = int(w * scale)
            scaled_height = int(h * scale)
            
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, 
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            final_pixmap = QPixmap(target_width, target_height)
            final_pixmap.fill(Qt.black)
            
            x_offset = (target_width - scaled_width) // 2
            y_offset = (target_height - scaled_height) // 2
            
            painter = QPainter(final_pixmap)
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()
            
            self.video_label.setPixmap(final_pixmap)
        else:
            self.video_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

    def set_area(self, area):
        """Set the area of interest and reset reference frame."""
        self.area = area
        self.reference_frame = None

    def get_area(self):
        """Get the current area of interest."""
        return self.area

    def get_effective_resolution(self):
        """Get the effective resolution."""
        if self.area:
            _, _, width, height = self.area
            return width, height
        else:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height


class TestWindow(QMainWindow):
    """Test application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTCamera Hybrid Detection Test")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        
        # Camera widget
        self.camera = QTCamera(camera_index=0, area=(500, 5, 600, 600))
        self.camera.object_detected.connect(self.on_object_detected)
        self.camera.set_fixed_display_size(600, 600)
        
        layout.addWidget(self.camera)
        
        # Snapshot button
        snapshot_btn = QPushButton("Capture Snapshot")
        snapshot_btn.clicked.connect(self.capture_snapshot)
        layout.addWidget(snapshot_btn)
        
        # Status label
        self.status_label = QLabel("Ready - Press 'Enable Detection' to start")
        layout.addWidget(self.status_label)
        
        central_widget.setLayout(layout)
        
    def on_object_detected(self, regions):
        """Handle object detection signal."""
        mode = self.camera.detection_mode.upper()
        self.status_label.setText(f"[{mode}] Objects detected: {len(regions)} regions")
    
    def capture_snapshot(self):
        """Capture a snapshot."""
        filename = self.camera.capture_snapshot()
        if filename:
            self.status_label.setText(f"Snapshot saved: {filename}")


def main():
    """Main function to run the test app."""
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()