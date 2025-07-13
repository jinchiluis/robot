import sys
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox)


class QTCamera(QWidget):
    """USB Camera control widget with static object detection."""
    
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
        self.detected_regions = []
        
        # Display size settings
        self.fixed_display_size = None  # (width, height) or None for automatic
        
        # Object detection parameters
        self.color_threshold = 30  # Threshold for color differences
        self.min_area = 1000 #from 500  # Minimum area for a valid detection
        self.blur_size = 5  # Smaller blur for better detail preservation
        self.min_dimension = 40 #10  # Minimum width/height for valid detections
        self.object_detection_enabled = False  # Start with detection disabled
        
        # Color space for detection
        self.color_space = 'LAB'  # Options: 'RGB', 'LAB', 'HSV'
        
        # Noise reduction parameters
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
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Color threshold control
        controls_layout.addWidget(QLabel("Color Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(5, 100)
        self.threshold_spin.setValue(self.color_threshold)
        self.threshold_spin.valueChanged.connect(lambda v: setattr(self, 'color_threshold', v))
        controls_layout.addWidget(self.threshold_spin)
        
        # Min area control
        controls_layout.addWidget(QLabel("Min Area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 5000)
        self.min_area_spin.setValue(self.min_area)
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.valueChanged.connect(lambda v: setattr(self, 'min_area', v))
        controls_layout.addWidget(self.min_area_spin)
        
        # Color space selector
        controls_layout.addWidget(QLabel("Color Space:"))
        self.color_space_combo = QComboBox()
        self.color_space_combo.addItems(['LAB', 'RGB', 'HSV'])
        self.color_space_combo.setCurrentText(self.color_space)
        self.color_space_combo.currentTextChanged.connect(lambda s: setattr(self, 'color_space', s))
        controls_layout.addWidget(self.color_space_combo)
        
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
        
    def set_fixed_display_size(self, width, height):
        """
        Set a fixed display size for the camera widget.
        The image will be scaled to fit while preserving aspect ratio.
        
        Args:
            width: Fixed width in pixels
            height: Fixed height in pixels
        """
        self.fixed_display_size = (width, height)
        self.video_label.setFixedSize(width, height)
        self.video_label.setMinimumSize(width, height)
        self.video_label.setScaledContents(False)  # We'll handle scaling manually
        
    def init_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Camera not found or cannot be opened.")
            return
            #raise RuntimeError(f"Cannot open camera {self.camera_index}")

        # THE MAGIC SEQUENCE - DO NOT CHANGE ORDER!
        self.cap.set(cv2.CAP_PROP_FPS, 30.0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Add buffer size reduction
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        
        # Detect objects if enabled
        display_frame = frame.copy()
        if self.reference_frame is not None and self.object_detection_enabled:
            self.detected_regions = self.detect_objects(frame)
            
            # Draw detection boxes
            for (ox, oy, ow, oh) in self.detected_regions:
                cv2.rectangle(display_frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
            
            # Emit signal if objects detected
            if self.detected_regions:
                self.object_detected.emit(self.detected_regions)
        
        # Add status text
        status_text = f"Detection: {'ON' if self.object_detection_enabled else 'OFF'} | Color: {self.color_space}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Display frame
        self.display_frame(display_frame)
        
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
            # LAB color space provides perceptually uniform differences
            # Calculate Euclidean distance in LAB space
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            # Weight the channels (L is less important for object detection)
            weights = np.array([0.5, 1.0, 1.0])
            diff = diff * weights
            distance = np.sqrt(np.sum(diff**2, axis=2))
        elif self.color_space == 'HSV':
            # For HSV, focus more on Hue and Saturation changes
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            # Circular hue difference
            h_diff = diff[:,:,0]
            h_diff = np.minimum(h_diff, 180 - h_diff)
            # Weight: Hue is most important, then Saturation, then Value
            weights = np.array([2.0, 1.5, 0.5])
            diff[:,:,0] = h_diff
            diff = diff * weights
            distance = np.sqrt(np.sum(diff**2, axis=2))
        else:  # RGB
            # Simple Euclidean distance in RGB space
            diff = frame1.astype(np.float32) - frame2.astype(np.float32)
            distance = np.sqrt(np.sum(diff**2, axis=2))
            
        return distance.astype(np.uint8)
        
    def detect_objects(self, frame):
        """Detect objects in the frame compared to reference frame."""
        if not self.object_detection_enabled or self.reference_frame is None:
            return []
        
        # Apply slight blur to reduce noise
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        
        # Convert both frames to selected color space
        current_color = self.convert_color_space(blurred_frame)
        reference_color = self.convert_color_space(self.reference_frame)
        
        # Calculate color difference
        color_diff = self.calculate_color_difference(current_color, reference_color)
        
        # Apply threshold
        _, thresh = cv2.threshold(color_diff, self.color_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up noise
        if self.use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (self.morph_kernel_size, self.morph_kernel_size))
            # Close small gaps
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # Remove small noise
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect nearby regions
        dilate_kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, dilate_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
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
    
    def reset_reference_frame(self):
        """Reset the reference frame to current frame."""
        if self.current_frame is not None:
            # Apply same blur as in detection for consistency
            self.reference_frame = cv2.GaussianBlur(self.current_frame, 
                                                   (self.blur_size, self.blur_size), 0)
               
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
        
        if enabled and reset_ref_frame:
            # Reset reference frame when enabling detection
            self.reset_reference_frame()
    
    def capture_snapshot(self, include_boxes=True):
        """Capture a snapshot of the current frame."""
        if self.current_frame is None:
            return None
        
        snapshot = self.current_frame.copy()
        
        if include_boxes and self.detected_regions:
            for (x, y, w, h) in self.detected_regions:
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save with timestamp
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
        
        # If fixed display size is set, scale with aspect ratio preservation
        if self.fixed_display_size:
            target_width, target_height = self.fixed_display_size
            
            # Calculate scaling to fit within target size while preserving aspect ratio
            scale_x = target_width / w
            scale_y = target_height / h
            scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
            
            # Calculate actual scaled size
            scaled_width = int(w * scale)
            scaled_height = int(h * scale)
            
            # Scale the pixmap
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, 
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create a new pixmap with target size and black background
            final_pixmap = QPixmap(target_width, target_height)
            final_pixmap.fill(Qt.black)
            
            # Calculate position to center the scaled image
            x_offset = (target_width - scaled_width) // 2
            y_offset = (target_height - scaled_height) // 2
            
            # Draw the scaled image onto the final pixmap
            from PySide6.QtGui import QPainter
            painter = QPainter(final_pixmap)
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()
            
            self.video_label.setPixmap(final_pixmap)
        else:
            # Use original behavior if no fixed size is set
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
        # import traceback
        # traceback.print_stack()
        self.area = area
        self.reference_frame = None  # Force rebuild of reference frame

    def get_area(self):
        """Get the current area of interest."""
        return self.area

    def get_effective_resolution(self):
        """Get the effective resolution (area size if defined, otherwise camera resolution)."""
        if self.area:
            _, _, width, height = self.area
            return width, height
        else:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height


# class TestWindow(QMainWindow):
#     """Test application window."""
    
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("QTCamera Static Object Detection Test")
        
#         # Central widget
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
        
#         # Layout
#         layout = QVBoxLayout()
        
#         # Camera widget with area selection
#         # Example: Only show a 600x600 area starting at position (500, 5)
#         self.camera = QTCamera(camera_index=0, area=(500, 5, 600, 600))
#         self.camera.object_detected.connect(self.on_object_detected)
        
#         # Set fixed display size for consistent appearance
#         self.camera.set_fixed_display_size(600, 600)
        
#         layout.addWidget(self.camera)
        
#         # Snapshot button
#         snapshot_btn = QPushButton("Capture Snapshot")
#         snapshot_btn.clicked.connect(self.capture_snapshot)
#         layout.addWidget(snapshot_btn)
        
#         # Status label
#         self.status_label = QLabel("Ready - Press 'Enable Detection' to start")
#         layout.addWidget(self.status_label)
        
#         central_widget.setLayout(layout)
        
#     def on_object_detected(self, regions):
#         """Handle object detection signal."""
#         self.status_label.setText(f"Objects detected: {len(regions)} regions")
    
#     def capture_snapshot(self):
#         """Capture a snapshot."""
#         filename = self.camera.capture_snapshot()
#         if filename:
#             self.status_label.setText(f"Snapshot saved: {filename}")


# def main():
#     """Main function to run the test app."""
#     app = QApplication(sys.argv)
#     window = TestWindow()
#     window.show()
#     sys.exit(app.exec())


# if __name__ == "__main__":
#     main()