import sys
import cv2
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSpinBox)


class QTCamera(QWidget):
    """USB Camera control widget with motion detection."""
    
    motion_detected = Signal(list)  # Emits list of detected regions
    
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
        self.motion_regions = []
        
        # Motion detection parameters
        self.threshold = 30
        self.min_area = 500
        self.blur_size = 21
        self.filter_thin_lines = True  # Can be disabled if needed
        self.min_dimension = 5  # Minimum width/height for valid detections
        
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
        
        # Threshold control
        controls_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(10, 100)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(lambda v: setattr(self, 'threshold', v))
        controls_layout.addWidget(self.threshold_spin)
        
        # Min area control
        controls_layout.addWidget(QLabel("Min Area:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 5000)
        self.min_area_spin.setValue(self.min_area)
        self.min_area_spin.setSingleStep(100)
        self.min_area_spin.valueChanged.connect(lambda v: setattr(self, 'min_area', v))
        controls_layout.addWidget(self.min_area_spin)
        
        # Reset button
        self.reset_btn = QPushButton("Reset Reference")
        self.reset_btn.clicked.connect(self.reset_reference_frame)
        controls_layout.addWidget(self.reset_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
    def init_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms for ~33 FPS
        
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
        
        # Detect motion
        if self.reference_frame is not None:
            self.motion_regions = self.detect_motion(frame)
            
            # Draw motion boxes
            for (mx, my, mw, mh) in self.motion_regions:
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
        
        # Display frame
        self.display_frame(frame)
        
    def filter_thin_line_detections(self, motion_regions):
        """
        Filter out thin line detections that are likely edge artifacts.
        Can be disabled by setting self.filter_thin_lines = False
        """
        if not self.filter_thin_lines:
            return motion_regions
        
        filtered_regions = []
        
        for (x, y, w, h) in motion_regions:
            # Check minimum dimensions
            if w < self.min_dimension or h < self.min_dimension:
                continue
            
            # Check aspect ratio - reject very thin lines
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 10:  # Very elongated shapes are likely lines
                continue
            
            # Passed all checks
            filtered_regions.append((x, y, w, h))
        
        return filtered_regions
    
    def detect_motion(self, frame):
        """Detect motion in the frame compared to reference frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Compute difference
        frame_delta = cv2.absdiff(self.reference_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Apply morphological opening to remove thin lines
        if self.filter_thin_lines:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))
        
        # Filter out thin lines
        motion_regions = self.filter_thin_line_detections(motion_regions)
        
        if motion_regions:
            self.motion_detected.emit(motion_regions)
        
        return motion_regions
    
    def reset_reference_frame(self):
        """Reset the reference frame to current frame."""
        if self.current_frame is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            self.reference_frame = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
    
    def capture_snapshot(self, include_boxes=True):
        """Capture a snapshot of the current frame."""
        if self.current_frame is None:
            return None
        
        snapshot = self.current_frame.copy()
        
        if include_boxes and self.motion_regions:
            for (x, y, w, h) in self.motion_regions:
                cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        cv2.imwrite(filename, snapshot)
        
        return filename
    
    def display_frame(self, frame):
        """Convert and display frame in Qt."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


class TestWindow(QMainWindow):
    """Test application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QTCamera Test App")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        
        # Camera widget with area selection
        # Example: Only show a 400x300 area starting at position (100, 50)
        self.camera = QTCamera(camera_index=0, area=(500, 10, 600, 600))
        self.camera.motion_detected.connect(self.on_motion_detected)
        layout.addWidget(self.camera)
        
        # Snapshot button
        snapshot_btn = QPushButton("Capture Snapshot")
        snapshot_btn.clicked.connect(self.capture_snapshot)
        layout.addWidget(snapshot_btn)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        central_widget.setLayout(layout)
        
    def on_motion_detected(self, regions):
        """Handle motion detection signal."""
        self.status_label.setText(f"Motion detected: {len(regions)} regions")
    
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