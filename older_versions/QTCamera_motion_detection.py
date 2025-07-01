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
        
        # Display size settings
        self.fixed_display_size = None  # (width, height) or None for automatic
        
        # Motion detection parameters
        self.threshold = 30
        self.min_area = 500
        self.blur_size = 21
        self.filter_thin_lines = True  # Can be disabled if needed
        self.min_dimension = 5  # Minimum width/height for valid detections
        self.motion_detection_enabled = False  # Start with motion detection disabled
        
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
        
        # Detect motion only if enabled
        if self.reference_frame is not None and self.motion_detection_enabled:
            self.motion_regions = self.detect_motion(frame)
            
            # Draw motion boxes
            for (mx, my, mw, mh) in self.motion_regions:
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            
        # NEW: Always show static differences when NOT detecting motion
        elif self.reference_frame is not None and not self.motion_detection_enabled:
            # Use the same motion detection logic but don't emit signals
            static_regions = self.get_frame_differences(frame)
            print(f"Found {len(static_regions)} static regions")  # Debug line
            
            # Draw rectangles with different color for debugging
            for i, (sx, sy, sw, sh) in enumerate(static_regions):
                print(f"  Drawing rectangle {i}: ({sx}, {sy}, {sw}, {sh})")
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                # Add text to make sure it's visible
                cv2.putText(frame, f"R{i}", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add debug text to frame
        debug_text = f"Motion: {'ON' if self.motion_detection_enabled else 'OFF'}"
        cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Display frame
        self.display_frame(frame)
        
    def get_frame_differences(self, frame):
       """Get differences without checking motion_detection_enabled."""
       import os
       from datetime import datetime
       
       os.makedirs("debug_images", exist_ok=True)
       
       # Check if we have reference frame
       if not hasattr(self, 'reference_frame_unblurred') or self.reference_frame_unblurred is None:
           print("DEBUG: No reference_frame_unblurred found!")
           return []
       
       # Add timestamp to debug
       timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
       print(f"\n[{timestamp}] DEBUG: get_frame_differences called")
       
       # Convert current frame to grayscale (unblurred)
       gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
       # Debug: Save both frames with visible difference indicator
       debug_ref = self.reference_frame_unblurred.copy()
       debug_curr = gray_current.copy()
       
       # Add text labels to distinguish them
       cv2.putText(debug_ref, "REFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
       cv2.putText(debug_curr, "CURRENT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
       
       cv2.imwrite("debug_images/01_current_gray.png", debug_curr)
       cv2.imwrite("debug_images/02_reference_unblurred.png", debug_ref)
       
       print(f"DEBUG: Current gray shape: {gray_current.shape}, dtype: {gray_current.dtype}")
       print(f"DEBUG: Reference shape: {self.reference_frame_unblurred.shape}, dtype: {self.reference_frame_unblurred.dtype}")
       
       # Check if frames are identical (this would explain no difference)
       if np.array_equal(self.reference_frame_unblurred, gray_current):
           print("WARNING: Reference and current frames are IDENTICAL!")
       
       # Compute absolute difference between unblurred frames
       frame_delta = cv2.absdiff(gray_current, self.reference_frame_unblurred)
       
       # Enhance the difference for visualization
       frame_delta_enhanced = cv2.normalize(frame_delta, None, 0, 255, cv2.NORM_MINMAX)
       cv2.imwrite("debug_images/03_frame_delta.png", frame_delta)
       cv2.imwrite("debug_images/03b_frame_delta_enhanced.png", frame_delta_enhanced)
       
       print(f"DEBUG: frame_delta min: {frame_delta.min()}, max: {frame_delta.max()}, mean: {frame_delta.mean():.2f}")
       print(f"DEBUG: Number of pixels with difference > 0: {np.sum(frame_delta > 0)}")
       print(f"DEBUG: Number of pixels with difference > {self.threshold}: {np.sum(frame_delta > self.threshold)}")
       
       # Apply threshold
       _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
       cv2.imwrite("debug_images/04_threshold.png", thresh)
       print(f"DEBUG: Number of white pixels after threshold: {np.sum(thresh == 255)}")
       
       # Rest of the method stays the same...
       # Dilate to connect nearby differences
       kernel_dilate = np.ones((5,5), np.uint8)
       thresh_dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)
       cv2.imwrite("debug_images/05_dilated.png", thresh_dilated)
       
       # Apply morphological opening to remove noise (if enabled)
       if self.filter_thin_lines:
           kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
           thresh_cleaned = cv2.morphologyEx(thresh_dilated, cv2.MORPH_OPEN, kernel_morph)
           cv2.imwrite("debug_images/06_morphology_cleaned.png", thresh_cleaned)
       else:
           thresh_cleaned = thresh_dilated
       
       # Find contours
       contours, _ = cv2.findContours(thresh_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       print(f"DEBUG: Number of contours found: {len(contours)}")
       
       # Draw all contours for debugging
       contour_img = np.zeros_like(thresh_cleaned)
       cv2.drawContours(contour_img, contours, -1, 255, 2)
       cv2.imwrite("debug_images/07_all_contours.png", contour_img)
       
       # Process contours
       motion_regions = []
       for i, contour in enumerate(contours):
           area = cv2.contourArea(contour)
           (x, y, w, h) = cv2.boundingRect(contour)
           
           if area >= self.min_area:
               print(f"DEBUG: Contour {i}: area={area}, bbox=({x},{y},{w},{h}) - ACCEPTED")
               motion_regions.append((x, y, w, h))
       
       # Filter thin lines
       filtered_regions = self.filter_thin_line_detections(motion_regions)
       print(f"DEBUG: Regions before filtering: {len(motion_regions)}, after: {len(filtered_regions)}")
       
       # Draw final detections
       final_img = frame.copy()
       for (x, y, w, h) in filtered_regions:
           cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.imwrite("debug_images/08_final_detections.png", final_img)
       
       return filtered_regions

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
        if not self.motion_detection_enabled:
            return []
            
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
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] DEBUG: reset_reference_frame() called!")
    
        if self.current_frame is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            self.reference_frame = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            self.reference_frame_unblurred = gray.copy()
            print(f"DEBUG: Reference frame reset. Unblurred shape: {self.reference_frame_unblurred.shape}")
        
            # Save reference frames for debugging with timestamp
            cv2.imwrite(f"debug_images/00_reference_frame_unblurred_{timestamp.replace(':', '_')}.png", self.reference_frame_unblurred)
        else:
            print("WARNING: current_frame is None, cannot reset reference!")
    
    def get_current_frame(self, include_boxes=True):
        """Get the current frame with optional motion detection boxes."""
        if self.current_frame is None:
            return None
    
        frame = self.current_frame.copy()
    
        if include_boxes and self.motion_regions:
            for (x, y, w, h) in self.motion_regions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        return frame
    
    def set_motion_detection_enabled(self, enabled, reset_ref_frame=True):
        """Enable or disable motion detection."""
        print("Motion Detection before", self.motion_detection_enabled)
        self.motion_detection_enabled = enabled
        print("Motion Detection after", self.motion_detection_enabled)
        if reset_ref_frame:
            # Reset reference frame when enabling motion detection
            self.reset_reference_frame()
    
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
        self.area = area
        self.reference_frame=None #Force Rebuild of reference frame

    def get_area(self):
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
        self.camera = QTCamera(camera_index=0, area=(500, 5, 600, 600))
        self.camera.set_motion_detection_enabled(True)
        self.camera.motion_detected.connect(self.on_motion_detected)
        
        # Set fixed display size for consistent appearance
        self.camera.set_fixed_display_size(600, 600)
        
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