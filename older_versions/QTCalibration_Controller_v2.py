import os
import json
from datetime import datetime
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QInputDialog, QGroupBox, QFrame)
from coordinate_transformer import CoordinateTransformer


class ClickableLabel(QLabel):
    """Label that can detect mouse clicks and hover with coordinates."""
    
    clicked = Signal(int, int)
    mouse_moved = Signal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.points = []  # List of (x, y, color) tuples
        self.hover_pos = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())
            
    def mouseMoveEvent(self, event):
        self.hover_pos = (event.x(), event.y())
        self.mouse_moved.emit(event.x(), event.y())
        self.update()
        
    def leaveEvent(self, event):
        self.hover_pos = None
        self.update()
        
    def add_point(self, x, y, color=QColor(255, 0, 0)):
        """Add a point to be drawn."""
        self.points.append((x, y, color))
        self.update()
        
    def clear_points(self):
        """Clear all points."""
        self.points = []
        self.update()
        
    def update_point_color(self, index, color):
        """Update color of a specific point."""
        if 0 <= index < len(self.points):
            x, y, _ = self.points[index]
            self.points[index] = (x, y, color)
            self.update()
            
    def remove_point(self, index):
        """Remove a point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.update()
            
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw points
        for x, y, color in self.points:
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
            
        # Draw hover coordinates
        if self.hover_pos:
            x, y = self.hover_pos
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            text = f"({x}, {y})"
            rect = painter.fontMetrics().boundingRect(text)
            rect.adjust(-2, -2, 2, 2)
            rect.moveTopLeft(QPoint(x, y))  # Changed this line
            painter.drawRect(rect)
            painter.drawText(x, y, text)  # Simplified this line

        painter.end()  # Add this line at the very end


class QTCalibration_Controller(QWidget):
    """Calibration widget for pixel to robot coordinate transformation."""
    
    # Signal for when calibration is saved
    calibration_saved = Signal(dict)
    
    # States
    STATE_LIVE = "LIVE"
    STATE_FROZEN = "FROZEN"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_TESTING = "TESTING"
    STATE_COMPLETE = "COMPLETE"
    
    def __init__(self, camera_widget=None, calibration_data=None, parent=None):
        """
        Initialize calibration widget.
        
        Args:
            camera_widget: QTCamera instance to get frames from (optional)
            calibration_data: Existing calibration data to load (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Handle camera - create if not provided (for standalone use)
        if camera_widget is None:
            from QTCamera import QTCamera
            self.camera = QTCamera(camera_index=0)
            self._owns_camera = True
        else:
            self.camera = camera_widget
            self._owns_camera = False
            
        self.transformer = CoordinateTransformer()
        self.state = self.STATE_LIVE
        self.frozen_frame = None
        self.current_point = None
        
        # Load calibration data if provided
        if calibration_data:
            success, message = self.transformer.import_calibration(calibration_data)
            if success:
                self.state = self.STATE_COMPLETE
                
        self.setup_ui()
        self.update_ui_state()
        
    def setup_ui(self):
        """Setup the user interface."""
        main_layout = QHBoxLayout()
        
        # Left side - Image display
        left_layout = QVBoxLayout()
        
        # Image label
        self.image_label = ClickableLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid gray")
        self.image_label.clicked.connect(self.on_image_clicked)
        self.image_label.mouse_moved.connect(self.on_mouse_moved)
        left_layout.addWidget(self.image_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.freeze_btn = QPushButton("Start Calibration")
        self.freeze_btn.clicked.connect(self.toggle_freeze)
        button_layout.addWidget(self.freeze_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_calibration)
        button_layout.addWidget(self.reset_btn)
        
        left_layout.addLayout(button_layout)
        
        # Right side - Controls
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_widget.setLayout(right_layout)
        
        # Current point input
        current_group = QGroupBox("Current Point")
        current_layout = QVBoxLayout()
        
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Robot X:"))
        self.robot_x_input = QLineEdit("0")
        coord_layout.addWidget(self.robot_x_input)
        coord_layout.addWidget(QLabel("Y:"))
        self.robot_y_input = QLineEdit("0")
        coord_layout.addWidget(self.robot_y_input)
        current_layout.addLayout(coord_layout)
        
        self.add_point_btn = QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.add_calibration_point)
        current_layout.addWidget(self.add_point_btn)
        
        self.current_pixel_label = QLabel("Click on image...")
        current_layout.addWidget(self.current_pixel_label)
        
        current_group.setLayout(current_layout)
        right_layout.addWidget(current_group)
        
        # Points list
        points_group = QGroupBox("Calibration Points")
        points_layout = QVBoxLayout()
        
        self.points_list = QListWidget()
        points_layout.addWidget(self.points_list)
        
        self.delete_point_btn = QPushButton("Delete Selected")
        self.delete_point_btn.clicked.connect(self.delete_selected_point)
        points_layout.addWidget(self.delete_point_btn)
        
        self.calculate_btn = QPushButton("Calculate Transformation")
        self.calculate_btn.clicked.connect(self.calculate_transformation)
        points_layout.addWidget(self.calculate_btn)
        
        self.points_count_label = QLabel("Points: 0/4")
        points_layout.addWidget(self.points_count_label)
        
        points_group.setLayout(points_layout)
        right_layout.addWidget(points_group)
        
        # Test mode
        test_group = QGroupBox("Test Mode")
        test_layout = QVBoxLayout()
        
        self.test_info_label = QLabel("Calculate transformation first")
        test_layout.addWidget(self.test_info_label)
        
        self.test_result_label = QLabel("Result: -")
        test_layout.addWidget(self.test_result_label)
        
        test_group.setLayout(test_layout)
        right_layout.addWidget(test_group)
        
        # Save button
        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.clicked.connect(self.save_calibration)
        right_layout.addWidget(self.save_btn)
        
        right_layout.addStretch()
        
        # Combine layouts
        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(right_widget, 1)
        
        self.setLayout(main_layout)
        
        # Start live view update
        self.startTimer(30)  # 30ms timer for updating display
        
    def timerEvent(self, event):
        """Update display based on current state."""
        if self.state == self.STATE_LIVE:
            frame = self.camera.get_current_frame(include_boxes=False)
            if frame is not None:
                self.display_frame(frame)
                
    def display_frame(self, frame):
        """Display a frame in the image label."""
        import cv2
        from PySide6.QtGui import QImage
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
    def toggle_freeze(self):
        """Toggle between live and frozen frame."""
        if self.state == self.STATE_LIVE:
            # Freeze current frame
            self.frozen_frame = self.camera.get_current_frame(include_boxes=False)
            if self.frozen_frame is not None:
                self.state = self.STATE_FROZEN
                self.display_frame(self.frozen_frame)
                self.freeze_btn.setText("Back to Live")
        else:
            # Back to live
            self.state = self.STATE_LIVE
            self.freeze_btn.setText("Start Calibration")
            self.image_label.clear_points()
            
        self.update_ui_state()
        
    def on_image_clicked(self, x, y):
        """Handle click on image."""
        if self.state in [self.STATE_FROZEN, self.STATE_CALIBRATING]:
            self.current_point = (x, y)
            self.current_pixel_label.setText(f"Pixel: ({x}, {y})")
            
            # Add temporary red point
            if hasattr(self, '_temp_point_index'):
                self.image_label.remove_point(self._temp_point_index)
            self.image_label.add_point(x, y, QColor(255, 0, 0))
            self._temp_point_index = len(self.image_label.points) - 1

            self.update_ui_state()
            
        elif self.state == self.STATE_TESTING:
            # Test mode - show predicted coordinates
            robot_coords = self.transformer.pixel_to_robot(x, y)
            if robot_coords:
                self.test_result_label.setText(f"Result: X:{int(robot_coords[0])} Y:{int(robot_coords[1])}")
                # Add blue test point
                self.image_label.add_point(x, y, QColor(0, 0, 255))
                
    def on_mouse_moved(self, x, y):
        """Handle mouse movement for coordinate display."""
        # Coordinates are displayed by ClickableLabel
        pass
        
    def add_calibration_point(self):
        """Add current point to calibration."""
        if self.current_point is None:
            QMessageBox.warning(self, "No Point", "Click on the image first!")
            return
            
        try:
            robot_x = int(self.robot_x_input.text())
            robot_y = int(self.robot_y_input.text())
        except ValueError:
            robot_x = 0
            robot_y = 0
            
        # Add to transformer
        pixel_x, pixel_y = self.current_point
        self.transformer.add_calibration_point(pixel_x, pixel_y, robot_x, robot_y)
        
        # Update point color to green
        if hasattr(self, '_temp_point_index'):
            self.image_label.update_point_color(self._temp_point_index, QColor(0, 255, 0))
            delattr(self, '_temp_point_index')
        
        # Add to list
        item_text = f"P({pixel_x},{pixel_y}) → R({robot_x},{robot_y})"
        self.points_list.addItem(item_text)
        
        # Update state
        point_count = len(self.transformer.pixel_points)
        self.points_count_label.setText(f"Points: {point_count}/4")
        
        if point_count >= 4:
            self.state = self.STATE_CALIBRATING
        
        # Clear current point
        self.current_point = None
        self.current_pixel_label.setText("Click on image...")
        self.robot_x_input.setText("0")
        self.robot_y_input.setText("0")
        
        self.update_ui_state()
        
    def delete_selected_point(self):
        """Delete selected point from list."""
        current_row = self.points_list.currentRow()
        if current_row >= 0:
            # Remove from transformer
            self.points_list.takeItem(current_row)
            
            # Rebuild transformer
            old_points = list(zip(self.transformer.pixel_points, self.transformer.robot_points))
            old_points.pop(current_row)
            
            self.transformer.clear_points()
            self.image_label.clear_points()
            
            for (px, py), (rx, ry) in old_points:
                self.transformer.add_calibration_point(px, py, rx, ry)
                self.image_label.add_point(int(px), int(py), QColor(0, 255, 0))
                
            # Update count
            point_count = len(self.transformer.pixel_points)
            self.points_count_label.setText(f"Points: {point_count}/4")
            
            # Update state
            if point_count < 4:
                self.state = self.STATE_FROZEN
                
            self.update_ui_state()
            
    def calculate_transformation(self):
        """Calculate the transformation matrix."""
        success, message = self.transformer.calculate_transformation()
        
        if success:
            self.state = self.STATE_TESTING
            self.test_info_label.setText("Click to test calibration")
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Failed", message)
            
        self.update_ui_state()
        
    def save_calibration(self):
        """Save calibration to file."""
        if not self.transformer.is_calibrated():
            QMessageBox.warning(self, "Not Calibrated", "Calculate transformation first!")
            return
            
        # Get name from user
        name, ok = QInputDialog.getText(self, "Save Calibration", "Enter calibration name:")
        if not ok or not name:
            return
            
        # Ensure .json extension
        if not name.endswith('.json'):
            name += '.json'
            
        # Create calibration directory if needed
        os.makedirs('calibration', exist_ok=True)
        filepath = os.path.join('calibration', name)
        
        # Check if file exists
        if os.path.exists(filepath):
            reply = QMessageBox.question(self, "File Exists", 
                                       f"'{name}' already exists. Choose another name?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.save_calibration()  # Retry
            return
            
        # Save calibration
        try:
            calibration_data = self.transformer.export_calibration()
            calibration_data['name'] = name
            calibration_data['created'] = datetime.now().isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            # Emit signal with calibration data
            self.calibration_saved.emit(calibration_data)
                
            QMessageBox.information(self, "Saved", f"Calibration saved as '{name}'")
            self.state = self.STATE_COMPLETE
            self.update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
            
    def reset_calibration(self):
        """Reset all calibration data."""
        self.transformer.clear_points()
        self.image_label.clear_points()
        self.points_list.clear()
        self.current_point = None
        self.current_pixel_label.setText("Click on image...")
        self.robot_x_input.setText("0")
        self.robot_y_input.setText("0")
        self.points_count_label.setText("Points: 0/4")
        self.test_info_label.setText("Calculate transformation first")
        self.test_result_label.setText("Result: -")
        
        if hasattr(self, '_temp_point_index'):
            delattr(self, '_temp_point_index')
            
        self.state = self.STATE_LIVE
        self.freeze_btn.setText("Start Calibration")
        self.update_ui_state()
        
    def update_ui_state(self):
        """Update UI elements based on current state."""
        # Enable/disable based on state
        is_frozen = self.state in [self.STATE_FROZEN, self.STATE_CALIBRATING]
        is_calibrating = self.state == self.STATE_CALIBRATING
        is_testing = self.state in [self.STATE_TESTING, self.STATE_COMPLETE]
        
        self.add_point_btn.setEnabled(is_frozen and self.current_point is not None)
        self.robot_x_input.setEnabled(is_frozen)
        self.robot_y_input.setEnabled(is_frozen)
        self.delete_point_btn.setEnabled(is_frozen and self.points_list.count() > 0)
        self.calculate_btn.setEnabled(is_calibrating)
        self.save_btn.setEnabled(is_testing)
        
        # Update status display
        status_text = f"State: {self.state}"
        self.setWindowTitle(f"Robot Calibration - {status_text}")
        
    def get_current_calibration(self):
        """Get current calibration data if available."""
        if self.transformer.is_calibrated():
            return self.transformer.export_calibration()
        return None
        
    def closeEvent(self, event):
        """Clean up on close."""
        # Only close camera if we created it
        if self._owns_camera and self.camera:
            self.camera.close()
        event.accept()