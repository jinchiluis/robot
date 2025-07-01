import os
import json
from datetime import datetime
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QMainWindow, QPushButton, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QInputDialog, QStatusBar)
from coordinate_transformer import CoordinateTransformer
from QTRobot import QTRobot
from QTCalibration_UI import setup_ui, display_frame, update_ui_state
from QTCalibration_UI import ClickableLabel


class QTCalibration_Controller(QMainWindow):
    """Calibration widget for pixel to robot coordinate transformation."""
    
    # Type hints for UI elements
    image_label: 'ClickableLabel'
    freeze_btn: QPushButton
    reset_btn: QPushButton
    robot_x_input: QLineEdit
    robot_y_input: QLineEdit
    add_point_btn: QPushButton
    current_pixel_label: QLabel
    points_list: QListWidget
    delete_point_btn: QPushButton
    calculate_btn: QPushButton
    points_count_label: QLabel
    test_info_label: QLabel
    test_result_label: QLabel
    save_btn: QPushButton
    status_bar: QStatusBar

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
            width, height = self.camera.get_effective_resolution()
            self.camera.set_fixed_display_size(int(0.6 * width), int(0.6 * height))
            
        self.transformer = CoordinateTransformer()
        self.state = self.STATE_LIVE
        self.frozen_frame = None
        self.current_point = None
        
        # Robot control attributes
        self.robot = None
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self._continuous_move)
        self.current_move_direction = None
        
        # Load calibration data if provided
        if calibration_data:
            success, message = self.transformer.import_calibration(calibration_data)
            if success:
                self.state = self.STATE_COMPLETE
            # Initialize robot

        try:
            self.robot = QTRobot("192.168.178.98")  # Update with your robot IP
            self.robot.robot_complete.connect(lambda: print("Robot movement complete"))
            self.robot.robot_error.connect(lambda err: print(f"Robot error: {err}"))
            # Enable keyboard focus for robot control
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception as e:
            print(f"Failed to initialize robot: {e}")
            self.robot = None        
        
        self.setup_ui()
        self.update_ui_state()
        
    def setup_ui(self):
        """Setup the user interface."""
        setup_ui(self)
   
    def on_direction_pressed(self, direction):
        """Handle directional button press."""
        if self.robot and self.state in [self.STATE_FROZEN, self.STATE_CALIBRATING]:
            self.current_move_direction = direction
            self._move_robot(direction)
            self.move_timer.start(200)

    def on_direction_released(self):
        """Handle directional button release."""
        self.move_timer.stop()
        self.current_move_direction = None

    def keyPressEvent(self, event):
        """Handle key press for robot control."""
        if not self.robot or self.state not in [self.STATE_FROZEN, self.STATE_CALIBRATING]:
            return
        
        key = event.key()
        direction_map = {
            Qt.Key_A: "L",  # A for Left
            Qt.Key_D: "R",  # D for Right
            Qt.Key_W: "U",  # W for Up
            Qt.Key_S: "D"   # S for Down
        }
    
        # Check for diagonal movements (modifier + key)
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            if key == Qt.Key_W and event.modifiers() == Qt.ShiftModifier:
                if self.current_move_direction == "L":
                    direction = "UL"
                elif self.current_move_direction == "R":
                    direction = "UR"
                else:
                    return
            elif key == Qt.Key_S and event.modifiers() == Qt.ShiftModifier:
                if self.current_move_direction == "L":
                    direction = "DL"
                elif self.current_move_direction == "R":
                    direction = "DR"
                else:
                    return
            else:
                return
        else:
            # Simple direction mapping
            direction = direction_map.get(key)
            if not direction:
                # Check for diagonal keys without shift
                if key == Qt.Key_Q:  # Q for UL
                    direction = "UL"
                elif key == Qt.Key_E:  # E for UR
                    direction = "UR"
                elif key == Qt.Key_Z:  # Z for DL
                    direction = "DL"
                elif key == Qt.Key_C:  # C for DR
                    direction = "DR"
                else:
                    return
            
        # Start continuous movement
        if not event.isAutoRepeat():  # Only on initial press
            self.current_move_direction = direction
            self._move_robot(direction)
            self.move_timer.start(200)  # Move every 200ms while held
                
          
    def keyReleaseEvent(self, event):
        """Handle key release to stop continuous movement."""
        if not self.robot or event.isAutoRepeat():
            return
            
        # Stop continuous movement
        self.move_timer.stop()
        self.current_move_direction = None
        
    def _continuous_move(self):
        """Called by timer for continuous movement."""
        if self.current_move_direction:
            self._move_robot(self.current_move_direction)
            
    def _move_robot(self, direction):
        """Move robot and update coordinates."""
        try:
            position = self.robot.controller_move(direction)
            if position:
                print("position:", position)
                # Update coordinate fields with current robot position
                self.robot_x_input.setText(str(int(position["x"])))
                self.robot_y_input.setText(str(int(position["y"])))
        except Exception as e:
            print(f"Robot move error: {e}")
            
    def timerEvent(self, event):
        """Update display based on current state."""
        if self.state == self.STATE_LIVE:
            frame = self.camera.get_current_frame(include_boxes=False)
            if frame is not None:
                self.display_frame(frame)
                
    def display_frame(self, frame):
        """Display a frame in the image label with aspect ratio preservation."""
        display_frame(self, frame)
        
    def toggle_freeze(self):
        """Toggle between live and frozen frame."""
        if self.state == self.STATE_LIVE:
            # Freeze current frame
            self.frozen_frame = self.camera.get_current_frame(include_boxes=False)
            if self.frozen_frame is not None:
                self.state = self.STATE_FROZEN
                self.display_frame(self.frozen_frame)
                self.freeze_btn.setText("Back to Live")
                
                # Turn on robot light if available
                if self.robot:
                    try:
                        self.robot.light_on(True)
                    except Exception as e:
                        print(f"Light control error: {e}")
        else:
            # Back to live
            self.state = self.STATE_LIVE
            self.freeze_btn.setText("Start Calibration")
            self.image_label.clear_points()
            
            # Turn off robot light if available
            if self.robot:
                try:
                    self.robot.light_on(False)
                except Exception as e:
                    print(f"Light control error: {e}")
            
        self.update_ui_state()
        
    def on_image_clicked(self, x, y):
        """Handle click on image (coordinates are already in image space)."""
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
        #move robot to start
        self.robot.initialize()
        
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
            
            # Add camera area information
            camera_area = self.camera.get_area()
            if camera_area is None:
            # No crop area set - use full camera resolution
                width, height = self.camera.get_effective_resolution()
                camera_area = (0, 0, width, height)

            calibration_data['camera_area'] = {
                'x': camera_area[0],
                'y': camera_area[1],
                'width': camera_area[2],
                'height': camera_area[3]
            }

            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            calibration_data['filepath'] = str(filepath)  
            # Emit signal with calibration data
            self.calibration_saved.emit(calibration_data)
                
            QMessageBox.information(self, "Saved", f"Calibration saved as '{name}'")
            self.state = self.STATE_COMPLETE
            self.update_ui_state()
            
            # Turn off robot light when complete
            if self.robot:
                try:
                    self.robot.initialize()
                except Exception as e:
                    print(f"Light control error: {e}")
            
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
        
        # Turn off robot light
        if self.robot:
            try:
                #self.robot.light_on(False)
                self.robot.initialize()
            except Exception as e:
                print(f"Light control error: {e}")
                
        self.update_ui_state()
        
    def update_ui_state(self):
        """Update UI elements based on current state."""
        update_ui_state(self)
        
    def get_current_calibration(self):
        """Get current calibration data if available."""
        if self.transformer.is_calibrated():
            return self.transformer.export_calibration()
        return None
        
    def closeEvent(self, event):
        """Clean up on close."""
        # Turn off robot light if on
        if self.robot:
            try:
                self.robot.light_on(False)
            except Exception as e:
                print(f"Light control error: {e}")
                
        # Stop move timer
        self.move_timer.stop()
        
        # Only close camera if we created it
        if self._owns_camera and self.camera:
            self.camera.close()
        event.accept()