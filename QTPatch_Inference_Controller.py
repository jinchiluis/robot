import sys
import json
import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog,
                               QPushButton, QLabel, QTextEdit, QStatusBar)
import pygame
from QTCamera import QTCamera
from patchcore_exth import SimplePatchCore
from coordinate_transformer import CoordinateTransformer
from QTRobot import QTRobot
from QTPatch_Inference_UI import setup_ui, update_ui_state


class State(Enum):
    """Application states"""
    IDLE = "Watching for motion"
    WAITING = "Motion detected, waiting to capture"
    PROCESSING = "Running PatchCore inference"
    ROBOT_MOVING = "Robot executing command"


class QTPatch_Inference_Controller(QMainWindow):
    """Main controller for PatchCore anomaly detection with live camera feed and robot control."""
    
    # Type hints for UI elements
    start_stop_btn: QPushButton
    load_model_btn: QPushButton
    load_calibration_btn: QPushButton
    state_label: QLabel
    model_status_label: QLabel
    calibration_status_label: QLabel
    anomaly_status_label: QLabel
    anomaly_score_label: QLabel
    threshold_label: QLabel
    heatmap_label: QLabel
    log_display: QTextEdit
    status_bar: QStatusBar
    
    # Signals
    camera_area_changed = Signal(int, int, int, int)  # x, y, width, height
    ready_state_changed = Signal(bool, str, str)  # is_ready, model_path, calibration_path
    
    def __init__(self, x, y, width, height, camera=None, calibration_data=None):
        """
        Initialize PatchCore inference controller.
        
        Args:
            x, y, width, height: Camera area coordinates
            camera: QTCamera instance to use (optional)
            calibration_data: Existing calibration data to load (optional)
        """
        super().__init__()
        self.setWindowTitle("QT PatchCore Inference Controller")
        self.setGeometry(100, 50, 1000, 650)
        
        # State management
        self.state = State.IDLE
        self.model_loaded = False
        self.calibration_loaded = False
        self.model_file_path = None
        self.calibration_file_path = None
        self.last_detection_location = None  # Store motion detection coordinates
        self.last_anomaly_score = None
        
        # PatchCore specific settings
        self.anomaly_threshold = 0.5  # Default threshold, will be loaded from model
        self.normal_action = "pass"  # Robot action for normal items
        self.anomaly_action = "remove"  # Robot action for anomalies
        
        # Components
        self.patchcore_model = None
        self.calibration_data = calibration_data
        self.coordinate_transformer = CoordinateTransformer()
        
        # Load camera index from config
        config_path = Path("config/config.json")
        camera_index = 0
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                camera_index = config.get("camera_index", 0)
            except Exception as e:
                print(f"Warning: Could not read camera_index from config: {e}")
        # Handle camera
        if camera is None:
            self.camera = QTCamera(camera_index=camera_index, area=(x, y, width, height))
            self._owns_camera = True
            self.camera.set_fixed_display_size(600, 600)
            self.store_camera = self.camera
        else:
            self.camera = camera
            self.store_camera = camera
            self._owns_camera = False
            self.camera.set_fixed_display_size(600, 600)
            
        # Initialize robot
        self.robot = QTRobot("192.168.178.98")
        self.robot.robot_complete.connect(self.on_robot_complete)
        self.robot.robot_error.connect(self.on_robot_error)
        
        # Setup UI
        self.setup_ui()

        # Load calibration data if provided
        if calibration_data:
            success, message = self.coordinate_transformer.import_calibration(calibration_data)
            if success:
                self.calibration_loaded = True
                if 'filepath' in calibration_data:
                    self.calibration_file_path = calibration_data['filepath']
                self.log(f"Calibration loaded from main controller")
                self.log(message)
        
        # Auto-load model and calibration if available
        QTimer.singleShot(0, self.auto_load_files)
       
        # Initialize sound using pygame
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("beep.mp3")
            
    def auto_load_files(self):
        """Automatically load model and calibration files from default folders."""
        # Check for PatchCore model file in "patchcore_models" folder
        model_folder = Path("patchcore_models")
        if model_folder.exists():
            # Look for .pth or .pkl files
            model_files = list(model_folder.glob("*.pth")) + list(model_folder.glob("*.pkl"))
            if model_files:
                # Take the most recently modified file
                model_file = max(model_files, key=lambda f: f.stat().st_mtime)
                try:
                    self.patchcore_model = SimplePatchCore(device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
                    self.patchcore_model.load(str(model_file))
                    self.model_loaded = True
                    self.model_status_label.setText(f"Model: {model_file.name}")
                    self.model_file_path = str(model_file)
                    
                    # Update threshold from model
                    if hasattr(self.patchcore_model, 'global_threshold'):
                        self.anomaly_threshold = self.patchcore_model.global_threshold
                        self.threshold_label.setText(f"Threshold: {self.anomaly_threshold:.4f}")
                    
                    self.log(f"Auto-loaded PatchCore model: {model_file}")
                except Exception as e:
                    self.log(f"Failed to auto-load model: {str(e)}")
        
        # Auto-load calibration (same as DINO version)
        if not self.calibration_loaded:
            calibration_folder = Path("calibration")
            if calibration_folder.exists():
                calibration_files = list(calibration_folder.glob("*.json"))
                if calibration_files:
                    calibration_file = max(calibration_files, key=lambda f: f.stat().st_mtime)
                    try:
                        with open(calibration_file, 'r') as f:
                            self.calibration_data = json.load(f)
                        
                        success, message = self.coordinate_transformer.import_calibration(self.calibration_data)
                        if success:
                            self.calibration_loaded = True
                            self.calibration_status_label.setText(f"Calibration: {calibration_file.name}")
                            self.calibration_file_path = str(calibration_file)
                            self.log(f"Auto-loaded calibration: {calibration_file}")
                            self.log(message)

                        if 'camera_area' in self.calibration_data:
                            area = self.calibration_data['camera_area']
                            camera_area = (area['x'], area['y'], area['width'], area['height'])
                            if self.camera is not None:
                                self.camera.set_area(camera_area)
                                self.log(f"Camera area set to: {camera_area}")
                                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])

                    except Exception as e:
                        self.log(f"Failed to auto-load calibration: {str(e)}")
        else:
            if self.calibration_data and 'name' in self.calibration_data:
                self.calibration_status_label.setText(f"Calibration: {self.calibration_data['name']}")
            else:
                self.calibration_status_label.setText("Calibration: Loaded")

        self.check_ready()

    def setup_ui(self):
        """Setup the user interface."""
        setup_ui(self)

    def on_view_switched(self, view_name):
        """Handle view switch from main controller."""
        if view_name == "PatchCore_Inference":
            self.camera = self.store_camera 
            print("PatchCore Inference view activated")
        else:
            print(f"Switched to {view_name} view")
            self.camera = None

    def log(self, message):
        """Add message to log display."""
        self.log_display.append(f"[{self.state.name}] {message}")
        
    def update_state(self, new_state):
        """Update application state."""
        self.state = new_state
        update_ui_state(self)
        self.log(f"State changed to: {new_state.value}")

    def load_model(self):
        """Load PatchCore model from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load PatchCore Model", "", "Model Files (*.pth *.pkl)"
        )
        
        if file_path:
            try:
                self.patchcore_model = SimplePatchCore(device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
                self.patchcore_model.load(file_path)
                self.model_loaded = True
                self.model_status_label.setText(f"Model: {Path(file_path).name}")
                self.model_file_path = file_path
                
                # Update threshold from model
                if hasattr(self.patchcore_model, 'global_threshold'):
                    self.anomaly_threshold = self.patchcore_model.global_threshold
                    self.threshold_label.setText(f"Threshold: {self.anomaly_threshold:.4f}")
                
                self.log(f"PatchCore model loaded: {file_path}")
                self.check_ready()
            except Exception as e:
                self.log(f"Failed to load model: {str(e)}")
                
    def load_calibration(self):
        """Load calibration JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.calibration_data = json.load(f)
                
                success, message = self.coordinate_transformer.import_calibration(self.calibration_data)
                if success:
                    self.calibration_loaded = True
                    self.calibration_status_label.setText(f"Calibration: {Path(file_path).name}")
                    self.log(f"Calibration loaded: {file_path}")
                    self.log(message)
                    self.check_ready()
                    self.calibration_file_path = file_path
                else:
                    self.log(f"Failed to import calibration: {message}")
                    
            except Exception as e:
                self.log(f"Failed to load calibration: {str(e)}")

        if self.calibration_data and 'camera_area' in self.calibration_data:
            area = self.calibration_data['camera_area']
            camera_area = (area['x'], area['y'], area['width'], area['height'])
            self.camera.set_area(camera_area)
            self.log(f"Camera area set to: {camera_area}")
            self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                
    def update_calibration(self, calibration_data):
        """Update calibration data from external source."""
        if calibration_data:
            success, message = self.coordinate_transformer.import_calibration(calibration_data)
            if success:
                self.calibration_data = calibration_data
                self.calibration_loaded = True
                if 'filepath' in calibration_data:
                    self.calibration_file_path = calibration_data['filepath']

                if 'name' in calibration_data:
                    self.calibration_status_label.setText(f"Calibration: {calibration_data['name']}")
                else:
                    self.calibration_status_label.setText("Calibration: Updated")
            
                if 'camera_area' in calibration_data:
                    area = calibration_data['camera_area']
                    camera_area = (area['x'], area['y'], area['width'], area['height'])
                    self.camera.set_area(camera_area)
                    self.log(f"Camera area set to: {camera_area}")
                    self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                    
                self.log("Calibration updated from main controller")
                self.log(message)
                self.check_ready()
            else:
                self.log(f"Failed to update calibration: {message}")
        self.check_ready()
                
    def check_ready(self):
        """Check if both model and calibration are loaded."""
        if self.model_loaded and self.calibration_loaded:
            self.start_stop_btn.setEnabled(True)
            self.log("Ready to start anomaly detection")
        
            model_path = self.model_file_path if hasattr(self, 'model_file_path') else None
            calibration_path = self.calibration_file_path if hasattr(self, 'calibration_file_path') else None
            
            self.ready_state_changed.emit(True, model_path, calibration_path)
        else:
            self.ready_state_changed.emit(False, None, None)
            
    def toggle_detection(self):
        """Start or stop motion detection."""
        if self.start_stop_btn.text() == "Start Detection":
            # Get the current center of detected motion to store
            if self.camera is not None:
                camera_area = self.camera.get_area()
                center_x = camera_area[0] + camera_area[2] // 2
                center_y = camera_area[1] + camera_area[3] // 2
                self.last_detection_location = (center_x, center_y)
                self.camera.set_object_detection_enabled(True)
                self.start_stop_btn.setText("Stop Detection")
                self.log("Motion detection started")
            else:
                self.log("Camera is not available. Cannot start detection.")
        else:
            self.camera.set_object_detection_enabled(False)
            self.update_state(State.IDLE)
            self.start_stop_btn.setText("Start Detection")
            self.log("Motion detection stopped")
            
    def on_object_detected(self):
        """Handle motion detection signal from camera."""
        if self.state == State.IDLE:
            # Store the motion detection location
            camera_area = self.camera.get_area()
            center_x = camera_area[0] + camera_area[2] // 2
            center_y = camera_area[1] + camera_area[3] // 2
            self.last_detection_location = (center_x, center_y)
            self.log(f"Motion detected at pixel: {self.last_detection_location}")
            
            self.update_state(State.WAITING)
            # Wait 3 seconds then capture
            QTimer.singleShot(5000, self.capture_and_process)

    def capture_and_process(self):
        """Capture frame and run PatchCore inference."""
        self.update_state(State.PROCESSING)
    
        # Get current frame WITHOUT detection boxes
        frame = self.camera.get_current_frame(include_boxes=False)
        if frame is None:
            self.log("Failed to capture frame")
            self.reset_to_idle()
            return

        # Disable motion detection
        self.camera.set_object_detection_enabled(False)

        # Check if we have a detected region
        if not self.camera.detected_regions:
            self.log("No object detected in frame")
            self.reset_to_idle()
            return
    
        # Get the first detected region
        x, y, width, height = self.camera.detected_regions[0]
    
        # Store the center for robot coordinates
        center_x = x + width // 2
        center_y = y + height // 2
        self.last_detection_location = (center_x, center_y)
    
        # Crop to detected region only
        cropped_frame = frame[y:y+height, x:x+width]
    
        # Save cropped frame for PatchCore processing
        temp_path = "temp_patchcore_capture.png"
        cv2.imwrite(temp_path, cropped_frame)

        try:
            # Run PatchCore inference on cropped object
            self.log(f"Running PatchCore on cropped object ({width}x{height} pixels)...")
            if self.patchcore_model is not None:                     
                #-> in sterile settings -> Set min_region_size between 25 and 50
                #-> In more chaotic settings, don't need to set this, because tiny anomalies are not detected anyway
                import time
                start_time = time.perf_counter()
                result_filtered = self.patchcore_model.predict(temp_path, return_heatmap=True, min_region_size=50)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"Prediction overall took {execution_time:.4f} seconds")
        
                # Extract results from both predictions
                anomaly_score = result_filtered['anomaly_score']
                is_anomaly = result_filtered['is_anomaly']
                threshold = result_filtered['threshold'] #normal and filtered have same threshold, filtered can return NORMAL, if there were no big anomalies
        
                # Store both scores
                self.last_anomaly_score = anomaly_score
        
                # Update UI with both scores
                self.anomaly_score_label.setText(f"Score normal: {anomaly_score:.4f}")
                status_text = "ANOMALY" if is_anomaly else "NORMAL"
                self.anomaly_status_label.setText(f"Status: {status_text}")
        
                #Display heatmap if available
                if 'heatmap' in result_filtered and result_filtered['heatmap'] is not None:
                    heatmap = result_filtered['heatmap']
                    # Convert numpy array to QPixmap
                    height, width, channel = heatmap.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(heatmap.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaledToWidth(250, Qt.SmoothTransformation)
                    self.heatmap_label.setPixmap(scaled_pixmap)
                else:
                    self.heatmap_label.setText("No heatmap")

                # Color code the status
                if is_anomaly:
                    self.anomaly_status_label.setStyleSheet("color: red; font-weight: bold;")
                    # Play beep sound on anomaly if available
                    try:
                        self.sound.play()
                    except Exception as e:
                        self.log(f"Sound playback failed: {e}")
                else:
                    self.anomaly_status_label.setStyleSheet("color: green; font-weight: bold;")
        
                # Log both results
                self.log(f"Normal score: {anomaly_score:.4f}, (threshold: {threshold:.4f}) - {status_text}")
                              
                # Process based on result
                if is_anomaly and self.last_detection_location:
                    # Convert pixel to robot coordinates
                    pixel_x, pixel_y = self.last_detection_location
                    robot_x, robot_y = self.pixel_to_robot_coords(pixel_x, pixel_y)
                    
                    robot_command = {
                        'status': 'anomaly',
                        'pixel': (pixel_x, pixel_y),
                        'robot': (robot_x, robot_y),
                        'action': self.anomaly_action
                    }
                    
                    self.log(f"Anomaly detected at pixel ({pixel_x},{pixel_y}) -> robot ({robot_x:.2f},{robot_y:.2f})")
                    self.send_to_robot([robot_command])
                elif not is_anomaly and self.last_detection_location:
                    # Optionally handle normal items
                    if self.normal_action != "pass":
                        pixel_x, pixel_y = self.last_detection_location
                        robot_x, robot_y = self.pixel_to_robot_coords(pixel_x, pixel_y)
                        
                        robot_command = {
                            'status': 'normal',
                            'pixel': (pixel_x, pixel_y),
                            'robot': (robot_x, robot_y),
                            'action': self.normal_action
                        }
                        
                        self.log(f"Normal item at pixel ({pixel_x},{pixel_y}) -> robot ({robot_x:.2f},{robot_y:.2f})")
                        self.send_to_robot([robot_command])
                    else:
                        self.log("Normal item detected - Start detection in 2 seconds.")
                        QTimer.singleShot(2000, self.reset_to_idle)
                else:
                    self.log("No valid detection location available. Start detection in 2 seconds.")
                    QTimer.singleShot(2000, self.reset_to_idle)
                    
        except Exception as e:
            self.log(f"Error during PatchCore inference: {str(e)}")
            QTimer.singleShot(2000, self.reset_to_idle)
        finally:
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def pixel_to_robot_coords(self, pixel_x, pixel_y):
        """Convert pixel coordinates to robot coordinates using calibration."""
        result = self.coordinate_transformer.pixel_to_robot(pixel_x, pixel_y)
        if result is not None:
            robot_x, robot_y = result
            return robot_x, robot_y
        else:
            self.log(f"Warning: Coordinate transformation failed, using pixel coordinates")
            return pixel_x, pixel_y
            
    def send_to_robot(self, commands):
        """Send commands to robot."""
        self.update_state(State.ROBOT_MOVING)
        
        # Create list of steps for robot
        steps = []
        for cmd in commands:
            action = cmd['action']
            robot_x, robot_y = cmd['robot']
            if action == self.anomaly_action:
                # For anomalies, process red plane
                steps.append(('process_blue_plane', robot_x, robot_y))
            elif action == self.normal_action:
                # For normal items, process blue plane
                steps.append(('process_nothing_plane', robot_x, robot_y))  
            steps.append((action, robot_x, robot_y))
        
        if steps:
            self.log(f"Sending {len(steps)} commands to robot...")
            self.robot.process_steps_qt(steps)
        else:
            self.log("No valid robot commands to execute")
            self.reset_to_idle()
        
    def on_robot_complete(self):
        """Handle robot completion."""
        self.log("Robot finished execution. Retarting detection in 2 seconds...")
        QTimer.singleShot(2000, self.reset_to_idle)
        
    def on_robot_error(self, error_msg):
        """Handle robot error."""
        self.log(f"Robot error: {error_msg}")
        self.update_state(State.IDLE)
        self.camera.set_object_detection_enabled(False)
        self.start_stop_btn.setText("Start Detection")
        
    def reset_to_idle(self, reset_ref_frame=False):
        """Reset to idle state and re-enable motion detection."""
        self.update_state(State.IDLE)
        self.last_detection_location = None  # Clear stored location
        
        if self.start_stop_btn.text() == "Stop Detection":
            self.camera.set_object_detection_enabled(True, reset_ref_frame)
            self.log("Motion detection re-enabled")
            
    def closeEvent(self, event):
        """Clean up on close."""
        if self._owns_camera and self.camera:
            self.camera.close()
        event.accept()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = QTPatch_Inference_Controller(500,0, 600, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()