import sys
import json
import cv2
import numpy as np
from pathlib import Path
from enum import Enum
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QTextEdit, QGroupBox)

from QTCamera import QTCamera
from dino_object_detector import ObjectDetector
from coordinate_transformer import CoordinateTransformer
from QTRobot import QTRobot


class State(Enum):
    """Application states"""
    IDLE = "Watching for motion"
    WAITING = "Motion detected, waiting to capture"
    PROCESSING = "Running DINO inference"
    ROBOT_MOVING = "Robot executing command"


class QTInference_Controller(QMainWindow):
    """Main controller for DINO inference with live camera feed and robot control."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT Inference Controller")
        
        # State management
        self.state = State.IDLE
        self.model_loaded = False
        self.calibration_loaded = False
        
        # Components
        self.detector = None
        self.calibration_data = None
        self.camera = None
        self.coordinate_transformer = CoordinateTransformer()
        
        # Initialize robot
        self.robot = QTRobot("192.168.178.98")
        self.robot.robot_complete.connect(self.on_robot_complete)
        self.robot.robot_error.connect(self.on_robot_error)
        
        # Setup UI
        self.setup_ui()
        
        # Auto-load model and calibration if available
        self.auto_load_files()
        
    def auto_load_files(self):
        """Automatically load model and calibration files from default folders."""
        # Check for model file in "model" folder
        model_folder = Path("model")
        if model_folder.exists():
            # Look for .pkl files in the model folder
            model_files = list(model_folder.glob("*.pkl"))
            if model_files:
                # Take the most recently modified .pkl file
                model_file = max(model_files, key=lambda f: f.stat().st_mtime)
                try:
                    self.detector = ObjectDetector(str(model_file))
                    self.model_loaded = True
                    self.model_status_label.setText(f"Model: {model_file.name}")
                    self.log(f"Auto-loaded model: {model_file}")
                except Exception as e:
                    self.log(f"Failed to auto-load model: {str(e)}")
        
        # Check for calibration file in "calibration" folder
        calibration_folder = Path("calibration")
        if calibration_folder.exists():
            # Look for .json files in the calibration folder
            calibration_files = list(calibration_folder.glob("*.json"))
            if calibration_files:
                # Take the most recently modified .json file
                calibration_file = max(calibration_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(calibration_file, 'r') as f:
                        self.calibration_data = json.load(f)
                    
                    # Import calibration data into the coordinate transformer
                    success, message = self.coordinate_transformer.import_calibration(self.calibration_data)
                    if success:
                        self.calibration_loaded = True
                        self.calibration_status_label.setText(f"Calibration: {calibration_file.name}")
                        self.log(f"Auto-loaded calibration: {calibration_file}")
                        self.log(message)
                    else:
                        self.log(f"Failed to import auto-loaded calibration: {message}")
                        
                except Exception as e:
                    self.log(f"Failed to auto-load calibration: {str(e)}")
        
        # Check if both are loaded and enable start button
        self.check_ready()

    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("Load DINO Model")
        self.load_model_btn.clicked.connect(self.load_model)
        control_layout.addWidget(self.load_model_btn)
        
        self.load_calibration_btn = QPushButton("Load Calibration")
        self.load_calibration_btn.clicked.connect(self.load_calibration)
        control_layout.addWidget(self.load_calibration_btn)
        
        self.start_stop_btn = QPushButton("Start Detection")
        self.start_stop_btn.clicked.connect(self.toggle_detection)
        self.start_stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_stop_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Camera widget
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout()
        
        self.camera = QTCamera(camera_index=0, area=(500, 0, 600, 600))
        self.camera.motion_detected.connect(self.on_motion_detected)
        camera_layout.addWidget(self.camera)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.state_label = QLabel(f"State: {self.state.value}")
        status_layout.addWidget(self.state_label)
        
        self.model_status_label = QLabel("Model: Not loaded")
        status_layout.addWidget(self.model_status_label)
        
        self.calibration_status_label = QLabel("Calibration: Not loaded")
        status_layout.addWidget(self.calibration_status_label)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Log display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        central_widget.setLayout(main_layout)
        
        # Initialize with motion detection disabled
        self.camera.set_motion_detection_enabled(False)
        
    def log(self, message):
        """Add message to log display."""
        self.log_display.append(f"[{self.state.name}] {message}")
        
    def update_state(self, new_state):
        """Update application state."""
        self.state = new_state
        self.state_label.setText(f"State: {self.state.value}")
        self.log(f"State changed to: {new_state.value}")
        
    def load_model(self):
        """Load DINO model from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load DINO Model", "", "Model Files (*.pkl)"
        )
        
        if file_path:
            try:
                self.detector = ObjectDetector(file_path)
                self.model_loaded = True
                self.model_status_label.setText(f"Model: {Path(file_path).name}")
                self.log(f"Model loaded: {file_path}")
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
                
                # Import calibration data into the coordinate transformer
                success, message = self.coordinate_transformer.import_calibration(self.calibration_data)
                if success:
                    self.calibration_loaded = True
                    self.calibration_status_label.setText(f"Calibration: {Path(file_path).name}")
                    self.log(f"Calibration loaded: {file_path}")
                    self.log(message)
                    self.check_ready()
                else:
                    self.log(f"Failed to import calibration: {message}")
                    
            except Exception as e:
                self.log(f"Failed to load calibration: {str(e)}")
                
    def check_ready(self):
        """Check if both model and calibration are loaded."""
        if self.model_loaded and self.calibration_loaded:
            self.start_stop_btn.setEnabled(True)
            self.log("Ready to start detection")
            
    def toggle_detection(self):
        """Start or stop motion detection."""
        if self.state == State.IDLE:
            # Start detection
            self.camera.set_motion_detection_enabled(True)
            self.start_stop_btn.setText("Stop Detection")
            self.log("Motion detection started")
        else:
            # Stop detection
            self.camera.set_motion_detection_enabled(False)
            self.update_state(State.IDLE)
            self.start_stop_btn.setText("Start Detection")
            self.log("Motion detection stopped")
            
    def on_motion_detected(self, regions):
        """Handle motion detection signal from camera."""
        if self.state == State.IDLE:
            self.log(f"Motion detected: {len(regions)} regions")
            self.update_state(State.WAITING)
            
            # Wait 3.5 second then capture
            QTimer.singleShot(3500, self.capture_and_process)
            
    def capture_and_process(self):
        """Capture frame and run DINO inference."""
        self.update_state(State.PROCESSING)
        
        # Get current frame
        frame = self.camera.get_current_frame()
        if frame is None:
            self.log("Failed to capture frame")
            self.reset_to_idle()
            return

        # Disable motion detection
        self.camera.set_motion_detection_enabled(False) #Disable here, because we need the green line in the image

        # Save frame temporarily for DINO processing
        temp_path = "temp_capture.png"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Run DINO inference
            self.log("Running DINO inference...")
            detections = self.detector.detect(temp_path, visualize=False, debug=False)
            
            # Process detections
            robot_commands = []
            for obj_type, centers in detections.items():
                for i, (x, y) in enumerate(centers):
                    # Convert pixel to robot coordinates using calibration
                    robot_x, robot_y = self.pixel_to_robot_coords(x, y)
                    robot_commands.append({
                        'object': obj_type,
                        'pixel': (x, y),
                        'robot': (robot_x, robot_y)
                    })
                    self.log(f"Detected {obj_type} at pixel ({x},{y}) -> robot ({robot_x:.2f},{robot_y:.2f})")
            
            if robot_commands:
                self.send_to_robot(robot_commands)
            else:
                self.log("No objects detected")
                self.reset_to_idle()
                
        except Exception as e:
            self.log(f"Error during inference: {str(e)}")
            self.reset_to_idle()
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
            # Return pixel coordinates as fallback if transformation fails
            self.log(f"Warning: Coordinate transformation failed, using pixel coordinates")
            return pixel_x, pixel_y
            
    def send_to_robot(self, commands):
        """Send commands to robot."""
        self.update_state(State.ROBOT_MOVING)
        
        # Create list of steps based on detections
        steps = []
        for cmd in commands:
            if 'red' in cmd['object'].lower():
                steps.append(('process_red_plane', cmd['robot'][0], cmd['robot'][1]))
            elif 'blue' in cmd['object'].lower():
                steps.append(('process_blue_plane', cmd['robot'][0], cmd['robot'][1]))
        
        if steps:
            self.log(f"Sending {len(steps)} commands to robot...")
            self.robot.process_steps_qt(steps)
        else:
            self.log("No valid robot commands to execute")
            self.reset_to_idle()
        
    def on_robot_complete(self):
        """Handle robot completion."""
        self.log("Robot finished execution")
        self.reset_to_idle()
        
    def on_robot_error(self, error_msg):
        """Handle robot error."""
        self.log(f"Robot error: {error_msg}")
        self.update_state(State.IDLE)
        # Disable detection on error
        self.camera.set_motion_detection_enabled(False)
        self.start_stop_btn.setText("Start Detection")
        
    def reset_to_idle(self):
        """Reset to idle state and re-enable motion detection."""
        self.update_state(State.IDLE)
        if self.start_stop_btn.text() == "Stop Detection":
            # Only re-enable if we're in "running" mode
            self.camera.set_motion_detection_enabled(True)
            self.log("Motion detection re-enabled")
            
    def closeEvent(self, event):
        """Clean up on close."""
        if self.camera:
            self.camera.close()
        event.accept()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = QTInference_Controller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()