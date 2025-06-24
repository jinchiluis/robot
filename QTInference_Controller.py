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
    
    # Add this signal
    camera_area_changed = Signal(int, int, int, int)  # x, y, width, height
    ready_state_changed = Signal(bool, str, str)  # is_ready, model_path, calibration_path
    
    def __init__(self, camera=None, calibration_data=None):
        """
        Initialize inference controller.
        
        Args:
            camera: QTCamera instance to use (optional)
            calibration_data: Existing calibration data to load (optional)
        """
        super().__init__()
        self.setWindowTitle("QT Inference Controller")
        
        # State management
        self.state = State.IDLE
        self.model_loaded = False
        self.calibration_loaded = False
        self.model_file_path = None
        self.calibration_file_path = None
        self.last_robot_commands = None
        
        # Components
        self.detector = None
        self.calibration_data = calibration_data
        self.coordinate_transformer = CoordinateTransformer()
        
        # Handle camera - create if not provided (for standalone use)
        if camera is None:
            self.camera = QTCamera(camera_index=0, area=(500, 0, 600, 600))
            self._owns_camera = True
            self.camera.set_fixed_display_size(400, 400) 
        else:
            self.camera = camera
            self._owns_camera = False
            # Get the effective resolution and set display size to 60% of it
            width, height = self.camera.get_effective_resolution()
            #self.camera.set_fixed_display_size(int(0.6 * width), int(0.6 * height))
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
                # Extract filepath if available
                if 'filepath' in calibration_data:
                    self.calibration_file_path = calibration_data['filepath']
                self.log(f"Calibration loaded from main controller")
                self.log(message)
        
        # Auto-load model and calibration if available
        #self.auto_load_files()      
        QTimer.singleShot(0, self.auto_load_files)  
       
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
                    self.model_file_path = str(model_file)  # When auto-loading
                    self.log(f"Auto-loaded model: {model_file}")
                except Exception as e:
                    self.log(f"Failed to auto-load model: {str(e)}")
        
        # Only auto-load calibration if not already loaded from constructor
        if not self.calibration_loaded:
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
                            self.calibration_file_path = str(calibration_file)  # When auto-loading
                            self.log(f"Auto-loaded calibration: {calibration_file}")
                            self.log(message)
                        else:
                            self.log(f"Failed to import auto-loaded calibration: {message}")

                        if 'camera_area' in self.calibration_data:
                            area = self.calibration_data['camera_area']
                            camera_area = (area['x'], area['y'], area['width'], area['height'])
                            self.camera.set_area(camera_area)
                            self.log(f"Camera area set to: {camera_area}")
                            # Emit signal to update main UI
                            self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])

                    except Exception as e:
                        self.log(f"Failed to auto-load calibration: {str(e)}")
        else:
            # Update UI to show calibration is loaded
            if self.calibration_data and 'name' in self.calibration_data:
                self.calibration_status_label.setText(f"Calibration: {self.calibration_data['name']}")
            else:
                self.calibration_status_label.setText("Calibration: Loaded")

        # Check if both are loaded and enable start button
        self.check_ready()

    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
             
        # Create horizontal layout for camera section and right panel
        content_layout = QHBoxLayout()
        
        # Left side - Camera section
        camera_section_layout = QVBoxLayout()
        
        # Camera controls (threshold, etc.) would go here if available from camera widget
        # You'll need to modify QTCamera to expose these controls separately
        
        # Camera widget
        # Add existing camera widget (don't create new one)
        self.camera.object_detected.connect(self.on_object_detected)
        camera_section_layout.addWidget(self.camera)
        
        # Control buttons below camera
        control_layout = QHBoxLayout()
        
        self.start_stop_btn = QPushButton("Start Detection")
        self.start_stop_btn.clicked.connect(self.toggle_detection)
        self.start_stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_stop_btn)
        
        self.load_model_btn = QPushButton("Load Detection Model")
        self.load_model_btn.clicked.connect(self.load_model)
        control_layout.addWidget(self.load_model_btn)
        
        self.load_calibration_btn = QPushButton("Load Calibration")
        self.load_calibration_btn.clicked.connect(self.load_calibration)
        control_layout.addWidget(self.load_calibration_btn)
        
        camera_section_layout.addLayout(control_layout)
        
        # Add camera section to content layout
        camera_section_widget = QWidget()
        camera_section_widget.setLayout(camera_section_layout)
        content_layout.addWidget(camera_section_widget)
        
        # Right panel with status and log
        right_panel_layout = QVBoxLayout()
        
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
        right_panel_layout.addWidget(status_group)
        
        # Log display
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(450)
        self.log_display.setMinimumHeight(150)  # Prevent stretching
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        log_group.setMaximumHeight(600)  # Prevent group box from stretching
        right_panel_layout.addWidget(log_group)
        
        # Add spacer to push everything to top
        right_panel_layout.addStretch()
        
        # Add right panel to content layout
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)
        right_panel_widget.setMaximumWidth(600)  # Optional: limit width of right panel
        content_layout.addWidget(right_panel_widget)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        central_widget.setLayout(main_layout)
        
        # Initialize with object detection disabled
        self.camera.set_object_detection_enabled(False)
        
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
                self.model_file_path = file_path  # Store the path
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
                    self.calibration_file_path = file_path 
                else:
                    self.log(f"Failed to import calibration: {message}")
                    
            except Exception as e:
                self.log(f"Failed to load calibration: {str(e)}")

        if self.calibration_data:  
            print("CaliDAta found")       
            if 'camera_area' in self.calibration_data:
                print("Camra found")
                area = self.calibration_data['camera_area']
                print(area)
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                self.log(f"Camera area set to: {camera_area}")
                # Emit signal to update main UI
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
 
                
    def update_calibration(self, calibration_data):
        """Update calibration data from external source (e.g., main controller)."""
        if calibration_data:
            success, message = self.coordinate_transformer.import_calibration(calibration_data)
            if success:
                self.calibration_data = calibration_data
                self.calibration_loaded = True
                # Extract filepath if available
                if 'filepath' in calibration_data:
                    self.calibration_file_path = calibration_data['filepath']

                # Update UI
                if 'name' in calibration_data:
                     self.calibration_status_label.setText(f"Calibration: {calibration_data['name']}")
                else:
                    self.calibration_status_label.setText("Calibration: Updated")
            
                # Set camera area if available in calibration data
                if 'camera_area' in calibration_data:
                    area = calibration_data['camera_area']
                    camera_area = (area['x'], area['y'], area['width'], area['height'])
                    self.camera.set_area(camera_area)
                    self.log(f"Camera area set to: {camera_area}")
                    # Emit signal to update main UI
                    self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                    
                    self.log("Calibration updated from main controller")
                    self.log(message)
                    self.check_ready()
                else:
                    self.log(f"Failed to update calibration: {message}")
        self.check_ready()
                
    def check_ready(self):
        """Check if both model and calibration are loaded."""
        print("Model",self.model_loaded,"Cali",self.calibration_loaded)
        if self.model_loaded and self.calibration_loaded:
            self.start_stop_btn.setEnabled(True)
            self.log("Ready to start detection")
        
            # Emit ready state with paths
            model_path = None
            if hasattr(self, 'model_file_path'):
                model_path = self.model_file_path
        
            calibration_path = None  
            if hasattr(self, 'calibration_file_path'):
                calibration_path = self.calibration_file_path
            
            self.ready_state_changed.emit(True, model_path, calibration_path)
        else:
            self.ready_state_changed.emit(False, None, None)
            
    def toggle_detection(self):
        """Start or stop motion detection."""
        if self.start_stop_btn.text() == "Start Detection":
        #if self.state == State.IDLE:
            # Start detection
            self.camera.set_object_detection_enabled(True)
            self.start_stop_btn.setText("Stop Detection")
            self.log("Motion detection started")
        else:
            # Stop detection
            print("Stop detection clicked")
            self.camera.set_object_detection_enabled(False)
            self.update_state(State.IDLE)
            self.start_stop_btn.setText("Start Detection")
            self.log("Motion detection stopped")
            
    def on_object_detected(self, regions):
        """Handle motion detection signal from camera."""
        if self.state == State.IDLE:
            self.update_state(State.WAITING)
            
            # Wait 4.5 second then capture
            QTimer.singleShot(3000, self.capture_and_process)

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
        print("capture and process started")
        self.camera.set_object_detection_enabled(False) 

        # Save frame temporarily for DINO processing
        temp_path = "temp_capture.png"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Run DINO inference
            self.log("Running DINO inference...")
            if self.detector is not None:
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
                    #QTimer.singleShot(8000, lambda: self.reset_to_idle(reset_ref_frame=True)) #in case we keep stuff on the table
            
            if robot_commands:
                print("Robot Commands",robot_commands,"Last Robot Commands",self.last_robot_commands)
                if not self.are_commands_similar(robot_commands, self.last_robot_commands): # dont do stuff twice
                    self.send_to_robot(robot_commands)
                    self.last_robot_commands = robot_commands
                else:
                    self.log("Already detected")
                    self.reset_to_idle()
            else:
                self.log("No objects detected")
                self.reset_to_idle()
                #QTimer.singleShot(8000, lambda: self.reset_to_idle(reset_ref_frame=True)) #in case we keep stuff on the table
                
        except Exception as e:
            self.log(f"Error during inference: {str(e)}")
            self.reset_to_idle()
        finally:
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def are_commands_similar(self, commands1, commands2, tolerance=15):
        """Check if two command sets are essentially the same."""
        if not commands1 or not commands2:
            return commands1 == commands2
    
        if len(commands1) != len(commands2):
            return False
    
        # Create sorted lists for comparison
        def command_key(cmd):
            return (cmd['object'], cmd['robot'][0], cmd['robot'][1])
    
        sorted1 = sorted(commands1, key=command_key)
        sorted2 = sorted(commands2, key=command_key)
    
        # Compare with tolerance for coordinates
        for c1, c2 in zip(sorted1, sorted2):
            if c1['object'] != c2['object']:
                return False
        
            # Check if robot coordinates are within tolerance
            dx = abs(c1['robot'][0] - c2['robot'][0])
            dy = abs(c1['robot'][1] - c2['robot'][1])
        
            if dx > tolerance or dy > tolerance:
                return False
    
        return True        

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
            elif 'jet' in cmd['object'].lower():
                steps.append(('process_red_plane', cmd['robot'][0], cmd['robot'][1]))
            elif 'bello' in cmd['object'].lower():
                steps.append(('process_blue_plane', cmd['robot'][0], cmd['robot'][1]))
            elif 'willi' in cmd['object'].lower():
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
        print("Robot error")
        self.camera.set_object_detection_enabled(False)
        self.start_stop_btn.setText("Start Detection")
        
    def reset_to_idle(self,reset_ref_frame=False):
        """Reset to idle state and re-enable motion detection."""
        self.update_state(State.IDLE)
        if self.start_stop_btn.text() == "Stop Detection":
            # Only re-enable if we're in "running" mode
            self.camera.set_object_detection_enabled(True,reset_ref_frame) #Set object detection, but dont reset reference frame (empty table)
            self.log("Motion detection re-enabled")
            
    def closeEvent(self, event):
        """Clean up on close."""
        # Only close camera if we created it
        if self._owns_camera and self.camera:
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