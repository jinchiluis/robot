import sys
import cv2 #must import!!! because camera inside
import json
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget, 
                               QMenuBar, QMenu, QStatusBar, QScrollArea,
                               QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSpinBox, QMessageBox, QInputDialog)

from QTCamera import QTCamera
from QTCalibration_Controller import QTCalibration_Controller
from QTInference_Controller import QTInference_Controller
from QTTraining_Controller import QTTraining_Controller
from QTConfiguration_Manager import QTConfiguration_Manager


class QTMain_Controller(QMainWindow):
    """Main controller that manages camera, calibration data, and page switching."""
    
    view_switched = Signal(str)  

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT Robot Vision System")
        self.setGeometry(100, 100, 1100, 900)
        
        # Shared resources
        self.camera = QTCamera(camera_index=0)
        #self.camera = QTCamera(camera_index=0, area=(500, 0, 600, 600))
        self.camera.set_fixed_display_size(600, 600)
        self.calibration_data = None

        # Configuration manager
        self.config_manager = QTConfiguration_Manager()

        # Setup UI
        self.setup_ui()
        
        # Connect signals
        self.calibration_controller.calibration_saved.connect(self.on_calibration_saved)
        
    def setup_ui(self):
        """Setup the main UI with menu and stacked widget."""
        # Create main central widget with vertical layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        
        # Create camera area controls
        self.create_camera_controls()
        central_layout.addWidget(self.camera_controls_widget)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        central_layout.addWidget(self.stacked_widget)
        
        # Set the layout and central widget
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        
        # Create calibration controller with shared camera
        self.calibration_controller = QTCalibration_Controller(
            camera_widget=self.camera,
            calibration_data=self.calibration_data
        )
        
        # Inference controller will be created lazily
        self.inference_controller = None
        self.inference_scroll_area = None
        
        # Training controller will be created lazily
        self.training_controller = None
        self.training_scroll_area = None
        
        # Add calibration to stacked widget
        self.stacked_widget.addWidget(self.calibration_controller)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Connect signals
        self.calibration_controller.calibration_saved.connect(self.on_calibration_saved)
        
        # Start with calibration page
        self.switch_to_calibration()
        
    def create_camera_controls(self):
        """Create the camera area control widgets."""
        self.camera_controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        
        self.area_status_label = QLabel("")
        controls_layout.addWidget(self.area_status_label)

        # X coordinate
        controls_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(0, 9999)
        self.x_spinbox.setValue(500)
        controls_layout.addWidget(self.x_spinbox)
        
        # Y coordinate
        controls_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(0, 9999)
        self.y_spinbox.setValue(0)
        controls_layout.addWidget(self.y_spinbox)
        
        # Width
        controls_layout.addWidget(QLabel("Width:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(100, 9999)
        self.width_spinbox.setValue(600)
        controls_layout.addWidget(self.width_spinbox)
        
        # Height
        controls_layout.addWidget(QLabel("Height:"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(100, 9999)
        self.height_spinbox.setValue(600)
        controls_layout.addWidget(self.height_spinbox)
        
        # Set Camera Area button
        self.set_area_button = QPushButton("Set Camera Area")
        self.set_area_button.clicked.connect(self.update_camera_area)
        controls_layout.addWidget(self.set_area_button)
        
        # Add stretch to push everything to the left
        controls_layout.addStretch()
        
        self.camera_controls_widget.setLayout(controls_layout)
        
    def update_camera_area(self):
        """Update the camera area based on spinbox values."""
        x = self.x_spinbox.value()
        y = self.y_spinbox.value()
        width = self.width_spinbox.value()
        height = self.height_spinbox.value()
        
        new_area = (x, y, width, height)
        self.camera.set_area(new_area)
        
        self.status_bar.showMessage(f"Camera area updated to: {new_area}", 3000)
        
    def create_menu_bar(self):
        """Create the menu bar with navigation options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")

        # Navigation menu
        nav_menu = menubar.addMenu("Navigation")
        
        # Calibration action
        calibration_action = QAction("Calibration", self)
        calibration_action.setShortcut("Ctrl+1")
        calibration_action.triggered.connect(self.switch_to_calibration)
        nav_menu.addAction(calibration_action)
        
        # Training action
        training_action = QAction("Training", self)
        training_action.setShortcut("Ctrl+2")
        training_action.triggered.connect(self.switch_to_training)
        nav_menu.addAction(training_action)

        # Inference action
        inference_action = QAction("Inference", self)
        inference_action.setShortcut("Ctrl+3")
        inference_action.triggered.connect(self.switch_to_inference)
        nav_menu.addAction(inference_action)

        # Load configuration action
        self.load_config_action = QAction("Load Configuration", self)
        self.load_config_action.setShortcut("Ctrl+O")
        self.load_config_action.triggered.connect(self.load_configuration)
        file_menu.addAction(self.load_config_action)
        
        # Save configuration action (disabled for now)
        self.save_config_action = QAction("Save Configuration", self)
        self.save_config_action.setShortcut("Ctrl+S")
        self.save_config_action.triggered.connect(self.save_configuration)
        self.save_config_action.setEnabled(False)  # Will implement later
        file_menu.addAction(self.save_config_action)
        
        file_menu.addSeparator()   

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def load_configuration(self):
        """Load a saved configuration."""
        config_path = self.config_manager.show_load_dialog(self)
        if not config_path:
            return
            
        config = self.config_manager.load_configuration(config_path)
        if not config:
            QMessageBox.critical(self, "Error", "Failed to load configuration")
            return
            
        # Show warnings if any files are missing
        if '_warnings' in config:
            warning_msg = "Configuration loaded with warnings:\n\n"
            warning_msg += "\n".join(config['_warnings'])
            QMessageBox.warning(self, "Configuration Warnings", warning_msg)
            
        # Load calibration if specified
        if config.get('calibration') and Path(config['calibration']).exists():
            try:
                with open(config['calibration'], 'r') as f:
                    self.calibration_data = json.load(f)
                    
                # Update calibration controller
                success, message = self.calibration_controller.transformer.import_calibration(self.calibration_data)
                if success:
                    self.calibration_controller.state = self.calibration_controller.STATE_COMPLETE
                    self.calibration_controller.update_ui_state()
                    
                    # Update camera area if present
                    if 'camera_area' in self.calibration_data:
                        area = self.calibration_data['camera_area']
                        self.camera.set_area((area['x'], area['y'], area['width'], area['height']))
                        self.x_spinbox.setValue(area['x'])
                        self.y_spinbox.setValue(area['y'])
                        self.width_spinbox.setValue(area['width'])
                        self.height_spinbox.setValue(area['height'])
                        
                    self.status_bar.showMessage(f"Loaded calibration: {Path(config['calibration']).name}", 3000)
                    
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to load calibration: {str(e)}")
                
        # Update inference controller if exists
        if self.inference_controller and config.get('calibration'):
            self.inference_controller.update_calibration(self.calibration_data)
            
        # Load model through inference controller
        if self.inference_controller and config.get('model') and Path(config['model']).exists():
            try:
                from dino_object_detector import ObjectDetector
                self.inference_controller.detector = ObjectDetector(config['model'])
                self.inference_controller.model_loaded = True
                self.inference_controller.model_status_label.setText(f"Model: {Path(config['model']).name}")
                self.inference_controller.log(f"Model loaded from configuration: {config['model']}")
                self.inference_controller.check_ready()
                self.status_bar.showMessage(f"Loaded model: {Path(config['model']).name}", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to load model: {str(e)}")
                
        # Store robot type for future use
        if config.get('robot_type'):
            self.robot_type = config['robot_type']
            
        self.status_bar.showMessage(f"Configuration loaded: {config['name']}", 5000)
        
    def save_configuration(self):
        """Save current configuration."""
        if not hasattr(self, 'current_model_path') or not hasattr(self, 'current_calibration_path'):
            QMessageBox.warning(self, "Cannot Save", "No model or calibration loaded")
            return
        
        name, ok = QInputDialog.getText(self, "Save Configuration", "Configuration name:")
        if not ok or not name:
            return
        
        success, message = self.config_manager.save_configuration(
            name, 
            self.current_calibration_path,
            self.current_model_path
        )
    
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            if "already exists" in message:
                # Ask for different name
                self.save_configuration()
            else:
                QMessageBox.critical(self, "Error", message)

    def switch_to_calibration(self):
        """Switch to calibration page."""
        self.stacked_widget.setCurrentWidget(self.calibration_controller)
        self.status_bar.showMessage("Calibration Mode")

        # Disable load configuration in calibration mode
        self.load_config_action.setEnabled(False)

        # Re-enable camera area controls for calibration mode
        self.x_spinbox.setEnabled(True)
        self.y_spinbox.setEnabled(True)
        self.width_spinbox.setEnabled(True)
        self.height_spinbox.setEnabled(True)
        self.set_area_button.setEnabled(True)
        self.area_status_label.setText("")
        #self.area_status_label.setStyleSheet("color: black;")
        self.view_switched.emit("Calibration")
        
    def switch_to_inference(self):
        """Switch to inference page."""
        # Remove camera from its current parent before switching
        #self.camera.setParent(None) 

        # Lazy initialization of inference controller
        #self.inference_controller = None
        if self.inference_controller is None:
            self.inference_controller = QTInference_Controller(
                x = self.x_spinbox.value(),
                y = self.y_spinbox.value(),
                width = self.width_spinbox.value(),
                height = self.height_spinbox.value(),
                camera=self.camera,
                calibration_data=self.calibration_data
            )
            # Connect to camera area change signal
            self.inference_controller.camera_area_changed.connect(self.on_camera_area_changed)
            self.inference_controller.ready_state_changed.connect(self.on_inference_ready_changed)
            
            # Create scroll area and add the inference controller to it
            self.inference_scroll_area = QScrollArea()
            self.inference_scroll_area.setWidget(self.inference_controller)
            self.inference_scroll_area.setWidgetResizable(True)
            self.inference_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.inference_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Add the scroll area to the stacked widget instead of the controller directly
            self.stacked_widget.addWidget(self.inference_scroll_area)

            # Connect the view_switched signal to a method in inference controller
            self.view_switched.connect(self.inference_controller.on_view_switched)
            
        # Disable manual controls when area comes from calibration
        self.x_spinbox.setEnabled(False)
        self.y_spinbox.setEnabled(False)
        self.width_spinbox.setEnabled(False)
        self.height_spinbox.setEnabled(False)
        self.set_area_button.setEnabled(False)
      
        # Update inference controller with latest calibration if available
        if self.calibration_data:
            self.inference_controller.update_calibration(self.calibration_data)
            # Update camera area spinboxes if area info is in calibration
            if 'camera_area' in self.calibration_data:
                area = self.calibration_data['camera_area']
                self.x_spinbox.setValue(area['x'])
                self.y_spinbox.setValue(area['y'])
                self.width_spinbox.setValue(area['width'])
                self.height_spinbox.setValue(area['height'])
                self.area_status_label.setText("(from calibration)")
                self.area_status_label.setStyleSheet("color: green;")
        
        self.stacked_widget.setCurrentWidget(self.inference_scroll_area)
        self.load_config_action.setEnabled(True)
        self.status_bar.showMessage("Inference Mode")

        self.view_switched.emit("Inference")

    def switch_to_training(self):
        """Switch to training page."""
        # Remove camera from its current parent before switching
        #self.camera.setParent(None)

        # Lazy initialization of training controller every time
        #self.training_controller = None
        if self.training_controller is None:
            print("3. Creating training controller...")
            self.training_controller = QTTraining_Controller(
                camera=self.camera,
                calibration_data=self.calibration_data
            )
            # Connect to camera area change signal
            self.training_controller.camera_area_changed.connect(self.on_camera_area_changed)
            
            # Create scroll area and add the training controller to it
            self.training_scroll_area = QScrollArea()
            self.training_scroll_area.setWidget(self.training_controller)
            self.training_scroll_area.setWidgetResizable(True)
            self.training_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.training_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Add the scroll area to the stacked widget instead of the controller directly
            self.stacked_widget.addWidget(self.training_scroll_area)

        # Disable manual controls when area comes from calibration
        self.x_spinbox.setEnabled(False)
        self.y_spinbox.setEnabled(False)
        self.width_spinbox.setEnabled(False)
        self.height_spinbox.setEnabled(False)
        self.set_area_button.setEnabled(False)
        
        # Update training controller with latest calibration if available
        if self.calibration_data:
            self.training_controller.update_calibration(self.calibration_data)
            # Update camera area spinboxes if area info is in calibration
            if 'camera_area' in self.calibration_data:
                area = self.calibration_data['camera_area']
                self.x_spinbox.setValue(area['x'])
                self.y_spinbox.setValue(area['y'])
                self.width_spinbox.setValue(area['width'])
                self.height_spinbox.setValue(area['height'])
                self.area_status_label.setText("(from calibration)")
                self.area_status_label.setStyleSheet("color: green;")
        
        self.stacked_widget.setCurrentWidget(self.training_scroll_area)
        self.load_config_action.setEnabled(False)
        self.status_bar.showMessage("Training Mode")

        self.view_switched.emit("Training")
        
    def on_calibration_saved(self, calibration_data):
        """Handle calibration saved signal."""
        self.calibration_data = calibration_data
        if 'filepath' in calibration_data:
            self.calibration_file_path = calibration_data['filepath']
        self.status_bar.showMessage("Calibration saved and loaded", 3000)

        # Update camera area spinboxes if area info is in calibration
        if 'camera_area' in calibration_data:
            area = calibration_data['camera_area']
            self.x_spinbox.setValue(area['x'])
            self.y_spinbox.setValue(area['y'])
            self.width_spinbox.setValue(area['width'])
            self.height_spinbox.setValue(area['height'])
            self.area_status_label.setText("(from calibration)")
            self.area_status_label.setStyleSheet("color: green;")
            
    def on_camera_area_changed(self, x, y, width, height):
        """Handle camera area change from inference/training controller, when calibration loaded."""
        self.x_spinbox.setValue(x)
        self.y_spinbox.setValue(y)
        self.width_spinbox.setValue(width)
        self.height_spinbox.setValue(height)
        #self.area_status_label.setText("(from calibration)")
        #self.area_status_label.setStyleSheet("color: green;")
    
    def on_inference_ready_changed(self, is_ready, model_path, calibration_path):
        """Handle inference controller ready state change."""
        if is_ready and self.stacked_widget.currentWidget() == self.inference_scroll_area:
            # Only enable save if we're actually in inference mode
            self.save_config_action.setEnabled(True)
            # Store paths for save dialog
            self.current_model_path = model_path
            self.current_calibration_path = calibration_path
        else:
            self.save_config_action.setEnabled(False)

    def closeEvent(self, event):
        """Clean up on close."""
        # Camera will be closed by the controller that owns it
        if self.camera:
            self.camera.close()
        event.accept()

def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = QTMain_Controller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()