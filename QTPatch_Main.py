import sys
import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget, 
                               QStatusBar, QScrollArea)

from QTPatch_Training_Controller import QTPatch_Training_Controller
from QTPatch_Inference_Controller import QTPatch_Inference_Controller
from QTCalibration_Controller import QTCalibration_Controller


class QTMain_Controller(QMainWindow):
    """Simple main controller for PatchCore training and inference."""
    
    view_switched = Signal(str)  # Signal when switching views

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore Vision System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Setup UI
        self.setup_ui()
        
        # Start with training page
        self.switch_to_training()
        
    def setup_ui(self):
        """Setup the main UI with menu and stacked widget."""
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create controllers
        self.training_controller = QTPatch_Training_Controller()
        self.inference_controller = QTPatch_Inference_Controller(0,0,1920,1080)
        self.calibration_controller = QTCalibration_Controller()
        
        # Create scroll areas for each controller
        self.training_scroll = QScrollArea()
        self.training_scroll.setWidget(self.training_controller)
        self.training_scroll.setWidgetResizable(True)
        self.training_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.training_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.inference_scroll = QScrollArea()
        self.inference_scroll.setWidget(self.inference_controller)
        self.inference_scroll.setWidgetResizable(True)
        self.inference_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.inference_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.calibration_scroll = QScrollArea()
        self.calibration_scroll.setWidget(self.calibration_controller)
        self.calibration_scroll.setWidgetResizable(True)
        self.calibration_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.calibration_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Add to stacked widget
        self.stacked_widget.addWidget(self.training_scroll)
        self.stacked_widget.addWidget(self.inference_scroll)
        self.stacked_widget.addWidget(self.calibration_scroll)
        
        # Connect calibration saved signal if needed
        #self.calibration_controller.calibration_saved.connect(self.on_calibration_saved)
        
        # Connect view_switched signal to inference controller
        #self.view_switched.connect(self.inference_controller.on_view_switched)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        #self.status_bar = QStatusBar()
        #self.setStatusBar(self.status_bar)
        #self.status_bar.showMessage("Ready")
        
    def create_menu_bar(self):
        """Create the menu bar with navigation options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Navigation menu
        nav_menu = menubar.addMenu("Navigation")
        
        # Training action
        training_action = QAction("Training", self)
        training_action.setShortcut("Ctrl+1")
        training_action.triggered.connect(self.switch_to_training)
        nav_menu.addAction(training_action)

        # Inference action
        inference_action = QAction("Inference", self)
        inference_action.setShortcut("Ctrl+2")
        inference_action.triggered.connect(self.switch_to_inference)
        nav_menu.addAction(inference_action)
        
        # Calibration action
        calibration_action = QAction("Calibration", self)
        calibration_action.setShortcut("Ctrl+3")
        calibration_action.triggered.connect(self.switch_to_calibration)
        nav_menu.addAction(calibration_action)
        
    def switch_to_training(self):
        """Switch to training page."""
        self.stacked_widget.setCurrentWidget(self.training_scroll)
        self.training_controller.auto_load_calibration() #in case we do view switch from calibration     
        #self.status_bar.showMessage("Training Mode - Capture normal samples")
        self.setWindowTitle("PatchCore Vision System - Training")
        self.view_switched.emit("Training")
        
    def switch_to_inference(self):
        """Switch to inference page."""
        self.stacked_widget.setCurrentWidget(self.inference_scroll)
        self.inference_controller.auto_load_files() #in case we do view switch from training
        #self.status_bar.showMessage("Inference Mode - Detect anomalies")
        self.setWindowTitle("PatchCore Vision System - Inference")
        self.view_switched.emit("PatchCore_Inference")
        
    def switch_to_calibration(self):
        """Switch to calibration page."""
        self.stacked_widget.setCurrentWidget(self.calibration_scroll)
        #self.status_bar.showMessage("Calibration Mode - Setup coordinate transformation")
        self.setWindowTitle("PatchCore Vision System - Calibration")
        self.view_switched.emit("Calibration")
        
    # def on_calibration_saved(self, calibration_data):
    #     """Handle calibration saved signal."""
    #     print(f"Calibration saved: {calibration_data.get('name', 'Unknown')}")
    #     # You could pass this to the inference controller if needed
    #     # self.inference_controller.load_calibration(calibration_data)
        
    def closeEvent(self, event):
        """Clean up on close."""
        # Controllers will handle their own cleanup
        event.accept()


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = QTMain_Controller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()