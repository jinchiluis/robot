from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QTextEdit, QGroupBox, QStatusBar, QFrame,
                               QSlider, QDoubleSpinBox)


def setup_ui(controller):
    """Setup the user interface for QTPatch_Inference_Controller."""
    central_widget = QWidget()
    controller.setCentralWidget(central_widget)
    
    # Main horizontal layout for camera section and right panel
    main_layout = QHBoxLayout()
    
    # Left side - Camera section
    camera_section_layout = QVBoxLayout()
    
    # Camera title
    camera_title = QLabel("PatchCore Anomaly Detection")
    camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    camera_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
    camera_section_layout.addWidget(camera_title)
    
    # Camera widget
    controller.camera.object_detected.connect(controller.on_object_detected)
    if controller.camera is not None:
        camera_section_layout.addWidget(controller.camera)
    else:
        print("Camera is None")
    
    # Control buttons below camera
    control_layout = QHBoxLayout()
    
    controller.start_stop_btn = QPushButton("Start Detection")
    controller.start_stop_btn.clicked.connect(controller.toggle_detection)
    controller.start_stop_btn.setEnabled(False)
    control_layout.addWidget(controller.start_stop_btn)
    
    controller.load_model_btn = QPushButton("Load PatchCore Model")
    controller.load_model_btn.clicked.connect(controller.load_model)
    control_layout.addWidget(controller.load_model_btn)
    
    controller.load_calibration_btn = QPushButton("Load Calibration")
    controller.load_calibration_btn.clicked.connect(controller.load_calibration)
    control_layout.addWidget(controller.load_calibration_btn)
    
    camera_section_layout.addLayout(control_layout)
    
    # Add camera section to main layout
    left_widget = QWidget()
    left_widget.setLayout(camera_section_layout)
    main_layout.addWidget(left_widget)
    
    # Right panel with status and log
    right_panel_layout = QVBoxLayout()
    
    # Status display
    status_group = QGroupBox("Status")
    status_layout = QVBoxLayout()
    
    controller.state_label = QLabel(f"State: {controller.state.value}")
    status_layout.addWidget(controller.state_label)
    
    controller.model_status_label = QLabel("Model: Not loaded")
    status_layout.addWidget(controller.model_status_label)
    
    controller.calibration_status_label = QLabel("Calibration: Not loaded")
    status_layout.addWidget(controller.calibration_status_label)
    
    # Separator line
    separator = QFrame()
    separator.setFrameStyle(QFrame.Shape.HLine | QFrame.Shadow.Sunken)
    status_layout.addWidget(separator)
    
    # Anomaly detection results
    anomaly_result_layout = QVBoxLayout()
    
    controller.anomaly_status_label = QLabel("Status: -")
    controller.anomaly_status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
    anomaly_result_layout.addWidget(controller.anomaly_status_label)
    
    controller.anomaly_score_label = QLabel("Score: -")
    anomaly_result_layout.addWidget(controller.anomaly_score_label)
    
    controller.threshold_label = QLabel("Threshold: -")
    anomaly_result_layout.addWidget(controller.threshold_label)
    
    #Heatmap display
    controller.heatmap_label = QLabel("No heatmap")
    controller.heatmap_label.setMinimumWidth(250)
    controller.heatmap_label.setMaximumWidth(250)
    controller.heatmap_label.setMaximumHeight(250)
    #controller.heatmap_label.setScaledContents(True)
    controller.heatmap_label.setStyleSheet("border: 1px solid #ccc;")
    controller.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    anomaly_result_layout.addWidget(controller.heatmap_label)

    status_layout.addLayout(anomaly_result_layout)
    
    status_group.setLayout(status_layout)
    right_panel_layout.addWidget(status_group)
    
    # Settings group
    settings_group = QGroupBox("Settings")
    settings_layout = QVBoxLayout()
    
    # Robot actions
    action_layout = QHBoxLayout()
    action_layout.addWidget(QLabel("Normal Action:"))
    controller.normal_action_label = QLabel(controller.normal_action)
    action_layout.addWidget(controller.normal_action_label)
    action_layout.addStretch()
    
    action_layout.addWidget(QLabel("Anomaly Action:"))
    controller.anomaly_action_label = QLabel(controller.anomaly_action)
    controller.anomaly_action_label.setStyleSheet("color: red;")
    action_layout.addWidget(controller.anomaly_action_label)
    action_layout.addStretch()
    
    settings_layout.addLayout(action_layout)
    
    settings_group.setLayout(settings_layout)
    right_panel_layout.addWidget(settings_group)
    
    # Log display
    log_group = QGroupBox("Log")
    log_layout = QVBoxLayout()
    
    controller.log_display = QTextEdit()
    controller.log_display.setReadOnly(True)
    controller.log_display.setMaximumHeight(350)
    controller.log_display.setMinimumHeight(150)
    log_layout.addWidget(controller.log_display)
    
    log_group.setLayout(log_layout)
    log_group.setMaximumHeight(450)
    right_panel_layout.addWidget(log_group)
    
    # Add spacer to push everything to top
    right_panel_layout.addStretch()
    
    # Add right panel to main layout
    right_panel_widget = QWidget()
    right_panel_widget.setLayout(right_panel_layout)
    right_panel_widget.setMaximumWidth(600)
    main_layout.addWidget(right_panel_widget)
    
    central_widget.setLayout(main_layout)
    
    # Status bar
    controller.status_bar = QStatusBar()
    controller.setStatusBar(controller.status_bar)
    
    # Initialize with object detection disabled
    controller.camera.set_object_detection_enabled(False)


def update_ui_state(controller):
    """Update UI elements based on current state."""
    # Update status display
    controller.state_label.setText(f"State: {controller.state.value}")
    controller.setWindowTitle(f"QT PatchCore Controller - {controller.state.value}")
    
    # Update status bar with current state
    if controller.state.name == "PROCESSING":
        controller.status_bar.showMessage("Running PatchCore inference...", 2000)
    elif controller.state.name == "ROBOT_MOVING":
        controller.status_bar.showMessage("Robot executing command...", 2000)
    elif controller.state.name == "WAITING":
        controller.status_bar.showMessage("Waiting to capture...", 2000)