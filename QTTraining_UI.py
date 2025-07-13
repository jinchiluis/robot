from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                               QTreeWidget, QSplitter, QStatusBar, QLabel)


def setup_ui(controller):
    """Setup the user interface for QTTraining_Controller."""
    # Central widget
    central_widget = QWidget()
    controller.setCentralWidget(central_widget)
    
    # Main horizontal layout with splitter
    main_layout = QHBoxLayout()
    splitter = QSplitter(Qt.Horizontal)
    
    # Left side - Camera
    left_widget = QWidget()
    left_layout = QVBoxLayout()
    
    # Camera title
    camera_title = QLabel("DINO Object Detection - Training")
    camera_title.setAlignment(Qt.AlignCenter)
    camera_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
    left_layout.addWidget(camera_title)
    
    # Camera widget - already initialized in __init__
    controller.camera.set_object_detection_enabled(True)
    left_layout.addWidget(controller.camera)
    
    # Camera buttons
    camera_buttons = QHBoxLayout()
    
    controller.snapshot_btn = QPushButton("Take Snapshot")
    controller.snapshot_btn.clicked.connect(controller.take_snapshot)
    camera_buttons.addWidget(controller.snapshot_btn)
    
    controller.load_calib_btn = QPushButton("Load Calibration")
    controller.load_calib_btn.clicked.connect(controller.load_calibration)
    camera_buttons.addWidget(controller.load_calib_btn)
    
    left_layout.addLayout(camera_buttons)
    left_widget.setLayout(left_layout)
    
    # Right side - Tree and controls
    right_widget = QWidget()
    right_layout = QVBoxLayout()
    
    # Tree widget
    controller.tree = QTreeWidget()
    controller.tree.setHeaderLabel("Training Data")
    controller.tree.setIconSize(QSize(64, 64))
    right_layout.addWidget(controller.tree)
    
    # Control buttons
    controls_layout = QHBoxLayout()
    
    controller.new_object_btn = QPushButton("New Object")
    controller.new_object_btn.clicked.connect(controller.new_object)
    controls_layout.addWidget(controller.new_object_btn)
    
    controller.add_snapshot_btn = QPushButton("Add Snapshot")
    controller.add_snapshot_btn.clicked.connect(controller.add_snapshot_to_object)
    controls_layout.addWidget(controller.add_snapshot_btn)
    
    controller.delete_btn = QPushButton("Delete")
    controller.delete_btn.clicked.connect(controller.delete_selected)
    controls_layout.addWidget(controller.delete_btn)
    
    right_layout.addLayout(controls_layout)
    
    # Train and Save buttons
    action_layout = QHBoxLayout()
    
    controller.train_btn = QPushButton("Train")
    controller.train_btn.clicked.connect(controller.train_model)
    action_layout.addWidget(controller.train_btn)
    
    controller.save_btn = QPushButton("Save")
    controller.save_btn.clicked.connect(controller.save_model)
    controller.save_btn.setEnabled(False)
    action_layout.addWidget(controller.save_btn)
    
    controller.clear_all_btn = QPushButton("Clear All")
    controller.clear_all_btn.clicked.connect(controller.clear_all)
    action_layout.addWidget(controller.clear_all_btn)
    
    right_layout.addLayout(action_layout)
    right_widget.setLayout(right_layout)
    
    # Add to splitter
    splitter.addWidget(left_widget)
    splitter.addWidget(right_widget)
    splitter.setSizes([600, 600])
    
    main_layout.addWidget(splitter)
    central_widget.setLayout(main_layout)
    
    # Status bar
    controller.status_bar = QStatusBar()
    controller.setStatusBar(controller.status_bar)
    
    # Store last training result
    controller.last_training_result = None
    controller.trainer = None


def update_ui_state(controller):
    """Update UI elements based on current state."""
    # This function can be expanded to handle UI state updates if needed
    pass