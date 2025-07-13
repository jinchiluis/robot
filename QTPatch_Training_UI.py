from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                               QSplitter, QStatusBar, QLabel, QScrollArea,
                               QGridLayout, QCheckBox, QFrame)
from PySide6.QtGui import QPixmap


class ThumbnailWidget(QFrame):
    """Widget for displaying a thumbnail with checkbox."""
    
    def __init__(self, pixmap, index):
        super().__init__()
        self.index = index
        self.setFrameStyle(QFrame.Box)
        self.setFixedSize(110, 130)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(False)
        layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)
        
        # Thumbnail
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        self.setLayout(layout)
        
    def isChecked(self):
        return self.checkbox.isChecked()


def setup_ui(controller):
    """Setup the user interface for QTPatch_Training_Controller."""
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
    camera_title = QLabel("PatchCore Training")
    camera_title.setAlignment(Qt.AlignCenter)
    camera_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
    left_layout.addWidget(camera_title)
    
    # Camera widget
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
    
    # Right side - Snapshots grid
    right_widget = QWidget()
    right_layout = QVBoxLayout()
    
    # Title and count
    title_layout = QHBoxLayout()
    snapshots_title = QLabel("Normal Samples")
    snapshots_title.setStyleSheet("font-size: 14px; font-weight: bold;")
    title_layout.addWidget(snapshots_title)
    
    controller.count_label = QLabel("Count: 0")
    controller.count_label.setStyleSheet("font-size: 14px;")
    title_layout.addWidget(controller.count_label)
    title_layout.addStretch()
    
    right_layout.addLayout(title_layout)
    
    # Scrollable area for thumbnails
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    # Container widget for grid
    controller.grid_container = QWidget()
    controller.grid_layout = QGridLayout()
    controller.grid_layout.setSpacing(10)
    controller.grid_container.setLayout(controller.grid_layout)
    
    scroll_area.setWidget(controller.grid_container)
    right_layout.addWidget(scroll_area)
    
    # Control buttons
    controls_layout = QHBoxLayout()
    
    controller.delete_btn = QPushButton("Delete Selected")
    controller.delete_btn.clicked.connect(controller.delete_selected)
    controls_layout.addWidget(controller.delete_btn)
    
    controller.clear_all_btn = QPushButton("Clear All")
    controller.clear_all_btn.clicked.connect(controller.clear_all)
    controls_layout.addWidget(controller.clear_all_btn)
    
    controller.params_btn = QPushButton("Parameters")
    controller.params_btn.clicked.connect(controller.set_parameters)
    controls_layout.addWidget(controller.params_btn)
    
    right_layout.addLayout(controls_layout)
    
    # Train and Save buttons
    action_layout = QHBoxLayout()
    
    controller.train_btn = QPushButton("Train Model")
    controller.train_btn.clicked.connect(controller.train_model)
    controller.train_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
    action_layout.addWidget(controller.train_btn)
    
    controller.save_btn = QPushButton("Save Model")
    controller.save_btn.clicked.connect(controller.save_model)
    controller.save_btn.setEnabled(False)
    controller.save_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
    action_layout.addWidget(controller.save_btn)
    
    right_layout.addLayout(action_layout)
    right_widget.setLayout(right_layout)
    
    # Add to splitter
    splitter.addWidget(left_widget)
    splitter.addWidget(right_widget)
    splitter.setSizes([600, 400])
    
    main_layout.addWidget(splitter)
    central_widget.setLayout(main_layout)
    
    # Status bar
    controller.status_bar = QStatusBar()
    controller.setStatusBar(controller.status_bar)
    
    # Add methods to controller for UI updates
    controller.update_snapshot_count = lambda: update_snapshot_count(controller)
    controller.update_snapshot_grid = lambda: update_snapshot_grid(controller)


def update_snapshot_count(controller):
    """Update the snapshot count label."""
    count = len(controller.snapshots)
    controller.count_label.setText(f"Count: {count}")
    
    # Update train button state
    controller.train_btn.setEnabled(count >= 5)


def update_snapshot_grid(controller):
    """Update the grid of snapshot thumbnails."""
    # Clear existing widgets
    while controller.grid_layout.count():
        child = controller.grid_layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    
    # Add thumbnails in a grid
    columns = 3
    for i, (filepath, thumbnail) in enumerate(controller.snapshots):
        row = i // columns
        col = i % columns
        
        # Create thumbnail widget with checkbox
        thumb_widget = ThumbnailWidget(thumbnail, i)
        controller.grid_layout.addWidget(thumb_widget, row, col)
    
    # Add stretch to push thumbnails to top-left
    controller.grid_layout.setRowStretch(controller.grid_layout.rowCount(), 1)
    controller.grid_layout.setColumnStretch(columns, 1)