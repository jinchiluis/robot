from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QBrush, QPen, QColor, QPixmap, QImage
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QListWidget, QSpinBox,
                               QMessageBox, QInputDialog, QGroupBox, QFrame, QGridLayout, QStatusBar)


class ClickableLabel(QLabel):
    """Label that can detect mouse clicks and hover with coordinates."""
    
    clicked = Signal(int, int)
    mouse_moved = Signal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.points = []  # List of (x, y, color) tuples
        self.hover_pos = None
        self.image_offset = (0, 0)  # Offset of actual image within label
        self.image_scale = 1.0  # Scale factor of the image
        self.actual_image_size = None  # Actual size of displayed image
        
    def set_image_transform(self, offset_x, offset_y, scale, actual_width, actual_height):
        """Set the image transformation parameters for coordinate mapping."""
        self.image_offset = (offset_x, offset_y)
        self.image_scale = scale
        self.actual_image_size = (actual_width, actual_height)
        
    def widget_to_image_coords(self, widget_x, widget_y):
        """Convert widget coordinates to image coordinates."""
        if self.actual_image_size is None:
            return widget_x, widget_y
            
        # Check if click is within the actual image area
        offset_x, offset_y = self.image_offset
        actual_width, actual_height = self.actual_image_size
        
        if (widget_x < offset_x or widget_x > offset_x + actual_width or
            widget_y < offset_y or widget_y > offset_y + actual_height):
            return None, None  # Click is outside image area
            
        # Convert to image coordinates
        image_x = int((widget_x - offset_x) / self.image_scale)
        image_y = int((widget_y - offset_y) / self.image_scale)
        
        return image_x, image_y
        
    def image_to_widget_coords(self, image_x, image_y):
        """Convert image coordinates to widget coordinates."""
        widget_x = int(image_x * self.image_scale + self.image_offset[0])
        widget_y = int(image_y * self.image_scale + self.image_offset[1])
        return widget_x, widget_y
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: # type: ignore
            # Convert widget coordinates to image coordinates
            image_x, image_y = self.widget_to_image_coords(event.x(), event.y())
            if image_x is not None and image_y is not None:
                self.clicked.emit(image_x, image_y)
            
    def mouseMoveEvent(self, event):
        self.hover_pos = (event.x(), event.y())
        # Convert to image coordinates for display
        image_x, image_y = self.widget_to_image_coords(event.x(), event.y())
        if image_x is not None and image_y is not None:
            self.mouse_moved.emit(image_x, image_y)
        self.update()
        
    def leaveEvent(self, event):
        self.hover_pos = None
        self.update()
        
    def add_point(self, x, y, color=QColor(255, 0, 0)):
        """Add a point to be drawn (in image coordinates)."""
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
        
        # Draw points (convert from image to widget coordinates)
        for x, y, color in self.points:
            widget_x, widget_y = self.image_to_widget_coords(x, y)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(widget_x - 5, widget_y - 5, 10, 10)
            
        # Draw hover coordinates
        if self.hover_pos:
            widget_x, widget_y = self.hover_pos
            image_x, image_y = self.widget_to_image_coords(widget_x, widget_y)
            
            if image_x is not None and image_y is not None:
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                text = f"({image_x}, {image_y})"
                rect = painter.fontMetrics().boundingRect(text)
                rect.adjust(-2, -2, 2, 2)
                rect.moveTopLeft(QPoint(widget_x + 10, widget_y - 10))  # Offset from cursor
                painter.drawRect(rect)
                painter.drawText(rect, Qt.AlignCenter, text)

        painter.end()


def setup_ui(controller):
    """Setup the user interface."""
    main_layout = QHBoxLayout()
    
    # Left side - Image display
    left_layout = QVBoxLayout()
    
    # Camera title
    camera_title = QLabel("Calibration")
    camera_title.setAlignment(Qt.AlignCenter)
    camera_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 2px;")
    camera_title.setMaximumHeight(30)  # Prevent stretching
    left_layout.addWidget(camera_title)
    
    # Image label
    controller.image_label = ClickableLabel()
    controller.image_label.setFixedSize(640, 480)
    # Remove setScaledContents - we'll handle scaling manually
    controller.image_label.setStyleSheet("border: 2px solid gray;")
    controller.image_label.clicked.connect(controller.on_image_clicked)
    controller.image_label.mouse_moved.connect(controller.on_mouse_moved)
    left_layout.addWidget(controller.image_label)

    camera_controls_layout = QHBoxLayout()
    
    # X coordinate
    camera_controls_layout.addWidget(QLabel("X-Value:"))
    controller.x_spinbox = QSpinBox()
    controller.x_spinbox.setRange(0, 9999)
    controller.x_spinbox.setValue(controller.camera_area_x)
    camera_controls_layout.addWidget(controller.x_spinbox)
    
    # Y coordinate
    camera_controls_layout.addWidget(QLabel("Y-Value:"))
    controller.y_spinbox = QSpinBox()
    controller.y_spinbox.setRange(0, 9999)
    controller.y_spinbox.setValue(controller.camera_area_y)
    camera_controls_layout.addWidget(controller.y_spinbox)
    
    # Width
    camera_controls_layout.addWidget(QLabel("Width:"))
    controller.width_spinbox = QSpinBox()
    controller.width_spinbox.setRange(100, 9999)
    controller.width_spinbox.setValue(controller.camera_area_width)
    camera_controls_layout.addWidget(controller.width_spinbox)
    
    # Height
    camera_controls_layout.addWidget(QLabel("Height:"))
    controller.height_spinbox = QSpinBox()
    controller.height_spinbox.setRange(100, 9999)
    controller.height_spinbox.setValue(controller.camera_area_height)
    camera_controls_layout.addWidget(controller.height_spinbox)
    
    # Set Camera Area button
    controller.set_area_button = QPushButton("Set Area")
    controller.set_area_button.clicked.connect(controller.update_camera_area)
    camera_controls_layout.addWidget(controller.set_area_button)
    
    left_layout.addLayout(camera_controls_layout)

    # Control buttons
    button_layout = QHBoxLayout()
    controller.freeze_btn = QPushButton("Start Calibration")
    controller.freeze_btn.clicked.connect(controller.toggle_freeze)
    button_layout.addWidget(controller.freeze_btn)
    controller.freeze_btn.setMaximumWidth(300)
    
    controller.reset_btn = QPushButton("Reset")
    controller.reset_btn.clicked.connect(controller.reset_calibration)
    button_layout.addWidget(controller.reset_btn)  
    controller.reset_btn.setMaximumWidth(300)
    
    left_layout.addLayout(button_layout)
    
    # Right side - Controls
    right_layout = QVBoxLayout()
    right_widget = QWidget()
    right_widget.setMinimumWidth(300)
    right_widget.setMaximumWidth(600)
    right_widget.setLayout(right_layout)

    
    # Current point input
    current_group = QGroupBox("Current Point")
    current_layout = QVBoxLayout()
    
    coord_layout = QHBoxLayout()
    coord_layout.addWidget(QLabel("Robot X:"))
    controller.robot_x_input = QLineEdit("0")
    coord_layout.addWidget(controller.robot_x_input)
    coord_layout.addWidget(QLabel("Y:"))
    controller.robot_y_input = QLineEdit("0")
    coord_layout.addWidget(controller.robot_y_input)
    current_layout.addLayout(coord_layout)

    create_directional_control(current_layout, controller)
    
    controller.add_point_btn = QPushButton("Add Point")
    controller.add_point_btn.clicked.connect(controller.add_calibration_point)
    current_layout.addWidget(controller.add_point_btn)
    
    controller.current_pixel_label = QLabel("Click on image...")
    current_layout.addWidget(controller.current_pixel_label)
    
    current_group.setLayout(current_layout)
    right_layout.addWidget(current_group)
    
    # Points list
    points_group = QGroupBox("Calibration Points")
    points_layout = QVBoxLayout()
    
    controller.points_list = QListWidget()
    points_layout.addWidget(controller.points_list)
    
    controller.delete_point_btn = QPushButton("Delete Selected")
    controller.delete_point_btn.clicked.connect(controller.delete_selected_point)
    points_layout.addWidget(controller.delete_point_btn)
    
    controller.calculate_btn = QPushButton("Calculate Transformation")
    controller.calculate_btn.clicked.connect(controller.calculate_transformation)
    points_layout.addWidget(controller.calculate_btn)
    
    controller.points_count_label = QLabel("Points: 0/4")
    points_layout.addWidget(controller.points_count_label)
    
    points_group.setLayout(points_layout)
    right_layout.addWidget(points_group)
    
    # Test mode
    test_group = QGroupBox("Test Mode")
    test_layout = QVBoxLayout()
    
    controller.test_info_label = QLabel("Calculate transformation first")
    test_layout.addWidget(controller.test_info_label)
    
    controller.test_result_label = QLabel("Result: -")
    test_layout.addWidget(controller.test_result_label)
    
    test_group.setLayout(test_layout)
    right_layout.addWidget(test_group)
    
    # Save button
    controller.save_btn = QPushButton("Save Calibration")
    controller.save_btn.clicked.connect(controller.save_calibration)
    right_layout.addWidget(controller.save_btn)
    
    right_layout.addStretch()
    
    # Combine layouts
    left_widget = QWidget()
    left_widget.setLayout(left_layout)
    main_layout.addWidget(left_widget)
    main_layout.addWidget(right_widget)
    
    # Create central widget for main window
    central_widget = QWidget()
    central_widget.setLayout(main_layout)
    controller.setCentralWidget(central_widget)
    
    # Status bar (matching training/inference)
    controller.status_bar = QStatusBar()
    controller.setStatusBar(controller.status_bar)
    
    # Start live view update
    controller.startTimer(30)  # 30ms timer for updating display

def create_directional_control(parent_layout, controller):
    """Create a 3x3 grid of directional control buttons."""
    control_frame = QFrame()
    control_frame.setMaximumSize(150, 150)
    grid_layout = QGridLayout(control_frame)
    grid_layout.setSpacing(2)
        
    # Button configurations: (row, col, direction, symbol) 
    buttons = [
        (0, 0, "UL", "↖"),
        (0, 1, "U", "↑"),
        (0, 2, "UR", "↗"),
        (1, 0, "L", "←"),
        (1, 2, "R", "→"),
        (2, 0, "DL", "↙"),
        (2, 1, "D", "↓"),
        (2, 2, "DR", "↘")
    ]
        
    for row, col, direction, symbol in buttons:
        btn = QPushButton(symbol)
        btn.setFixedSize(40, 40)
        btn.setFocusPolicy(Qt.NoFocus)  # Prevent stealing keyboard focus
        btn.pressed.connect(lambda d=direction: controller.on_direction_pressed(d))
        btn.released.connect(controller.on_direction_released)
        grid_layout.addWidget(btn, row, col)
        
    # Add empty center
    center_label = QLabel()
    center_label.setFixedSize(40, 40)
    grid_layout.addWidget(center_label, 1, 1)
        
    parent_layout.addWidget(control_frame)
    parent_layout.addWidget(QLabel("Use arrows or WASD+QEZC keys"))

def display_frame(controller, frame):
    """Display a frame in the image label with aspect ratio preservation."""
    import cv2
    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    pixmap = QPixmap.fromImage(qt_image)
    
    # Get label dimensions
    label_width = controller.image_label.width()
    label_height = controller.image_label.height()
    
    # Calculate scaling to fit within label while preserving aspect ratio
    scale_x = label_width / w
    scale_y = label_height / h
    scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
    
    # Calculate actual scaled size
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    
    # Scale the pixmap
    scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, 
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    # Create a new pixmap with label size and black background
    final_pixmap = QPixmap(label_width, label_height)
    final_pixmap.fill(Qt.black)
    
    # Calculate position to center the scaled image
    x_offset = (label_width - scaled_width) // 2
    y_offset = (label_height - scaled_height) // 2
    
    # Draw the scaled image onto the final pixmap
    painter = QPainter(final_pixmap)
    painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
    painter.end()
    
    # Update the label's transformation parameters
    controller.image_label.set_image_transform(x_offset, y_offset, scale, scaled_width, scaled_height)
    
    # Set the pixmap
    controller.image_label.setPixmap(final_pixmap)


def update_ui_state(controller):
    """Update UI elements based on current state."""
    # Enable/disable based on state
    is_frozen = controller.state in [controller.STATE_FROZEN, controller.STATE_CALIBRATING]
    is_calibrating = controller.state == controller.STATE_CALIBRATING
    is_testing = controller.state in [controller.STATE_TESTING, controller.STATE_COMPLETE]
    
    controller.add_point_btn.setEnabled(is_frozen and controller.current_point is not None)
    controller.robot_x_input.setEnabled(is_frozen)
    controller.robot_y_input.setEnabled(is_frozen)
    controller.delete_point_btn.setEnabled(is_frozen and controller.points_list.count() > 0)
    controller.calculate_btn.setEnabled(is_calibrating)
    controller.save_btn.setEnabled(is_testing)
    
    # Update status display
    status_text = f"State: {controller.state}"
    controller.setWindowTitle(f"Robot Calibration - {status_text}")