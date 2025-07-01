import sys
import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                               QVBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QMessageBox, QInputDialog, QFileDialog, QLabel,
                               QProgressDialog, QSplitter, QStatusBar)

from QTCamera import QTCamera
from dino_object_detector import OneShotObjectTrainer


class TrainingThread(QThread):
    """Thread for training the model without blocking UI."""
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, training_data, save_debug=False):
        super().__init__()
        self.training_data = training_data
        self.save_debug = save_debug
        
    def run(self):
        try:
            trainer = OneShotObjectTrainer()
            results = trainer.train(self.training_data, objects_per_image=1, 
                                  save_debug_images=self.save_debug)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class QTTraining_Controller(QMainWindow):
    # Add this signal
    camera_area_changed = Signal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, camera=None, calibration_data=None):
        super().__init__()
        self.setWindowTitle("Training Data Collection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.snapshots = {}  # {object: [(image_path, thumbnail_pixmap), ...]}
        self.snapshot_counter = 0
        self.temp_dir = Path("temp_training_images")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)
        
        # Tree item references for efficient updates
        self.object_items = {}  # {object_name: QTreeWidgetItem}
        
        # Calibration
        self.calibration_data = calibration_data
        self.calibration_loaded = False
        
        # Handle camera - create if not provided (for standalone use)
        if camera is None:
            self.camera = QTCamera(camera_index=0)
            self._owns_camera = True
            self.camera.set_fixed_display_size(600, 600)
        else:
            self.camera = camera
            self._owns_camera = False
            # Keep the display size as set by main app or use 600x600
            self.camera.set_fixed_display_size(600, 600)
        
        self.setup_ui()
        self.update_status("Ready")
        
        # Load calibration data if provided
        if calibration_data:
            self.calibration_loaded = True
            self.update_status("Calibration loaded from main controller")
            
            if 'camera_area' in calibration_data:
                area = calibration_data['camera_area']
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                # Emit signal to update main UI
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
        
        # Auto-load calibration if available
        QTimer.singleShot(0, self.auto_load_calibration)
        
    def auto_load_calibration(self):
        """Automatically load calibration file from default folder."""
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
                        
                        self.calibration_loaded = True
                        self.update_status(f"Auto-loaded calibration: {calibration_file.name}")
                        
                        if 'camera_area' in self.calibration_data:
                            area = self.calibration_data['camera_area']
                            camera_area = (area['x'], area['y'], area['width'], area['height'])
                            self.camera.set_area(camera_area)
                            # Emit signal to update main UI
                            self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                            
                    except Exception as e:
                        self.update_status(f"Failed to auto-load calibration: {str(e)}")
        
    def setup_ui(self):
        """Setup the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout with splitter
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Camera
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Camera widget - already initialized in __init__
        self.camera.set_object_detection_enabled(True)
        left_layout.addWidget(self.camera)
        
        # Camera buttons
        camera_buttons = QHBoxLayout()
        
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        camera_buttons.addWidget(self.snapshot_btn)
        
        self.load_calib_btn = QPushButton("Load Calibration")
        self.load_calib_btn.clicked.connect(self.load_calibration)
        camera_buttons.addWidget(self.load_calib_btn)
        
        left_layout.addLayout(camera_buttons)
        left_widget.setLayout(left_layout)
        
        # Right side - Tree and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Training Data")
        self.tree.setIconSize(QSize(64, 64))
        right_layout.addWidget(self.tree)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.new_object_btn = QPushButton("New Object")
        self.new_object_btn.clicked.connect(self.new_object)
        controls_layout.addWidget(self.new_object_btn)
        
        self.add_snapshot_btn = QPushButton("Add Snapshot")
        self.add_snapshot_btn.clicked.connect(self.add_snapshot_to_object)
        controls_layout.addWidget(self.add_snapshot_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)
        controls_layout.addWidget(self.delete_btn)
        
        right_layout.addLayout(controls_layout)
        
        # Train and Save buttons
        action_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_model)
        action_layout.addWidget(self.train_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all)
        action_layout.addWidget(self.clear_all_btn)
        
        right_layout.addLayout(action_layout)
        right_widget.setLayout(right_layout)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Store last training result
        self.last_training_result = None
        self.trainer = None
        
    def update_status(self, message):
        """Update status bar."""
        self.status_bar.showMessage(message)
        
    def load_calibration(self):
        """Load camera calibration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)")
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.calibration_data = json.load(f)
                
                self.calibration_loaded = True
                self.update_status(f"Calibration loaded: {Path(file_path).name}")
                
                QMessageBox.information(self, "Success", "Calibration loaded successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load calibration: {str(e)}")
            
            # Apply camera area settings outside try block like in inference controller
            if self.calibration_data:
                if 'camera_area' in self.calibration_data:
                    area = self.calibration_data['camera_area']
                    print("cali", self.calibration_data['camera_area'])
                    camera_area = (area['x'], area['y'], area['width'], area['height'])
                    self.camera.set_area(camera_area)
                    self.update_status(f"Camera area set to: {camera_area}")
                    # Emit signal to update main UI
                    self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
    
    def update_calibration(self, calibration_data):
        """Update calibration data from external source (e.g., main controller)."""
        if calibration_data:
            self.calibration_data = calibration_data
            self.calibration_loaded = True
            
            # Update status
            if 'name' in calibration_data:
                self.update_status(f"Calibration updated: {calibration_data['name']}")
            else:
                self.update_status("Calibration updated from main controller")
            
            # Set camera area if available in calibration data
            if 'camera_area' in calibration_data:
                area = calibration_data['camera_area']
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                # Emit signal to update main UI
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                
    def take_snapshot(self):
        """Take a snapshot and temporarily store it."""
        frame = self.camera.get_current_frame(include_boxes=True)
        if frame is None:
            return
            
        # Save full image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{self.snapshot_counter}_{timestamp}.png"
        filepath = self.temp_dir / filename
        cv2.imwrite(str(filepath), frame)
        self.snapshot_counter += 1
        
        # Create thumbnail from green box regions
        if self.camera.detected_regions:
            x, y, w, h = self.camera.detected_regions[0]
            thumbnail_region = frame[y:y+h, x:x+w]
        
            # Convert to QPixmap for display
            rgb_image = cv2.cvtColor(thumbnail_region, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            thumbnail = QPixmap.fromImage(qt_image).scaled(64, 64, Qt.KeepAspectRatio, 
                                                      Qt.SmoothTransformation)
        else:
            # No green box, use placeholder
            thumbnail = QPixmap(64, 64)
            thumbnail.fill(Qt.gray)
            
        # Add to selected object or store temporarily
        selected = self.tree.currentItem()
        if selected and selected.parent() is None:  # Object selected
            object_name = selected.text(0).split(' (')[0]  # Remove count from object name
            self.add_snapshot(object_name, str(filepath), thumbnail)
        else:
            self.update_status(f"Snapshot saved: {filename} (select object to add)")
            # Store temporarily
            if "_temp_snapshots" not in self.snapshots:
                self.snapshots["_temp_snapshots"] = []
            self.snapshots["_temp_snapshots"].append((str(filepath), thumbnail))
            
    def add_snapshot_to_object(self):
        """Add the last snapshot to selected object."""
        selected = self.tree.currentItem()
        if not selected or selected.parent() is not None:
            QMessageBox.warning(self, "Warning", "Please select an object")
            return
            
        if "_temp_snapshots" not in self.snapshots or not self.snapshots["_temp_snapshots"]:
            QMessageBox.warning(self, "Warning", "No snapshot available. Take a snapshot first.")
            return
            
        object_name = selected.text(0).split(' (')[0]  # Remove count from object name
        filepath, thumbnail = self.snapshots["_temp_snapshots"].pop()
        self.add_snapshot(object_name, filepath, thumbnail)
        
    def add_snapshot(self, object_name, filepath, thumbnail):
        """Add a snapshot to an object."""
        if object_name not in self.snapshots:
            self.snapshots[object_name] = []
        self.snapshots[object_name].append((filepath, thumbnail))
        
        # Update tree - just add the new image, don't rebuild everything
        if object_name in self.object_items:
            # Object already exists in tree, just add the image
            object_item = self.object_items[object_name]
            
            # Add new image as child
            image_count = len(self.snapshots[object_name])
            image_item = QTreeWidgetItem([f"Image {image_count}"])
            image_item.setIcon(0, QIcon(thumbnail))
            object_item.addChild(image_item)
            
            # Update object label with new count
            object_item.setText(0, f"{object_name} ({image_count})")
        else:
            # Object doesn't exist in tree yet, do full update
            self.update_tree()
            
        self.update_status(f"Added snapshot to {object_name}")
        
    def new_object(self):
        """Create a new object."""
        name, ok = QInputDialog.getText(self, "New Object", "Object name:")
        if ok and name:
            if name in self.snapshots:
                QMessageBox.warning(self, "Warning", "Object already exists")
                return
                
            self.snapshots[name] = []
            
            # Add to tree
            object_item = QTreeWidgetItem([f"{name} (0)"])
            self.tree.addTopLevelItem(object_item)
            self.object_items[name] = object_item
            object_item.setExpanded(True)
            
            # Select the new object
            self.tree.setCurrentItem(object_item)
            
            self.update_status(f"Created object: {name}")
            
    def rename_object(self):
        """Rename selected object."""
        selected = self.tree.currentItem()
        if not selected or selected.parent() is not None:
            return
            
        old_name = selected.text(0).split(' (')[0]  # Remove count
        new_name, ok = QInputDialog.getText(self, "Rename Object", 
                                           "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            if new_name in self.snapshots:
                QMessageBox.warning(self, "Warning", "Object already exists")
                return
                
            # Update data
            self.snapshots[new_name] = self.snapshots.pop(old_name)
            
            # Update tree references
            self.object_items[new_name] = self.object_items.pop(old_name)
            
            # Update tree item text
            count = len(self.snapshots[new_name])
            selected.setText(0, f"{new_name} ({count})")
            
            self.update_status(f"Renamed object: {old_name} → {new_name}")
            
    def delete_selected(self):
        """Delete selected item."""
        selected = self.tree.currentItem()
        if not selected:
            return
            
        if selected.parent() is None:
            # Object selected
            object_name = selected.text(0).split(' (')[0]  # Remove count
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Delete object '{object_name}' and all its images?")
            if reply == QMessageBox.Yes:
                # Delete image files
                for filepath, _ in self.snapshots[object_name]:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                        
                # Remove from data and tree
                del self.snapshots[object_name]
                del self.object_items[object_name]
                
                # Remove from tree widget
                index = self.tree.indexOfTopLevelItem(selected)
                self.tree.takeTopLevelItem(index)
                
                self.update_status(f"Deleted object: {object_name}")
        else:
            # Image selected
            object_item = selected.parent()
            object_name = object_item.text(0).split(' (')[0]  # Remove count
            index = object_item.indexOfChild(selected)
            filepath, _ = self.snapshots[object_name][index]
            
            # Delete file
            try:
                os.remove(filepath)
            except:
                pass
                
            # Remove from data
            del self.snapshots[object_name][index]
            
            # Remove from tree
            object_item.removeChild(selected)
            
            # Update object label count
            count = len(self.snapshots[object_name])
            object_item.setText(0, f"{object_name} ({count})")
            
            # Re-number remaining images
            for i in range(object_item.childCount()):
                object_item.child(i).setText(0, f"Image {i+1}")
            
            self.update_status(f"Deleted image from {object_name}")
            
    def clear_all(self):
        """Clear all data after confirmation."""
        reply = QMessageBox.question(self, "Confirm Clear All", 
                                   "Delete all objects and images?")
        if reply == QMessageBox.Yes:
            # Delete all files
            for object_name in self.snapshots:
                for filepath, _ in self.snapshots[object_name]:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                        
            # Clear data
            self.snapshots.clear()
            self.object_items.clear()
            
            # Clear tree
            self.tree.clear()
            
            self.update_status("Cleared all data")
            
    def update_tree(self):
        """Fully rebuild the tree widget."""
        self.tree.clear()
        self.object_items.clear()
        
        for object_name, images in self.snapshots.items():
            if object_name == "_temp_snapshots":
                continue
                
            # Create object item
            object_item = QTreeWidgetItem([f"{object_name} ({len(images)})"])
            self.tree.addTopLevelItem(object_item)
            self.object_items[object_name] = object_item
            
            # Add images
            for i, (filepath, thumbnail) in enumerate(images):
                image_item = QTreeWidgetItem([f"Image {i+1}"])
                image_item.setIcon(0, QIcon(thumbnail))
                object_item.addChild(image_item)
                
            object_item.setExpanded(True)
            
    def train_model(self):
        """Train the model with current data."""
        # Check if we have data
        valid_objects = {k: v for k, v in self.snapshots.items() 
                          if k != "_temp_snapshots" and v}
        
        if len(valid_objects) < 2:
            QMessageBox.warning(self, "Warning", 
                              "Need at least 2 objects with images to train")
            return
            
        # Prepare training data
        training_data = {}
        for object_name, images in valid_objects.items():
            training_data[object_name] = [filepath for filepath, _ in images]
            
        # Show progress dialog
        progress = QProgressDialog("Training model...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Train in thread
        self.training_thread = TrainingThread(training_data)
        self.training_thread.finished.connect(lambda r: self.on_training_finished(r, progress))
        self.training_thread.error.connect(lambda e: self.on_training_error(e, progress))
        self.training_thread.start()
        
    def on_training_finished(self, results, progress):
        """Handle training completion."""
        progress.close()
        
        # Store results
        self.last_training_result = results
        self.trainer = OneShotObjectTrainer()
        self.trainer.prototypes = results['prototypes']
        self.trainer.config = results['config']
        
        # Enable save button
        self.save_btn.setEnabled(True)
        
        # Show results
        msg = "Training completed!\n\n"
        msg += "Feature counts:\n"
        for obj_type, count in results['feature_counts'].items():
            msg += f"  {obj_type}: {count} features\n"
            
        msg += "\nPrototype similarities:\n"
        prototypes = results['prototypes']
        obj_types = list(prototypes.keys())
        for i, type1 in enumerate(obj_types):
            for j, type2 in enumerate(obj_types):
                if i < j:
                    sim = np.dot(prototypes[type1], prototypes[type2])
                    msg += f"  {type1} ↔ {type2}: {sim:.3f}\n"
                    
        QMessageBox.information(self, "Training Results", msg)
        self.update_status("Training completed")
        
    def on_training_error(self, error, progress):
        """Handle training error."""
        progress.close()
        QMessageBox.critical(self, "Training Error", f"Training failed: {error}")
        self.update_status("Training failed")
        
    def save_model(self):
        """Save the trained model."""
        if not self.trainer:
            return
            
        # Get model name
        name, ok = QInputDialog.getText(self, "Save Model", "Model name:")
        if not ok or not name:
            return
            
        # Check if file exists
        filepath = self.model_dir / f"{name}.pkl"
        if filepath.exists():
            reply = QMessageBox.question(self, "File Exists", 
                                       f"Model '{name}' already exists. Overwrite?")
            if reply != QMessageBox.Yes:
                # Ask for new name
                self.save_model()
                return
                
        # Save model
        try:
            self.trainer.save_model(str(filepath))
            QMessageBox.information(self, "Success", f"Model saved as '{name}.pkl'")
            self.update_status(f"Model saved: {name}.pkl")
            self.save_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
            
    def closeEvent(self, event):
        """Clean up on close."""
        # Clean up temp files
        for object_name in self.snapshots:
            for filepath, _ in self.snapshots[object_name]:
                try:
                    os.remove(filepath)
                except:
                    pass
        
        # Only close camera if we created it
        if self._owns_camera and self.camera:
            self.camera.close()
            
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = QTTraining_Controller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()