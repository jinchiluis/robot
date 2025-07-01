import sys
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QMessageBox, 
                               QInputDialog, QFileDialog, QProgressDialog, QDialog,
                               QLineEdit, QDialogButtonBox, QLabel, QSlider,
                               QHBoxLayout, QVBoxLayout, QFormLayout)

from QTCamera import QTCamera
from patchcore_exth import SimplePatchCore
from QTPatch_Training_UI import setup_ui


class TrainingThread(QThread):
    """Thread for training PatchCore model without blocking UI."""
    finished = Signal(bool)
    error = Signal(str)
    status = Signal(str)
    
    def __init__(self, train_dir, val_dir, sample_ratio, threshold_percentile, backbone):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.sample_ratio = sample_ratio
        self.threshold_percentile = threshold_percentile
        self.backbone = backbone
        self.model = None
        
    def run(self):
        try:
            self.status.emit("Initializing PatchCore model...")
            self.model = SimplePatchCore(backbone=self.backbone)
            
            self.status.emit("Training on normal samples...")
            self.model.fit(
                train_dir=self.train_dir,
                val_dir=self.val_dir,
                sample_ratio=self.sample_ratio,
                threshold_percentile=self.threshold_percentile
            )
            
            self.finished.emit(True)
        except Exception as e:
            self.error.emit(str(e))


class SaveModelDialog(QDialog):
    """Simple dialog for saving model with name."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Save Model")
        self.setModal(True)
        self.resize(400, 120)
        
        layout = QVBoxLayout()
        
        # Model name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model name:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def get_model_name(self):
        """Get the entered model name."""
        return self.name_input.text().strip()


class TrainingParametersDialog(QDialog):
    """Dialog for setting training parameters."""
    
    def __init__(self, parent, sample_ratio=0.15, threshold_percentile=99):
        super().__init__(parent)
        self.setWindowTitle("Training Parameters")
        self.setModal(True)
        self.resize(400, 200)
        
        layout = QFormLayout()
        
        # Sample ratio slider (1-30%)
        self.sample_ratio_slider = QSlider(Qt.Horizontal)
        self.sample_ratio_slider.setMinimum(1)
        self.sample_ratio_slider.setMaximum(100)
        self.sample_ratio_slider.setValue(int(sample_ratio * 100))
        self.sample_ratio_label = QLabel(f"{int(sample_ratio * 100)}%")
        self.sample_ratio_slider.valueChanged.connect(
            lambda v: self.sample_ratio_label.setText(f"{v}%")
        )
        
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(self.sample_ratio_slider)
        sample_layout.addWidget(self.sample_ratio_label)
        layout.addRow("Sample Ratio:", sample_layout)
        
        # Threshold percentile slider (90-100%)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(90)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(threshold_percentile)
        self.threshold_label = QLabel(f"{threshold_percentile}%")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v}%")
        )
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        layout.addRow("Threshold Percentile:", threshold_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
        
    def get_parameters(self):
        """Get the selected parameters."""
        return {
            'sample_ratio': self.sample_ratio_slider.value() / 100.0,
            'threshold_percentile': self.threshold_slider.value()
        }


class QTPatch_Training_Controller(QMainWindow):
    """Training controller for PatchCore anomaly detection model."""
    
    # Signal for camera area changes
    camera_area_changed = Signal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, camera=None, calibration_data=None):
        super().__init__()
        self.setWindowTitle("PatchCore Training")
        self.setGeometry(100, 100, 1000, 700)
        
        # Data storage
        self.snapshots = []  # List of (filepath, thumbnail) tuples
        self.snapshot_counter = 0
        self.temp_dir = Path("temp_patchcore_training")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create train and val subdirectories
        self.train_dir = self.temp_dir / "train"
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir = self.temp_dir / "val"
        self.val_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.model_dir = Path("patchcore_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.sample_ratio = 0.30  # 30% default... 1%: ~95% AUROC//10%: ~99% AUROC//25%: ~99.5% AUROC//50%: ~99.6% AUROC
        self.threshold_percentile = 99  # 99% default
        self.backbone = 'wide_resnet50_2'  # Default backbone
        
        # Calibration
        self.calibration_data = calibration_data
        self.calibration_loaded = False
        
        # Handle camera
        if camera is None:
            self.camera = QTCamera(camera_index=0)
            self._owns_camera = True
            self.camera.set_fixed_display_size(600, 600)
        else:
            self.camera = camera
            self._owns_camera = False
            self.camera.set_fixed_display_size(600, 600)
        
        # Initialize UI
        setup_ui(self)
        self.update_status("Ready")
        
        # Load calibration data if provided
        if calibration_data:
            self.calibration_loaded = True
            self.update_status("Calibration loaded from main controller")
            
            if 'camera_area' in calibration_data:
                area = calibration_data['camera_area']
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
        
        # Auto-load calibration if available
        QTimer.singleShot(0, self.auto_load_calibration)
        
        # Training state
        self.trained_model = None
        
    def auto_load_calibration(self):
        """Automatically load calibration file from default folder."""
        if not self.calibration_loaded:
            calibration_folder = Path("calibration")
            if calibration_folder.exists():
                calibration_files = list(calibration_folder.glob("*.json"))
                if calibration_files:
                    calibration_file = max(calibration_files, key=lambda f: f.stat().st_mtime)
                    try:
                        import json
                        with open(calibration_file, 'r') as f:
                            self.calibration_data = json.load(f)
                        
                        self.calibration_loaded = True
                        self.update_status(f"Auto-loaded calibration: {calibration_file.name}")
                        
                        if 'camera_area' in self.calibration_data:
                            area = self.calibration_data['camera_area']
                            camera_area = (area['x'], area['y'], area['width'], area['height'])
                            self.camera.set_area(camera_area)
                            self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                            
                    except Exception as e:
                        self.update_status(f"Failed to auto-load calibration: {str(e)}")
    
    def update_status(self, message):
        """Update status bar."""
        self.status_bar.showMessage(message)
        
    def load_calibration(self):
        """Load camera calibration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)")
        
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    self.calibration_data = json.load(f)
                
                self.calibration_loaded = True
                self.update_status(f"Calibration loaded: {Path(file_path).name}")
                
                QMessageBox.information(self, "Success", "Calibration loaded successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load calibration: {str(e)}")
            
            if self.calibration_data and 'camera_area' in self.calibration_data:
                area = self.calibration_data['camera_area']
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
    
    def update_calibration(self, calibration_data):
        """Update calibration data from external source."""
        if calibration_data:
            self.calibration_data = calibration_data
            self.calibration_loaded = True
            
            if 'name' in calibration_data:
                self.update_status(f"Calibration updated: {calibration_data['name']}")
            else:
                self.update_status("Calibration updated from main controller")
            
            if 'camera_area' in calibration_data:
                area = calibration_data['camera_area']
                camera_area = (area['x'], area['y'], area['width'], area['height'])
                self.camera.set_area(camera_area)
                self.camera_area_changed.emit(area['x'], area['y'], area['width'], area['height'])
                
    def take_snapshot(self):
        """Take a snapshot and add to collection."""
        # Get frame WITHOUT boxes (include_boxes=False)
        frame = self.camera.get_current_frame(include_boxes=False)
        if frame is None:
            return
        
        # Check if we have a detected region
        if not self.camera.detected_regions:
            QMessageBox.warning(self, "Warning", "No object detected in frame")
            return
        
        # Get the first detected region
        x, y, w, h = self.camera.detected_regions[0]
        
        # Crop to detected region only
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Save cropped image (this is what will be used for training)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"normal_{self.snapshot_counter}_{timestamp}.png"
        filepath = self.temp_dir / filename
        cv2.imwrite(str(filepath), cropped_frame)
        self.snapshot_counter += 1
        
        # Create thumbnail from the same cropped region
        rgb_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        thumbnail = QPixmap.fromImage(qt_image).scaled(100, 100, Qt.KeepAspectRatio, 
                                                  Qt.SmoothTransformation)
        
        # Add to snapshots
        self.snapshots.append((str(filepath), thumbnail))
        
        # Update UI
        self.update_snapshot_grid()
        self.update_snapshot_count()
        self.update_status(f"Snapshot taken: {filename}")
        
        # Update UI
        self.update_snapshot_grid()
        self.update_snapshot_count()
        self.update_status(f"Snapshot taken: {filename}")
        
    def delete_selected(self):
        """Delete selected snapshots."""
        selected_indices = []
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'isChecked') and widget.isChecked():
                selected_indices.append(i)
        
        if not selected_indices:
            QMessageBox.warning(self, "Warning", "No snapshots selected")
            return
            
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Delete {len(selected_indices)} selected snapshot(s)?")
        if reply == QMessageBox.Yes:
            # Delete in reverse order to maintain indices
            for i in reversed(selected_indices):
                if i < len(self.snapshots):
                    filepath, _ = self.snapshots[i]
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    del self.snapshots[i]
            
            self.update_snapshot_grid()
            self.update_snapshot_count()
            self.update_status(f"Deleted {len(selected_indices)} snapshots")
            
    def clear_all(self):
        """Clear all snapshots after confirmation."""
        if not self.snapshots:
            return
            
        reply = QMessageBox.question(self, "Confirm Clear All", 
                                   "Delete all snapshots?")
        if reply == QMessageBox.Yes:
            # Delete all files
            for filepath, _ in self.snapshots:
                try:
                    os.remove(filepath)
                except:
                    pass
                    
            # Clear data
            self.snapshots.clear()
            
            # Update UI
            self.update_snapshot_grid()
            self.update_snapshot_count()
            self.update_status("Cleared all snapshots")
            
    def set_parameters(self):
        """Show dialog to set training parameters."""
        dialog = TrainingParametersDialog(self, self.sample_ratio, self.threshold_percentile)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            self.sample_ratio = params['sample_ratio']
            self.threshold_percentile = params['threshold_percentile']
            self.update_status(f"Parameters updated: Sample ratio={self.sample_ratio:.0%}, "
                             f"Threshold={self.threshold_percentile}%")
            
    def train_model(self):
        """Train PatchCore model with 80/20 train/val split."""
        if len(self.snapshots) < 5:
            QMessageBox.warning(self, "Warning", 
                              "Need at least 5 snapshots to train (for 80/20 split)")
            return
            
        # Clear previous train/val directories
        for f in self.train_dir.glob("*.png"):
            f.unlink()
        for f in self.val_dir.glob("*.png"):
            f.unlink()
            
        # Split data 80/20
        n_samples = len(self.snapshots)
        n_train = int(n_samples * 0.8)
        
        # Shuffle for random split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Copy files to train/val directories
        for i, idx in enumerate(train_indices):
            src_path, _ = self.snapshots[idx]
            dst_path = self.train_dir / f"train_{i:04d}.png"
            import shutil
            shutil.copy2(src_path, dst_path)
            
        for i, idx in enumerate(val_indices):
            src_path, _ = self.snapshots[idx]
            dst_path = self.val_dir / f"val_{i:04d}.png"
            import shutil
            shutil.copy2(src_path, dst_path)
        
        # Show progress dialog
        progress = QProgressDialog("Training PatchCore model...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Update progress text
        def update_progress(text):
            progress.setLabelText(text)
            QApplication.processEvents()
        
        # Train in thread
        self.training_thread = TrainingThread(
            str(self.train_dir),
            str(self.val_dir),
            self.sample_ratio,
            self.threshold_percentile,
            self.backbone
        )
        
        self.training_thread.status.connect(update_progress)
        self.training_thread.finished.connect(lambda s: self.on_training_finished(s, progress))
        self.training_thread.error.connect(lambda e: self.on_training_error(e, progress))
        self.training_thread.start()
        
    def on_training_finished(self, success, progress):
        """Handle training completion."""
        progress.close()
        
        if success:
            # Store the trained model
            self.trained_model = self.training_thread.model
            
            # Enable save button
            self.save_btn.setEnabled(True)
            
            # Show results
            msg = f"Training completed!\n\n"
            msg += f"Training samples: {len(list(self.train_dir.glob('*.png')))}\n"
            msg += f"Validation samples: {len(list(self.val_dir.glob('*.png')))}\n"
            msg += f"Memory bank size: {self.trained_model.memory_bank.shape}\n"
            msg += f"Threshold: {self.trained_model.global_threshold:.6f}"
            
            QMessageBox.information(self, "Training Complete", msg)
            self.update_status("Training completed successfully")
            
    def on_training_error(self, error, progress):
        """Handle training error."""
        progress.close()
        QMessageBox.critical(self, "Training Error", f"Training failed: {error}")
        self.update_status("Training failed")
        
    def save_model(self):
        """Save the trained model."""
        if not self.trained_model:
            return
            
        # Show save dialog
        dialog = SaveModelDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
            
        # Get model name
        name = dialog.get_model_name()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a model name")
            return
            
        # Check if file exists
        filepath = self.model_dir / f"{name}.pth"
        if filepath.exists():
            reply = QMessageBox.question(self, "File Exists", 
                                       f"Model '{name}' already exists. Overwrite?")
            if reply != QMessageBox.Yes:
                return
                
        # Save model
        try:
            self.trained_model.save(str(filepath))
            QMessageBox.information(self, "Success", f"Model saved as '{name}.pth'")
            self.update_status(f"Model saved: {name}.pth")
            self.save_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
            
    def closeEvent(self, event):
        """Clean up on close."""
        # Clean up temp files
        for filepath, _ in self.snapshots:
            try:
                os.remove(filepath)
            except:
                pass
                
        # Clean up temp directories
        try:
            for f in self.train_dir.glob("*.png"):
                f.unlink()
            for f in self.val_dir.glob("*.png"):
                f.unlink()
            self.train_dir.rmdir()
            self.val_dir.rmdir()
            self.temp_dir.rmdir()
        except:
            pass
        
        # Only close camera if we created it
        if self._owns_camera and self.camera:
            self.camera.close()
            
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = QTPatch_Training_Controller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()