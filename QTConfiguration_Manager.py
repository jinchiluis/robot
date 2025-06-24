import json
import os
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                               QListWidgetItem, QPushButton, QLabel, QMessageBox,
                               QGroupBox, QDialogButtonBox)


class ConfigurationDialog(QDialog):
    """Dialog for selecting a configuration to load."""
    
    def __init__(self, configurations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Configuration")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        
        self.selected_config = None
        self.configurations = configurations
        
        self.setup_ui()
        self.populate_list()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout()
        
        # List of configurations
        self.config_list = QListWidget()
        self.config_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.config_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.config_list)
        
        # Details group
        self.details_group = QGroupBox("Configuration Details")
        details_layout = QVBoxLayout()
        
        self.name_label = QLabel("Name: -")
        self.created_label = QLabel("Created: -")
        self.model_label = QLabel("Model: -")
        self.calibration_label = QLabel("Calibration: -")
        self.robot_label = QLabel("Robot Type: -")
        
        details_layout.addWidget(self.name_label)
        details_layout.addWidget(self.created_label)
        details_layout.addWidget(self.model_label)
        details_layout.addWidget(self.calibration_label)
        details_layout.addWidget(self.robot_label)
        details_layout.addStretch()
        
        self.details_group.setLayout(details_layout)
        layout.addWidget(self.details_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def populate_list(self):
        """Populate the list with available configurations."""
        for config_path, config_data in self.configurations:
            item = QListWidgetItem()
            
            # Main text with name
            item.setText(config_data['name'])
            
            # Store full path and data
            item.setData(Qt.UserRole, config_path)
            item.setData(Qt.UserRole + 1, config_data)
            
            # Add created date as tooltip
            if 'created' in config_data:
                created_str = config_data['created']
                # Parse and format the date nicely
                try:
                    created_dt = QDateTime.fromString(created_str, Qt.ISODate)
                    formatted_date = created_dt.toString("yyyy-MM-dd hh:mm")
                    item.setToolTip(f"Created: {formatted_date}")
                except:
                    item.setToolTip(f"Created: {created_str}")
            
            self.config_list.addItem(item)
            
    def on_selection_changed(self):
        """Handle selection change."""
        current_item = self.config_list.currentItem()
        if current_item:
            config_data = current_item.data(Qt.UserRole + 1)
            self.update_details(config_data)
            self.ok_button.setEnabled(True)
        else:
            self.clear_details()
            self.ok_button.setEnabled(False)
            
    def update_details(self, config_data):
        """Update the details panel."""
        self.name_label.setText(f"Name: {config_data.get('name', 'Unknown')}")
        
        # Format created date
        created_str = config_data.get('created', 'Unknown')
        try:
            created_dt = QDateTime.fromString(created_str, Qt.ISODate)
            formatted_date = created_dt.toString("yyyy-MM-dd hh:mm:ss")
            self.created_label.setText(f"Created: {formatted_date}")
        except:
            self.created_label.setText(f"Created: {created_str}")
        
        # Extract just filename from paths
        model_path = config_data.get('model', 'None')
        model_name = Path(model_path).name if model_path != 'None' else 'None'
        self.model_label.setText(f"Model: {model_name}")
        
        calib_path = config_data.get('calibration', 'None')
        calib_name = Path(calib_path).name if calib_path != 'None' else 'None'
        self.calibration_label.setText(f"Calibration: {calib_name}")
        
        robot_type = config_data.get('robot_type', 'Default')
        self.robot_label.setText(f"Robot Type: {robot_type}")
        
    def clear_details(self):
        """Clear the details panel."""
        self.name_label.setText("Name: -")
        self.created_label.setText("Created: -")
        self.model_label.setText("Model: -")
        self.calibration_label.setText("Calibration: -")
        self.robot_label.setText("Robot Type: -")
        
    def accept(self):
        """Accept the dialog."""
        current_item = self.config_list.currentItem()
        if current_item:
            self.selected_config = current_item.data(Qt.UserRole)
        super().accept()


class QTConfiguration_Manager:
    """Manages workspace configurations (calibration + model + metadata)."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
    def save_configuration(self, name, calibration_path=None, model_path=None, robot_type=None):
        """
        Save current configuration.
        
        Args:
            name: Configuration name
            calibration_path: Path to calibration file
            model_path: Path to model file
            robot_type: Robot type identifier
            
        Returns:
            Tuple (success, message)
        """
        if not name:
            return False, "Configuration name cannot be empty"
            
        # Ensure .json extension
        if not name.endswith('.json'):
            name += '.json'
            
        config_path = self.config_dir / name
        
        # Check if already exists
        if config_path.exists():
            return False, f"Configuration '{name}' already exists"
            
        # Create configuration
        config = {
            "name": name.replace('.json', ''),
            "created": datetime.now().isoformat(),
            "calibration": str(calibration_path) if calibration_path else None,
            "model": str(model_path) if model_path else None,
            "robot_type": robot_type,
            "version": "1.0"
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True, f"Configuration saved as '{name}'"
        except Exception as e:
            return False, f"Failed to save configuration: {str(e)}"
            
    def load_configuration(self, config_path):
        """
        Load and validate configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dict or None if failed
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Validate that referenced files exist
            issues = []
            
            if config.get('calibration'):
                if not Path(config['calibration']).exists():
                    issues.append(f"Calibration file not found: {config['calibration']}")
                    
            if config.get('model'):
                if not Path(config['model']).exists():
                    issues.append(f"Model file not found: {config['model']}")
                    
            if issues:
                # Still return config but with warnings
                config['_warnings'] = issues
                
            return config
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {config_path}: {e}")
            return None
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return None
            
    def get_available_configurations(self):
        """
        List all available configurations.
        
        Returns:
            List of tuples (config_path, config_data)
        """
        configurations = []
        
        for config_file in self.config_dir.glob("*.json"):
            config_data = self.load_configuration(config_file)
            if config_data:
                configurations.append((config_file, config_data))
                
        # Sort by creation date (newest first)
        configurations.sort(key=lambda x: x[1].get('created', ''), reverse=True)
        
        return configurations
        
    def show_load_dialog(self, parent=None):
        """
        Show dialog to select configuration.
        
        Args:
            parent: Parent widget
            
        Returns:
            Selected configuration path or None
        """
        configurations = self.get_available_configurations()
        
        if not configurations:
            QMessageBox.information(parent, "No Configurations", 
                                  "No saved configurations found.")
            return None
            
        dialog = ConfigurationDialog(configurations, parent)
        if dialog.exec() == QDialog.Accepted:
            return dialog.selected_config
            
        return None
        
    def delete_configuration(self, config_path):
        """
        Delete a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Tuple (success, message)
        """
        try:
            os.remove(config_path)
            return True, "Configuration deleted"
        except Exception as e:
            return False, f"Failed to delete: {str(e)}"