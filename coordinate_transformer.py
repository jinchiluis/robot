import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any

class CoordinateTransformer:
    """
    Transforms pixel coordinates to robot coordinates using affine transformation.
    Handles non-linear mappings where robot coordinates may vary non-uniformly
    across the workspace due to mechanical constraints or calibration differences.
    """
    
    def __init__(self):
        self.pixel_points = []
        self.robot_points = []
        self.transformation_matrix = None
        self.is_calibrated_flag = False
        self.calibration_error = None
        
    def add_calibration_point(self, pixel_x: float, pixel_y: float, 
                            robot_x: float, robot_y: float) -> None:
        """
        Add a calibration point pair (pixel coordinates -> robot coordinates).
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels  
            robot_x: X coordinate in robot space (mm)
            robot_y: Y coordinate in robot space (mm)
        """
        self.pixel_points.append([pixel_x, pixel_y])
        self.robot_points.append([robot_x, robot_y])
        
        # Reset calibration when new points are added
        self.is_calibrated_flag = False
        self.transformation_matrix = None
        
    def clear_points(self) -> None:
        """Clear all calibration points and reset transformation."""
        self.pixel_points = []
        self.robot_points = []
        self.transformation_matrix = None
        self.is_calibrated_flag = False
        self.calibration_error = None
        
    def calculate_transformation(self) -> Tuple[bool, str]:
        """
        Calculate the affine transformation matrix from pixel to robot coordinates.
        Requires at least 3 points for affine transformation (6 parameters).
        Uses least squares fitting for overdetermined systems (>3 points).
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if len(self.pixel_points) < 3:
            return False, f"Need at least 3 calibration points, have {len(self.pixel_points)}"
            
        try:
            # Convert to numpy arrays
            pixel_array = np.array(self.pixel_points, dtype=np.float64)
            robot_array = np.array(self.robot_points, dtype=np.float64)
            
            # Create homogeneous coordinates for pixels [x, y, 1]
            num_points = len(self.pixel_points)
            A = np.ones((num_points, 3))
            A[:, 0] = pixel_array[:, 0]  # x coordinates
            A[:, 1] = pixel_array[:, 1]  # y coordinates
            
            # Solve for transformation parameters using least squares
            # We need to solve for both X and Y transformations separately
            # X_robot = a*x_pixel + b*y_pixel + c
            # Y_robot = d*x_pixel + e*y_pixel + f
            
            # Solve for X transformation parameters [a, b, c]
            x_params, x_residuals, x_rank, x_s = np.linalg.lstsq(A, robot_array[:, 0], rcond=None)
            
            # Solve for Y transformation parameters [d, e, f]  
            y_params, y_residuals, y_rank, y_s = np.linalg.lstsq(A, robot_array[:, 1], rcond=None)
            
            # Create the full transformation matrix
            # [a  b  c]
            # [d  e  f]
            # [0  0  1]
            self.transformation_matrix = np.array([
                [x_params[0], x_params[1], x_params[2]],
                [y_params[0], y_params[1], y_params[2]],
                [0, 0, 1]
            ])
            
            # Calculate calibration error (RMS error)
            total_error = 0
            for i in range(num_points):
                predicted = self.pixel_to_robot(self.pixel_points[i][0], self.pixel_points[i][1])
                if predicted is not None:
                    error_x = predicted[0] - self.robot_points[i][0]
                    error_y = predicted[1] - self.robot_points[i][1]
                    total_error += error_x**2 + error_y**2
                    
            self.calibration_error = np.sqrt(total_error / num_points)
            self.is_calibrated_flag = True
            
            return True, f"Calibration successful! RMS error: {self.calibration_error:.2f}mm"
            
        except np.linalg.LinAlgError as e:
            return False, f"Linear algebra error: {str(e)}"
        except Exception as e:
            return False, f"Calculation failed: {str(e)}"
    
    def pixel_to_robot(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to robot coordinates using the calibrated transformation.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple of (robot_x, robot_y) in mm, or None if not calibrated
        """
        if not self.is_calibrated_flag or self.transformation_matrix is None:
            return None
            
        try:
            # Create homogeneous coordinate [x, y, 1]
            pixel_homogeneous = np.array([pixel_x, pixel_y, 1])
            print("Image pixel:",pixel_x, pixel_y)
            
            # Apply transformation
            robot_homogeneous = self.transformation_matrix @ pixel_homogeneous
            print("Robot Coord:",robot_homogeneous[0], robot_homogeneous[1])
            
            return (robot_homogeneous[0], robot_homogeneous[1])
            
        except Exception:
            return None
    
    def robot_to_pixel(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert robot coordinates to pixel coordinates (inverse transformation).
        
        Args:
            robot_x: X coordinate in robot space (mm)
            robot_y: Y coordinate in robot space (mm)
            
        Returns:
            Tuple of (pixel_x, pixel_y), or None if not calibrated or non-invertible
        """
        if not self.is_calibrated_flag or self.transformation_matrix is None:
            return None
            
        try:
            # Calculate inverse transformation matrix
            inverse_matrix = np.linalg.inv(self.transformation_matrix)
            
            # Create homogeneous coordinate [x, y, 1]
            robot_homogeneous = np.array([robot_x, robot_y, 1])
            
            # Apply inverse transformation
            pixel_homogeneous = inverse_matrix @ robot_homogeneous
            
            return (pixel_homogeneous[0], pixel_homogeneous[1])
            
        except np.linalg.LinAlgError:
            return None
        except Exception:
            return None
    
    def is_calibrated(self) -> bool:
        """Check if the transformer has been calibrated."""
        return self.is_calibrated_flag
    
    def get_calibration_points(self) -> List[Dict[str, float]]:
        """
        Get all calibration points as a list of dictionaries.
        
        Returns:
            List of calibration point dictionaries with keys: pixel_x, pixel_y, robot_x, robot_y
        """
        points = []
        for i in range(len(self.pixel_points)):
            points.append({
                'pixel_x': self.pixel_points[i][0],
                'pixel_y': self.pixel_points[i][1],
                'robot_x': self.robot_points[i][0],
                'robot_y': self.robot_points[i][1]
            })
        return points
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get the current transformation matrix."""
        return self.transformation_matrix.copy() if self.transformation_matrix is not None else None
    
    def get_calibration_error(self) -> Optional[float]:
        """Get the RMS calibration error in mm."""
        return self.calibration_error
    
    def generate_robot_command(self, pixel_x: float, pixel_y: float, 
                             z: float = -10, speed: int = 10) -> Optional[Dict[str, Any]]:
        """
        Generate a robot command dictionary for the given pixel coordinates.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            z: Z coordinate for robot (default: -10)
            speed: Movement speed (default: 10)
            
        Returns:
            Dictionary with robot command parameters, or None if not calibrated
        """
        robot_coords = self.pixel_to_robot(pixel_x, pixel_y)
        if robot_coords is None:
            return None
            
        return {
            'command': 'move_to_position',
            'x': round(robot_coords[0], 2),
            'y': round(robot_coords[1], 2),
            'z': z,
            'speed': speed,
            'pixel_source': {'x': pixel_x, 'y': pixel_y}
        }
    
    def export_calibration(self) -> Dict[str, Any]:
        """
        Export calibration data to a dictionary for saving.
        
        Returns:
            Dictionary containing all calibration data
        """
        export_data = {
            'calibration_points': self.get_calibration_points(),
            'is_calibrated': self.is_calibrated_flag,
            'calibration_error': self.calibration_error,
            'transformation_matrix': self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
            'num_points': len(self.pixel_points)
        }
        return export_data
    
    def import_calibration(self, calibration_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Import calibration data from a dictionary.
        
        Args:
            calibration_data: Dictionary containing calibration data
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Clear existing data
            self.clear_points()
            
            # Import calibration points
            if 'calibration_points' in calibration_data:
                for point in calibration_data['calibration_points']:
                    self.add_calibration_point(
                        point['pixel_x'], point['pixel_y'],
                        point['robot_x'], point['robot_y']
                    )
            
            # Import transformation matrix if available
            if calibration_data.get('transformation_matrix') is not None:
                self.transformation_matrix = np.array(calibration_data['transformation_matrix'])
                self.is_calibrated_flag = calibration_data.get('is_calibrated', False)
                self.calibration_error = calibration_data.get('calibration_error')
                
                return True, f"Imported {len(self.pixel_points)} calibration points successfully"
            else:
                # Recalculate transformation if matrix not saved
                if len(self.pixel_points) >= 3:
                    return self.calculate_transformation()
                else:
                    return True, f"Imported {len(self.pixel_points)} calibration points (transformation will be calculated when enough points are added)"
                    
        except Exception as e:
            self.clear_points()
            return False, f"Import failed: {str(e)}"
    
    def validate_calibration(self) -> Tuple[bool, str, List[Dict[str, float]]]:
        """
        Validate the current calibration by checking prediction accuracy on calibration points.
        
        Returns:
            Tuple of (is_valid: bool, message: str, error_details: List[Dict])
        """
        if not self.is_calibrated_flag:
            return False, "Not calibrated", []
            
        error_details = []
        max_error = 0
        total_error = 0
        
        for i in range(len(self.pixel_points)):
            px, py = self.pixel_points[i]
            actual_rx, actual_ry = self.robot_points[i]
            
            predicted = self.pixel_to_robot(px, py)
            if predicted is None:
                return False, "Prediction failed during validation", []
                
            pred_rx, pred_ry = predicted
            error_x = pred_rx - actual_rx
            error_y = pred_ry - actual_ry
            error_magnitude = np.sqrt(error_x**2 + error_y**2)
            
            error_details.append({
                'point_index': i,
                'pixel_x': px,
                'pixel_y': py,
                'actual_robot_x': actual_rx,
                'actual_robot_y': actual_ry,
                'predicted_robot_x': pred_rx,
                'predicted_robot_y': pred_ry,
                'error_x': error_x,
                'error_y': error_y,
                'error_magnitude': error_magnitude
            })
            
            max_error = max(max_error, error_magnitude)
            total_error += error_magnitude
            
        avg_error = total_error / len(self.pixel_points)
        
        # Consider calibration valid if average error < 5mm and max error < 15mm
        is_valid = avg_error < 5.0 and max_error < 15.0
        
        message = f"Validation: Avg error: {avg_error:.2f}mm, Max error: {max_error:.2f}mm"
        if not is_valid:
            message += " (Consider recalibrating - errors are high)"
            
        return is_valid, message, error_details