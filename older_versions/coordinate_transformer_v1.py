import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class CoordinateTransformer:
    """
    Handles transformation between pixel coordinates and robot workspace coordinates.
    """
    
    def __init__(self):
        self.calibration_points = []
        self.transform_matrix = None
        self.is_perspective = False
        
    def add_calibration_point(self, pixel_x: float, pixel_y: float, robot_x: float, robot_y: float) -> None:
        """Add a calibration point pair."""
        point = {
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'robot_x': robot_x,
            'robot_y': robot_y
        }
        self.calibration_points.append(point)
    
    def clear_points(self) -> None:
        """Clear all calibration points and transformation matrix."""
        self.calibration_points = []
        self.transform_matrix = None
        self.is_perspective = False
    
    def calculate_transformation(self) -> Tuple[bool, str]:
        """
        Calculate the transformation matrix from calibration points.
        Returns (success, message)
        """
        if len(self.calibration_points) < 4:
            return False, "Need at least 4 calibration points"
        
        try:
            # Prepare point arrays
            pixel_points = np.array([
                [p['pixel_x'], p['pixel_y']] for p in self.calibration_points
            ], dtype=np.float32)
            
            robot_points = np.array([
                [p['robot_x'], p['robot_y']] for p in self.calibration_points
            ], dtype=np.float32)
            
            # Use homography for all cases with 4+ points
            # This finds the best-fit perspective transformation using all points
            self.transform_matrix, mask = cv2.findHomography(pixel_points, robot_points, cv2.RANSAC, 5.0)
            self.is_perspective = True
            
            # Count inliers from RANSAC
            inliers = np.sum(mask) if mask is not None else len(self.calibration_points)
            
            return True, f"Homography transformation calculated successfully using {len(self.calibration_points)} points ({inliers} inliers)"
                
        except Exception as e:
            return False, f"Failed to calculate transformation: {str(e)}"
    
    def is_calibrated(self) -> bool:
        """Check if transformation matrix is available."""
        return self.transform_matrix is not None
    
    def pixel_to_robot(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to robot coordinates.
        Returns (robot_x, robot_y) or None if not calibrated.
        """
        if not self.is_calibrated():
            return None
        
        try:
            if self.is_perspective:
                # Perspective transformation
                pixel_coord = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
                robot_coord = cv2.perspectiveTransform(pixel_coord, self.transform_matrix)
                return float(robot_coord[0][0][0]), float(robot_coord[0][0][1])
            else:
                # Affine transformation
                pixel_coord = np.array([[pixel_x, pixel_y, 1]], dtype=np.float32).T
                robot_coord = self.transform_matrix @ pixel_coord
                return float(robot_coord[0][0]), float(robot_coord[1][0])
                
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return None
    
    def robot_to_pixel(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert robot coordinates to pixel coordinates (inverse transformation).
        Returns (pixel_x, pixel_y) or None if not calibrated.
        """
        if not self.is_calibrated():
            return None
        
        try:
            if self.is_perspective:
                # Inverse perspective transformation
                robot_coord = np.array([[[robot_x, robot_y]]], dtype=np.float32)
                inverse_matrix = cv2.invert(self.transform_matrix)[1]
                pixel_coord = cv2.perspectiveTransform(robot_coord, inverse_matrix)
                return float(pixel_coord[0][0][0]), float(pixel_coord[0][0][1])
            else:
                # Inverse affine transformation
                robot_coord = np.array([[robot_x, robot_y, 1]], dtype=np.float32).T
                inverse_matrix = cv2.invertAffineTransform(self.transform_matrix)
                # Convert to 3x3 for matrix multiplication
                full_inverse = np.vstack([inverse_matrix, [0, 0, 1]])
                pixel_coord = full_inverse @ robot_coord
                return float(pixel_coord[0][0]), float(pixel_coord[1][0])
                
        except Exception as e:
            print(f"Error in inverse coordinate transformation: {e}")
            return None
    
    def generate_robot_command(self, pixel_x: float, pixel_y: float, 
                             z: float = -10.0, t: float = 3.14, spd: float = 10) -> Optional[Dict[str, Any]]:
        """
        Generate robot movement command from pixel coordinates.
        Returns robot command dictionary or None if conversion fails.
        """
        robot_coords = self.pixel_to_robot(pixel_x, pixel_y)
        if robot_coords is None:
            return None
        
        return {
            "T": 104,  # CMD_XYZT_GOAL_CTRL
            "x": robot_coords[0],
            "y": robot_coords[1],
            "z": z,
            "t": t,
            "spd": spd
        }
    
    def get_calibration_accuracy(self) -> Optional[float]:
        """
        Calculate the average error of the calibration by testing all calibration points.
        Returns average error in robot coordinate units or None if not calibrated.
        """
        if not self.is_calibrated() or len(self.calibration_points) < 4:
            return None
        
        total_error = 0.0
        for point in self.calibration_points:
            predicted = self.pixel_to_robot(point['pixel_x'], point['pixel_y'])
            if predicted is None:
                return None
            
            actual = (point['robot_x'], point['robot_y'])
            error = np.sqrt((predicted[0] - actual[0])**2 + (predicted[1] - actual[1])**2)
            total_error += error
        
        return total_error / len(self.calibration_points)
    
    def export_calibration(self) -> Dict[str, Any]:
        """Export calibration data for saving."""
        return {
            'calibration_points': self.calibration_points,
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'is_perspective': self.is_perspective
        }
    
    def import_calibration(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Import calibration data from saved file.
        Returns (success, message)
        """
        try:
            self.calibration_points = data.get('calibration_points', [])
            
            matrix_data = data.get('transform_matrix')
            if matrix_data is not None:
                self.transform_matrix = np.array(matrix_data, dtype=np.float32)
                self.is_perspective = data.get('is_perspective', False)
            else:
                self.transform_matrix = None
                self.is_perspective = False
            
            return True, f"Calibration imported successfully with {len(self.calibration_points)} points"
            
        except Exception as e:
            return False, f"Failed to import calibration: {str(e)}"
    
    def get_workspace_bounds(self) -> Optional[Dict[str, float]]:
        """
        Get the bounds of the robot workspace based on calibration points.
        Returns dict with min/max x/y values or None if not enough points.
        """
        if len(self.calibration_points) < 2:
            return None
        
        robot_coords = [(p['robot_x'], p['robot_y']) for p in self.calibration_points]
        x_coords = [coord[0] for coord in robot_coords]
        y_coords = [coord[1] for coord in robot_coords]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }