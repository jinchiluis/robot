import cv2
import numpy as np
from ultralytics import YOLO
import torch

class DotDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize the dot detector with YOLO model.
        
        Args:
            model_path (str): Path to custom YOLO model. If None, uses YOLOv8n
            confidence_threshold (float): Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model - you'll need to train this for red/blue dots specifically
        # or use a pre-trained model if available
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Using YOLOv8 nano as base - you'll need to train this for dots
            self.model = YOLO('yolov8n.pt')
            print("Warning: Using base YOLOv8 model. Train a custom model for red/blue dots.")
    
    def detect_dots(self, image_path):
        """
        Detect red and blue dots in an image using YOLO.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary with 'red_dots' and 'blue_dots' lists containing coordinates
        """
        # Run YOLO inference
        results = self.model(image_path, conf=self.confidence_threshold)
        
        red_dots = []
        blue_dots = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center coordinates
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Assuming class_id 0 = red dot, class_id 1 = blue dot
                    # You'll need to adjust this based on your model's class mapping
                    if class_id == 0:  # Red dot
                        red_dots.append({
                            'center': (center_x, center_y),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence)
                        })
                    elif class_id == 1:  # Blue dot
                        blue_dots.append({
                            'center': (center_x, center_y),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence)
                        })
        
        return {
            'red_dots': red_dots,
            'blue_dots': blue_dots
        }
    
    def detect_dots_with_fallback(self, image_path):
        """
        Detect dots using YOLO first, then fallback to color-based detection.
        This is useful if you don't have a trained YOLO model yet.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary with 'red_dots' and 'blue_dots' lists containing coordinates
        """
        # Try YOLO first
        yolo_results = self.detect_dots(image_path)
        
        # If no dots found with YOLO, use color-based detection as fallback
        if not yolo_results['red_dots'] and not yolo_results['blue_dots']:
            print("No dots detected with YOLO, using color-based fallback...")
            return self._detect_dots_by_color(image_path)
        
        return yolo_results
    
    def _detect_dots_by_color(self, image_path):
        """
        Fallback method using color-based detection for red and blue dots.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary with 'red_dots' and 'blue_dots' lists containing coordinates
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and blue in HSV
        # Red color range (accounting for hue wrap-around)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Blue color range
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        red_dots = self._find_circular_contours(red_mask)
        blue_dots = self._find_circular_contours(blue_mask)
        
        return {
            'red_dots': red_dots,
            'blue_dots': blue_dots
        }
    
    def _find_circular_contours(self, mask):
        """
        Find circular contours in a binary mask and return their centers.
        
        Args:
            mask (np.array): Binary mask
            
        Returns:
            list: List of dictionaries with dot information
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dots = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter out small noise
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Only consider circular shapes
                    if circularity > 0.5:
                        # Get bounding box and center
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        dots.append({
                            'center': (center_x, center_y),
                            'bbox': (x, y, x + w, y + h),
                            'confidence': circularity,
                            'area': area
                        })
        
        return dots
    
    def visualize_detections(self, image_path, output_path=None):
        """
        Visualize detected dots on the image.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image. If None, displays image
        """
        # Get detections
        detections = self.detect_dots_with_fallback(image_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Draw red dots
        for dot in detections['red_dots']:
            center = dot['center']
            cv2.circle(image, center, 10, (0, 0, 255), 2)  # Red circle
            cv2.putText(image, f"R({center[0]},{center[1]})", 
                       (center[0] + 15, center[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw blue dots
        for dot in detections['blue_dots']:
            center = dot['center']
            cv2.circle(image, center, 10, (255, 0, 0), 2)  # Blue circle
            cv2.putText(image, f"B({center[0]},{center[1]})", 
                       (center[0] + 15, center[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to {output_path}")
        else:
            cv2.imshow('Dot Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = DotDetector(confidence_threshold=0.5)
    
    # Detect dots in an image
    image_path = "your_image.jpg"
    results = detector.detect_dots_with_fallback(image_path)
    
    # Print results
    print(f"Found {len(results['red_dots'])} red dots:")
    for i, dot in enumerate(results['red_dots']):
        print(f"  Red dot {i+1}: Center {dot['center']}, Confidence: {dot['confidence']:.2f}")
    
    print(f"\nFound {len(results['blue_dots'])} blue dots:")
    for i, dot in enumerate(results['blue_dots']):
        print(f"  Blue dot {i+1}: Center {dot['center']}, Confidence: {dot['confidence']:.2f}")
    
    # Visualize results
    detector.visualize_detections(image_path, "output_with_detections.jpg")