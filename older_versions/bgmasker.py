#!/usr/bin/env python3
"""
Background masking utilities for removing/cropping backgrounds in images
"""

import cv2
import numpy as np
from PIL import Image


class BackgroundMasker:
    """Various methods to remove/ignore background in images"""
    
    @staticmethod
    def center_crop_percent(image, crop_percent=0.8):
        """Simple center crop - removes X% from each edge
        
        Args:
            image: PIL Image
            crop_percent: Keep this percentage of the image (0.8 = crop 20% from edges)
        Returns:
            PIL Image: Cropped image
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        h, w = image_np.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        cropped = image_np[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return Image.fromarray(cropped)
    
    @staticmethod
    def edge_based_crop(image, edge_threshold=50, margin=20):
        """Crop to bounding box of detected edges - good for objects on uniform background
        
        Args:
            image: PIL Image
            edge_threshold: Canny edge detection threshold
            margin: Pixels to add around detected object
        Returns:
            PIL Image: Cropped image
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return Image.fromarray(image_np)
        
        # Get bounding box of all contours
        x_min, y_min = image_np.shape[1], image_np.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image_np.shape[1], x_max + margin)
        y_max = min(image_np.shape[0], y_max + margin)
        
        cropped = image_np[y_min:y_max, x_min:x_max]
        
        return Image.fromarray(cropped)