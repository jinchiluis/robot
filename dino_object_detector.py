import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# You'll need to install: pip install torch torchvision opencv-python pillow tqdm


class FeatureExtractorHelper:
    """
    Helper class to ensure consistent feature extraction between training and inference.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load DINOv2 model
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def find_green_boxes(self, image, debug_prefix=None):
        """Find green boxes in the image - used for both training and inference"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color - using original values that worked
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Save debug images if prefix provided
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_1_original.jpg", image)
            cv2.imwrite(f"{debug_prefix}_2_green_mask.jpg", mask)
        
        # For thin lines, use minimal morphological operations
        kernel = np.ones((2, 2), np.uint8)  # Much smaller kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_3_mask_morphed.jpg", mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_prefix:
            print(f"DEBUG {debug_prefix}: Found {len(contours)} contours")
            contour_img = image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(f"{debug_prefix}_4_all_contours.jpg", contour_img)
        
        boxes = []
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            if debug_prefix:
                print(f"  Contour {i}: bbox=({x},{y},{w},{h})")
            
            # Filter out very small boxes (noise)
            if w > 20 and h > 20:
                # For thin outlines, we need to find the interior
                # Create a mask for this specific contour
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                
                # Find the bounding box of the filled contour
                filled_contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if filled_contours:
                    x, y, w, h = cv2.boundingRect(filled_contours[0])
                    
                    # Extract the region inside the box with minimal margin
                    margin = 2  # Smaller margin for thin boxes
                    x_start = max(0, x + margin)
                    y_start = max(0, y + margin)
                    x_end = min(image.shape[1], x + w - margin)
                    y_end = min(image.shape[0], y + h - margin)
                    
                    region = image[y_start:y_end, x_start:x_end]
                    
                    if region.size > 0:  # Ensure region is valid
                        boxes.append({
                            'bbox': (x, y, x+w, y+h),
                            'region': region,
                            'center': (x + w//2, y + h//2)
                        })
                        
                        if debug_prefix:
                            cv2.imwrite(f"{debug_prefix}_5_region_{i}.jpg", region)
                            print(f"    -> Added as box, region shape: {region.shape}")
            else:
                if debug_prefix:
                    print(f"    -> Skipped (too small)")
        
        if debug_prefix:
            print(f"DEBUG {debug_prefix}: Returning {len(boxes)} boxes")
            # Draw final detected boxes
            final_img = image.copy()
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box['bbox']
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(final_img, f"Box {j}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"{debug_prefix}_6_final_boxes.jpg", final_img)
        
        return boxes
    
    def extract_features_batch(self, images):
        """Extract DINOv2 features from a batch of images"""
        if not images:
            return np.array([])
        
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            else:
                pil_images.append(img)
        
        tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        
        with torch.no_grad():
            features = self.dino_model(tensors).cpu().numpy()
        
        # L2 normalize
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        return normalized
    
    def extract_features_from_green_boxes(self, image, debug_prefix=None):
        """Extract features from all green boxes in an image"""
        boxes = self.find_green_boxes(image, debug_prefix=debug_prefix)
        
        if not boxes:
            return np.array([]), []
        
        regions = [box['region'] for box in boxes]
        features = self.extract_features_batch(regions)
        
        return features, boxes


class OneShotObjectTrainer:
    """
    Trains a one-shot object detection model using DINOv2 features.
    Uses green boxes to identify objects in training images.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.helper = FeatureExtractorHelper(device)
        self.prototypes = {}
        self.config = {
            'similarity_threshold': 0.75,
            'nms_distance_threshold': 50  # Minimum pixel distance between detections
        }
    
    def train(self, training_images_dict: Dict[str, List[str]], 
              objects_per_image: int = 1, 
              save_debug_images: bool = False):
        """
        Train the model on provided images.
        Images should have green boxes around objects.
        """
        print("Starting training process...")
        all_features = {obj_type: [] for obj_type in training_images_dict.keys()}
        
        for obj_type, image_paths in training_images_dict.items():
            print(f"\nProcessing {obj_type} images...")
            
            for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {obj_type}")):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Extract features from green boxes
                debug_prefix = f"debug_gb_{obj_type}_{img_idx}" if save_debug_images else None
                features, boxes = self.helper.extract_features_from_green_boxes(image, debug_prefix=debug_prefix)
                
                if len(features) == 0:
                    print(f"Warning: No green boxes found in {img_path}")
                    continue
                
                # Use up to objects_per_image features
                features_to_use = features[:objects_per_image]
                all_features[obj_type].extend(features_to_use)
                
                if save_debug_images:
                    debug_img = image.copy()
                    for i, box in enumerate(boxes[:objects_per_image]):
                        x1, y1, x2, y2 = box['bbox']
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(debug_img, f"{obj_type} #{i+1}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite(f"debug_train_{obj_type}_{img_idx}.jpg", debug_img)
        
        # Create prototypes
        print("\nCreating prototypes...")
        for obj_type in all_features:
            if all_features[obj_type]:
                prototype = np.mean(all_features[obj_type], axis=0)
                self.prototypes[obj_type] = prototype / np.linalg.norm(prototype)
                print(f"{obj_type}: {len(all_features[obj_type])} feature vectors")
            else:
                print(f"Warning: No features extracted for {obj_type}")
        
        # Check prototype similarities
        print("\nPrototype similarities:")
        obj_types = list(self.prototypes.keys())
        for i, type1 in enumerate(obj_types):
            for j, type2 in enumerate(obj_types):
                if i < j:
                    sim = np.dot(self.prototypes[type1], self.prototypes[type2])
                    print(f"  {type1} <-> {type2}: {sim:.3f}")
                    if sim > 0.9:
                        print(f"  WARNING: Very high similarity between {type1} and {type2}!")
        
        return {
            'prototypes': self.prototypes,
            'config': self.config,
            'feature_counts': {k: len(v) for k, v in all_features.items()}
        }
    
    def save_model(self, save_path: str):
        """Save trained model to disk"""
        model_data = {
            'prototypes': self.prototypes,
            'config': self.config,
            'model_info': {
                'feature_dim': 384,
                'dinov2_model': 'dinov2_vits14'
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")


class ObjectDetector:
    """
    Object detection using trained one-shot model.
    Uses green boxes to identify candidate objects.
    """
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.helper = FeatureExtractorHelper(device)
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.prototypes = model_data['prototypes']
        self.config = model_data['config']
        self.object_types = list(self.prototypes.keys())
        
        print(f"Loaded model with object types: {self.object_types}")
    
    def _apply_nms(self, detections, distance_threshold):
        """Apply non-maximum suppression based on center distance"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for det in detections:
            # Check if too close to already kept detections
            too_close = False
            for kept_det in keep:
                dist = np.sqrt((det['center'][0] - kept_det['center'][0])**2 + 
                             (det['center'][1] - kept_det['center'][1])**2)
                if dist < distance_threshold:
                    too_close = True
                    break
            
            if not too_close:
                keep.append(det)
        
        return keep
    
    def detect(self, image_path: str, visualize: bool = False, debug: bool = False, 
              max_objects_per_type: int = None) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect objects in image using green boxes.
        
        Returns:
            Dictionary mapping object types to lists of (x, y) center coordinates
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Extract features from green boxes
        debug_prefix = f"debug_detect_{Path(image_path).stem}" if debug else None
        features, boxes = self.helper.extract_features_from_green_boxes(image, debug_prefix=debug_prefix)
        
        if len(features) == 0:
            print("No green boxes found in the image.")
            return {obj_type: [] for obj_type in self.object_types}
        
        if debug:
            print(f"Found {len(boxes)} green boxes in the image")
        
        # Compare each feature with prototypes
        features = np.array(features)
        prototype_matrix = np.array([self.prototypes[obj_type] for obj_type in self.object_types])
        similarities = features @ prototype_matrix.T
        
        # Find best matches
        detections = []
        for i, (box, sims) in enumerate(zip(boxes, similarities)):
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            
            if best_sim > self.config['similarity_threshold']:
                detections.append({
                    'center': box['center'],
                    'bbox': box['bbox'],
                    'confidence': best_sim,
                    'object_type': self.object_types[best_idx]
                })
                
                if debug:
                    print(f"  Box {i}: {self.object_types[best_idx]} (conf: {best_sim:.3f})")
        
        # Group by object type and apply NMS
        results = {obj_type: [] for obj_type in self.object_types}
        
        for obj_type in self.object_types:
            obj_detections = [d for d in detections if d['object_type'] == obj_type]
            kept = self._apply_nms(obj_detections, self.config.get('nms_distance_threshold', 50))
            
            # Limit number of objects if specified
            if max_objects_per_type and len(kept) > max_objects_per_type:
                kept = kept[:max_objects_per_type]
            
            results[obj_type] = [det['center'] for det in kept]
            
            if debug:
                print(f"{obj_type}: {len(kept)} objects detected")
        
        print(f"Detection complete. Found {sum(len(v) for v in results.values())} objects.")
        
        if visualize:
            # Visualization showing detected objects
            vis_image = image.copy()
            
            # Draw all green boxes first (in gray)
            for box in boxes:
                x1, y1, x2, y2 = box['bbox']
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            
            # Draw detected objects
            for obj_type in self.object_types:
                obj_detections = [d for d in detections if d['object_type'] == obj_type]
                kept = self._apply_nms(obj_detections, self.config.get('nms_distance_threshold', 50))
                
                for det in kept[:max_objects_per_type] if max_objects_per_type else kept:
                    center = det['center']
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Draw thicker rectangle for detected objects
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"{obj_type}: {det['confidence']:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return results, vis_image
        
        return results


# Example usage
if __name__ == "__main__":
    # Training phase
    trainer = OneShotObjectTrainer()
    
    # Prepare training images - can be any object types now
    training_data = {
        'red_object': ['training/r1.png', 'training/r2.png', 'training/r3.png', 'training/r4.png', 'training/r5.png', 'training/r6.png', 'training/r7.png'],
        'blue_object': ['training/b1.png', 'training/b2.png', 'training/b3.png', 'training/b4.png', 'training/b5.png', 'training/b6.png', 'training/b7.png']
    }
    
    # Train model
    results = trainer.train(training_data, objects_per_image=1, save_debug_images=False)
    
    # Save model
    trainer.save_model('oneshot_model.pkl')
    
    # Inference phase
    detector = ObjectDetector('oneshot_model.pkl')
    
    # Detect objects
    detections = detector.detect('training/inference.png', debug=True)
    
    # Print results
    for obj_type, centers in detections.items():
        print(f"\n{obj_type}:")
        for i, (x, y) in enumerate(centers):
            print(f"  Object {i+1}: Center at ({x}, {y})")