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

class OneShotObjectTrainer:
    """
    Trains a one-shot object detection model using DINOv2 features.
    Optimized for fast training and inference.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', resize_factor=0.3):
        self.device = device
        self.resize_factor = resize_factor
        print(f"Using device: {self.device}")
        print(f"Image resize factor: {self.resize_factor}")
        
        # Load smaller, faster DINOv2 model
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_model.eval()
        
        # Multi-scale transforms for richer features
        # All sizes must be divisible by 14 for DINOv2's patch size
        self.transforms = {
            'normal': transforms.Compose([
                transforms.Resize((224, 224)),  # 224 = 14 * 16
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'small': transforms.Compose([
                transforms.Resize((196, 196)),  # 196 = 14 * 14
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'large': transforms.Compose([
                transforms.Resize((252, 252)),  # 252 = 14 * 18
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        
        self.transform = self.transforms['normal']  # Keep for compatibility
        
        self.prototypes = {}
        self.adaptive_thresholds = {}  # Per-class thresholds
        self.background_prototype = None  # For hard negative mining
        self.prototype_weights = {}  # For attention-weighted prototypes
        
        self.config = {
            'window_sizes': [(96, 96)],  # Will be updated dynamically
            'stride_ratio': 0.75,
            'similarity_threshold': 0.75,  # Default, will be adaptive
            'nms_threshold': 0.8,
            'batch_size': 32,
            'resize_factor': resize_factor,
            'use_multi_scale': True,
            'use_spatial_pyramid': True,
            'use_hard_negatives': True,
            'feature_augmentation': True,
            'augmentation_noise': 0.05,
        }
    
    def _extract_features_batch(self, images, multi_scale=False):
        """Extract DINOv2 features from a batch of images efficiently"""
        # Convert numpy arrays to PIL images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            else:
                pil_images.append(img)
        
        if multi_scale and self.config.get('use_multi_scale', True):
            # Extract features at multiple scales
            all_features = []
            for scale_name, transform in self.transforms.items():
                tensors = torch.stack([transform(img) for img in pil_images]).to(self.device)
                with torch.no_grad():
                    features = self.dino_model(tensors).cpu().numpy()
                all_features.append(features)
            
            # Concatenate multi-scale features
            features = np.concatenate(all_features, axis=1)
        else:
            # Single scale extraction (original behavior)
            tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
            with torch.no_grad():
                features = self.dino_model(tensors).cpu().numpy()
        
        # L2 normalize each feature vector
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Spatial pyramid pooling if enabled
        if self.config.get('use_spatial_pyramid', True) and not multi_scale:
            pyramid_features = self._spatial_pyramid_pooling(pil_images)
            normalized = np.concatenate([normalized, pyramid_features], axis=1)
            normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True)
        
        return normalized
    
    def _spatial_pyramid_pooling(self, images):
        """Extract features from spatial sub-regions"""
        all_pyramid_features = []
        
        for img in images:
            width, height = img.size
            pyramid_features = []
            
            # 2x2 grid
            for i in range(2):
                for j in range(2):
                    x1 = i * width // 2
                    y1 = j * height // 2
                    x2 = (i + 1) * width // 2
                    y2 = (j + 1) * height // 2
                    
                    sub_img = img.crop((x1, y1, x2, y2))
                    tensor = self.transform(sub_img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        sub_features = self.dino_model(tensor).cpu().numpy()[0]
                    
                    pyramid_features.append(sub_features)
            
            # Average pool the sub-region features
            pyramid_features = np.mean(pyramid_features, axis=0)
            all_pyramid_features.append(pyramid_features)
        
        return np.array(all_pyramid_features)
    
    def _augment_features(self, features):
        """Apply feature augmentation with small noise"""
        if self.config.get('feature_augmentation', True):
            noise = np.random.normal(0, self.config.get('augmentation_noise', 0.05), features.shape)
            augmented = features + noise
            # Re-normalize
            augmented = augmented / np.linalg.norm(augmented, axis=1, keepdims=True)
            return np.vstack([features, augmented])
        return features
    
    def _resize_image(self, image):
        """Resize image based on resize_factor"""
        if self.resize_factor < 1.0:
            h, w = image.shape[:2]
            new_h = int(h * self.resize_factor)
            new_w = int(w * self.resize_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image
    
    def _sliding_window_batch(self, image, window_size, stride):
        """Generate all windows from an image at once"""
        h, w = image.shape[:2]
        window_h, window_w = window_size
        
        windows = []
        positions = []
        
        for y in range(0, h - window_h + 1, stride):
            for x in range(0, w - window_w + 1, stride):
                windows.append(image[y:y+window_h, x:x+window_w])
                positions.append((x, y))
        
        return windows, positions
    
    def _find_distinct_regions(self, image, n_regions=1):
        """Find the most distinct regions in an image using batch processing"""
        h, w = image.shape[:2]
        
        # Collect all windows
        all_windows = []
        all_positions = []
        window_sizes = []
        
        for window_size in self.config['window_sizes']:
            window_h, window_w = window_size
            stride = int(window_w * self.config['stride_ratio'])
            windows, positions = self._sliding_window_batch(image, window_size, stride)
            all_windows.extend(windows)
            all_positions.extend(positions)
            window_sizes.extend([window_size] * len(windows))
        
        if not all_windows:
            return []
        
        # Extract features for all windows in batches
        features = []
        for i in range(0, len(all_windows), self.config['batch_size']):
            batch = all_windows[i:i+self.config['batch_size']]
            batch_features = self._extract_features_batch(batch, multi_scale=True)
            features.extend(batch_features)
        
        features = np.array(features)
        
        # Find most distinct regions
        selected_indices = []
        
        for _ in range(n_regions):
            if len(selected_indices) == 0:
                # First region: find most unique (lowest avg similarity to all others)
                avg_similarities = np.mean(features @ features.T, axis=1)
                best_idx = np.argmin(avg_similarities)
            else:
                # Next regions: find most different from already selected
                selected_features = features[selected_indices]
                similarities_to_selected = features @ selected_features.T
                max_similarities = np.max(similarities_to_selected, axis=1)
                
                # Mask out regions too close to already selected
                for idx in selected_indices:
                    x1, y1 = all_positions[idx]
                    for i, (x2, y2) in enumerate(all_positions):
                        if np.sqrt((x2-x1)**2 + (y2-y1)**2) < 50:
                            max_similarities[i] = 1.0
                
                best_idx = np.argmin(max_similarities)
            
            selected_indices.append(best_idx)
        
        # Return selected regions
        results = []
        for idx in selected_indices:
            x, y = all_positions[idx]
            window_h, window_w = window_sizes[idx]
            results.append({
                'bbox': (x, y, x+window_w, y+window_h),
                'features': features[idx],
                'center': (x + window_w//2, y + window_h//2),
                'window_size': (window_h, window_w)
            })
        
        return results
    
    def _collect_background_regions(self, all_features, all_regions):
        """Collect hard negative samples (background regions)"""
        if not self.config.get('use_hard_negatives', True):
            return None
        
        # Flatten all object features
        all_object_features = []
        for obj_type in all_features:
            all_object_features.extend(all_features[obj_type])
        
        if not all_object_features:
            return None
        
        all_object_features = np.array(all_object_features)
        
        # Find regions that are far from all object features
        background_features = []
        
        for regions in all_regions:
            for region in regions:
                feature = region['features']
                # Calculate max similarity to any object
                similarities = feature @ all_object_features.T
                max_similarity = np.max(similarities) if similarities.size > 0 else 0
                
                # If this region is dissimilar to all objects, it's background
                if max_similarity < 0.5:  # Threshold for background
                    background_features.append(feature)
        
        if background_features:
            background_prototype = np.mean(background_features, axis=0)
            return background_prototype / np.linalg.norm(background_prototype)
        
        return None
    
    def _compute_attention_weights(self, features):
        """Compute attention weights for prototype creation"""
        if len(features) < 2:
            return np.ones(len(features))
        
        features = np.array(features)
        # Compute pairwise similarities
        similarities = features @ features.T
        
        # Average similarity of each feature to all others
        avg_similarities = (np.sum(similarities, axis=1) - 1) / (len(features) - 1)
        
        # Convert to weights (higher weight for more typical examples)
        weights = np.exp(avg_similarities * 2)  # Temperature=2
        weights = weights / np.sum(weights)
        
        return weights
    
    def _update_window_sizes(self, all_regions):
        """Dynamically update window sizes based on detected objects"""
        all_sizes = []
        for regions in all_regions:
            for region in regions:
                h, w = region['window_size']
                all_sizes.append((h, w))
        
        if all_sizes:
            # Find the most common sizes
            unique_sizes, counts = np.unique(all_sizes, axis=0, return_counts=True)
            # Keep top 3 most common sizes
            top_indices = np.argsort(counts)[-3:]
            self.config['window_sizes'] = [tuple(size) for size in unique_sizes[top_indices]]
            print(f"Updated window sizes to: {self.config['window_sizes']}")
    
    def train(self, training_images_dict: Dict[str, List[str]], 
              objects_per_image: int = 1, 
              save_debug_images: bool = False):
        """
        Train the model on provided images using batch processing.
        Much faster than the previous implementation.
        """
        print("Starting training process...")
        all_features = {obj_type: [] for obj_type in training_images_dict.keys()}
        all_regions = []
        
        # Process each object type
        for obj_type, image_paths in training_images_dict.items():
            print(f"\nProcessing {obj_type} images...")
            
            for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {obj_type}")):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Resize for faster processing
                original_size = image.shape[:2]
                image = self._resize_image(image)
                
                # Find distinct regions (objects)
                regions = self._find_distinct_regions(image, objects_per_image)
                all_regions.append(regions)
                
                if not regions:
                    print(f"Warning: No objects found in {img_path}")
                    continue
                
                # Add features (with augmentation)
                for region in regions:
                    base_features = region['features']
                    all_features[obj_type].append(base_features)
                    
                    # Add augmented versions
                    if self.config.get('feature_augmentation', True):
                        for _ in range(2):  # Add 2 augmented versions
                            noise = np.random.normal(0, self.config.get('augmentation_noise', 0.05), base_features.shape)
                            aug_features = base_features + noise
                            aug_features = aug_features / np.linalg.norm(aug_features)
                            all_features[obj_type].append(aug_features)
                
                # Save debug image
                if save_debug_images:
                    debug_img = image.copy()
                    for region in regions:
                        x1, y1, x2, y2 = region['bbox']
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_img, obj_type, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imwrite(f"debug_{obj_type}_{img_idx}.jpg", debug_img)
        
        # Update window sizes based on detected objects
        self._update_window_sizes(all_regions)
        
        # Collect background features
        self.background_prototype = self._collect_background_regions(all_features, all_regions)
        if self.background_prototype is not None:
            print("Collected background prototype from hard negatives")
        
        # Create prototypes with attention weighting
        print("\nCreating prototypes...")
        for obj_type in all_features:
            if all_features[obj_type]:
                features = np.array(all_features[obj_type])
                
                # Compute attention weights
                weights = self._compute_attention_weights(features)
                self.prototype_weights[obj_type] = weights
                
                # Weighted average
                prototype = np.average(features, axis=0, weights=weights)
                self.prototypes[obj_type] = prototype / np.linalg.norm(prototype)
                print(f"{obj_type}: {len(all_features[obj_type])} feature vectors")
            else:
                print(f"Warning: No features extracted for {obj_type}")
        
        # Compute adaptive thresholds
        print("\nComputing adaptive thresholds...")
        obj_types = list(self.prototypes.keys())
        
        for obj_type in obj_types:
            # Find similarity to other classes
            similarities_to_others = []
            for other_type in obj_types:
                if other_type != obj_type:
                    sim = np.dot(self.prototypes[obj_type], self.prototypes[other_type])
                    similarities_to_others.append(sim)
            
            # Set threshold based on inter-class similarity
            if similarities_to_others:
                max_similarity = max(similarities_to_others)
                # Higher threshold for classes that are similar to others
                self.adaptive_thresholds[obj_type] = max(0.75, min(0.95, max_similarity + 0.15))
            else:
                self.adaptive_thresholds[obj_type] = self.config['similarity_threshold']
            
            print(f"  {obj_type}: threshold = {self.adaptive_thresholds[obj_type]:.3f}")
        
        # Check prototype similarities
        print("\nPrototype similarities:")
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
            'feature_counts': {k: len(v) for k, v in all_features.items()},
            'adaptive_thresholds': self.adaptive_thresholds,
            'background_prototype': self.background_prototype,
            'prototype_weights': self.prototype_weights
        }
    
    def save_model(self, save_path: str):
        """Save trained model to disk"""
        model_data = {
            'prototypes': self.prototypes,
            'config': self.config,
            'adaptive_thresholds': self.adaptive_thresholds,
            'background_prototype': self.background_prototype,
            'prototype_weights': self.prototype_weights,
            'model_info': {
                'feature_dim': 384 * (3 if self.config.get('use_multi_scale', True) else 1),
                'dinov2_model': 'dinov2_vits14',
                'resize_factor': self.config['resize_factor']
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")
        print(f"Resize factor saved: {self.config['resize_factor']}")


class ObjectDetector:
    """
    Fast object detection using trained one-shot model.
    Uses batch processing for efficient inference.
    """
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_model.eval()
        
        # Multi-scale transforms
        # All sizes must be divisible by 14 for DINOv2's patch size
        self.transforms = {
            'normal': transforms.Compose([
                transforms.Resize((224, 224)),  # 224 = 14 * 16
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'small': transforms.Compose([
                transforms.Resize((196, 196)),  # 196 = 14 * 14
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'large': transforms.Compose([
                transforms.Resize((252, 252)),  # 252 = 14 * 18
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        
        self.transform = self.transforms['normal']  # Keep for compatibility
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.prototypes = model_data['prototypes']
        self.config = model_data['config']
        self.adaptive_thresholds = model_data.get('adaptive_thresholds', {})
        self.background_prototype = model_data.get('background_prototype', None)
        
        # Ensure resize_factor is loaded
        if 'resize_factor' not in self.config:
            if 'model_info' in model_data and 'resize_factor' in model_data['model_info']:
                self.config['resize_factor'] = model_data['model_info']['resize_factor']
            else:
                print("WARNING: No resize_factor found in saved model, defaulting to 1.0")
                self.config['resize_factor'] = 1.0
        
        # Use default thresholds if adaptive ones not found
        if not self.adaptive_thresholds:
            for obj_type in self.prototypes:
                self.adaptive_thresholds[obj_type] = self.config['similarity_threshold']
        
        self.object_types = list(self.prototypes.keys())
        print(f"Loaded model with object types: {self.object_types}")
        print(f"Using resize factor: {self.config['resize_factor']}")
        print(f"Adaptive thresholds: {self.adaptive_thresholds}")
    
    def _extract_features_batch(self, images, multi_scale=False):
        """Extract features from multiple images efficiently"""
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            else:
                pil_images.append(img)
        
        if multi_scale and self.config.get('use_multi_scale', True):
            # Extract features at multiple scales
            all_features = []
            for scale_name, transform in self.transforms.items():
                tensors = torch.stack([transform(img) for img in pil_images]).to(self.device)
                with torch.no_grad():
                    features = self.dino_model(tensors).cpu().numpy()
                all_features.append(features)
            
            # Concatenate multi-scale features
            features = np.concatenate(all_features, axis=1)
        else:
            # Single scale extraction
            tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
            with torch.no_grad():
                features = self.dino_model(tensors).cpu().numpy()
        
        # L2 normalize
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Spatial pyramid pooling if enabled (only for single scale)
        if self.config.get('use_spatial_pyramid', True) and not multi_scale:
            pyramid_features = self._spatial_pyramid_pooling(pil_images)
            normalized = np.concatenate([normalized, pyramid_features], axis=1)
            normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True)
        
        return normalized
    
    def _spatial_pyramid_pooling(self, images):
        """Extract features from spatial sub-regions"""
        all_pyramid_features = []
        
        for img in images:
            width, height = img.size
            pyramid_features = []
            
            # 2x2 grid
            for i in range(2):
                for j in range(2):
                    x1 = i * width // 2
                    y1 = j * height // 2
                    x2 = (i + 1) * width // 2
                    y2 = (j + 1) * height // 2
                    
                    sub_img = img.crop((x1, y1, x2, y2))
                    tensor = self.transform(sub_img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        sub_features = self.dino_model(tensor).cpu().numpy()[0]
                    
                    pyramid_features.append(sub_features)
            
            # Average pool the sub-region features
            pyramid_features = np.mean(pyramid_features, axis=0)
            all_pyramid_features.append(pyramid_features)
        
        return np.array(all_pyramid_features)
    
    def _calibrate_confidence(self, confidence, obj_type):
        """Apply temperature scaling for confidence calibration"""
        temperature = 1.5  # Can be tuned
        calibrated = np.exp(confidence / temperature) / (np.exp(confidence / temperature) + np.exp((1 - confidence) / temperature))
        return calibrated
    
    def detect(self, image_path: str, visualize: bool = False, debug: bool = False, 
              max_objects_per_type: int = None) -> Dict[str, List[Tuple[int, int]]]:
        """
        Fast object detection using batch processing.
        
        Args:
            image_path: Path to image
            visualize: Whether to return visualization
            debug: Print debug information
            max_objects_per_type: Maximum objects to return per type (None = no limit)
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize for faster processing
        resize_factor = self.config.get('resize_factor', 1.0)
        if resize_factor < 1.0:
            image = cv2.resize(original_image, None, fx=resize_factor, fy=resize_factor, 
                             interpolation=cv2.INTER_AREA)
            print(f"Resized from {original_image.shape[:2]} to {image.shape[:2]}")
        else:
            image = original_image
        
        # Collect all windows
        all_windows = []
        all_positions = []
        window_info = []
        
        for window_size in self.config['window_sizes']:
            window_h, window_w = window_size
            stride = int(window_w * self.config['stride_ratio'])
            
            h, w = image.shape[:2]
            for y in range(0, h - window_h + 1, stride):
                for x in range(0, w - window_w + 1, stride):
                    all_windows.append(image[y:y+window_h, x:x+window_w])
                    all_positions.append((x, y))
                    window_info.append((window_h, window_w))
        
        if not all_windows:
            return {obj_type: [] for obj_type in self.object_types}
        
        # Extract features for all windows in batches
        print(f"Processing {len(all_windows)} windows...")
        all_features = []
        use_multi_scale = self.config.get('use_multi_scale', True)
        
        for i in range(0, len(all_windows), self.config['batch_size']):
            batch = all_windows[i:i+self.config['batch_size']]
            batch_features = self._extract_features_batch(batch, multi_scale=use_multi_scale)
            all_features.extend(batch_features)
        
        all_features = np.array(all_features)
        
        # Compare all features with prototypes at once
        detections = []
        prototype_matrix = np.array([self.prototypes[obj_type] for obj_type in self.object_types])
        similarities = all_features @ prototype_matrix.T  # Shape: (n_windows, n_object_types)
        
        # Check background similarity if available
        if self.background_prototype is not None:
            bg_similarities = all_features @ self.background_prototype
            is_background = bg_similarities > 0.85  # High similarity to background
        else:
            is_background = np.zeros(len(all_features), dtype=bool)
        
        # Find best matches
        best_matches = np.argmax(similarities, axis=1)
        best_similarities = np.max(similarities, axis=1)
        
        # Filter by adaptive threshold and background check
        for i, (similarity, obj_idx) in enumerate(zip(best_similarities, best_matches)):
            if is_background[i]:
                continue  # Skip background regions
                
            obj_type = self.object_types[obj_idx]
            threshold = self.adaptive_thresholds.get(obj_type, self.config['similarity_threshold'])
            
            if similarity > threshold:
                x, y = all_positions[i]
                window_h, window_w = window_info[i]
                
                # Calibrate confidence
                calibrated_confidence = self._calibrate_confidence(similarity, obj_type)
                
                # Scale coordinates back
                if resize_factor < 1.0:
                    center = (
                        int((x + window_w//2) / resize_factor),
                        int((y + window_h//2) / resize_factor)
                    )
                    bbox = (
                        int(x / resize_factor),
                        int(y / resize_factor),
                        int((x + window_w) / resize_factor),
                        int((y + window_h) / resize_factor)
                    )
                else:
                    center = (x + window_w//2, y + window_h//2)
                    bbox = (x, y, x + window_w, y + window_h)
                
                detections.append({
                    'bbox': bbox,
                    'center': center,
                    'confidence': calibrated_confidence,
                    'raw_confidence': similarity,
                    'object_type': obj_type
                })
        
        if debug:
            print(f"\nBefore NMS: {len(detections)} detections")
            for obj_type in self.object_types:
                obj_dets = [d for d in detections if d['object_type'] == obj_type]
                print(f"  {obj_type}: {len(obj_dets)} detections")
                if obj_dets:
                    confs = [d['raw_confidence'] for d in obj_dets]
                    print(f"    Raw confidences: min={min(confs):.3f}, max={max(confs):.3f}")
                    cal_confs = [d['confidence'] for d in obj_dets]
                    print(f"    Calibrated: min={min(cal_confs):.3f}, max={max(cal_confs):.3f}")
        
        # Apply NMS
        results = {obj_type: [] for obj_type in self.object_types}
        all_kept = []
        
        for obj_type in self.object_types:
            obj_detections = [d for d in detections if d['object_type'] == obj_type]
            kept_detections = self._non_max_suppression(obj_detections)
            
            # Limit number of objects if specified
            if max_objects_per_type and len(kept_detections) > max_objects_per_type:
                # Keep only the highest confidence detections
                kept_detections = kept_detections[:max_objects_per_type]
            
            results[obj_type] = [det['center'] for det in kept_detections]
            all_kept.extend(kept_detections)
            
            if debug:
                print(f"  {obj_type} after NMS: {len(kept_detections)} detections")
        
        print(f"Detection complete. Found {sum(len(v) for v in results.values())} objects.")
        
        if visualize:
            return results, self._visualize(original_image, all_kept)
        
        return results
    
    def _non_max_suppression(self, detections, threshold=None):
        """Apply NMS to remove overlapping detections - using center distance instead of IoU"""
        if not detections:
            return []
        
        threshold = threshold or self.config['nms_threshold']
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for det1 in detections:
            should_keep = True
            center1 = det1['center']
            
            for det2 in keep:
                center2 = det2['center']
                
                # Calculate distance between centers
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # If centers are too close, it's likely the same object
                # Use the larger dimension of the detection as reference
                x1, y1, x2, y2 = det1['bbox']
                box_size = max(x2 - x1, y2 - y1)
                
                if distance < box_size * threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det1)
        
        return keep
    
    def _visualize(self, image, detections):
        """Create visualization of detections"""
        vis_image = image.copy()
        colors = {'green_object': (0, 255, 0), 'yellow_object': (0, 255, 255)}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = det['center']
            obj_type = det['object_type']
            color = colors.get(obj_type, (255, 0, 0))
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_image, center, 5, color, -1)
            cv2.putText(vis_image, f"{obj_type}: {det['confidence']:.2f}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image


# Example usage
if __name__ == "__main__":
    # Training phase
    trainer = OneShotObjectTrainer()
    
    # Prepare training images
    training_data = {
        'red_object': ['training/r1.png', 'training/r2.png', 'training/r3.png', 'training/r4.png', 'training/r5.png', 'training/r6.png', 'training/r7.png'],  # 10 images
        'blue_object': ['training/b1.png', 'training/b2.png', 'training/b3.png', 'training/b4.png', 'training/b5.png', 'training/b6.png', 'training/b7.png']   # 10 images
    }
    
    # Train model
    results = trainer.train(training_data, objects_per_image=1, save_debug_images=True)
    
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