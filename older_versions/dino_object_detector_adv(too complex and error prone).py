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
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractorHelper:
    """
    Advanced helper class using patch-level features and multi-scale processing.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load DINOv2 model
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_model.eval()
        
        # Basic normalization without resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.patch_size = 14  # ViT-S/14
        
    def find_green_boxes(self, image, debug_prefix=None):
        """Find green boxes in the image - same as original implementation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_1_original.jpg", image)
            cv2.imwrite(f"{debug_prefix}_2_green_mask.jpg", mask)
        
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 20 and h > 20:
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                
                filled_contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if filled_contours:
                    x, y, w, h = cv2.boundingRect(filled_contours[0])
                    
                    margin = 2
                    x_start = max(0, x + margin)
                    y_start = max(0, y + margin)
                    x_end = min(image.shape[1], x + w - margin)
                    y_end = min(image.shape[0], y + h - margin)
                    
                    region = image[y_start:y_end, x_start:x_end]
                    
                    if region.size > 0:
                        boxes.append({
                            'bbox': (x, y, x+w, y+h),
                            'region': region,
                            'center': (x + w//2, y + h//2)
                        })
        
        return boxes
    
    def get_adaptive_scales(self, box_size):
        """Get adaptive scales based on box size"""
        w, h = box_size
        base_size = np.sqrt(w * h)
        
        if base_size < 50:
            return [1.5, 2.0, 2.5]  # Small boxes need upscaling
        elif base_size < 150:
            return [0.75, 1.0, 1.5]  # Medium boxes
        else:
            return [0.5, 0.75, 1.0]  # Large boxes need downscaling
    
    def extract_patch_features(self, image, scale=1.0):
        """Extract patch-level features from an image at given scale"""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Resize if needed
        if scale != 1.0:
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            pil_image = pil_image.resize(new_size, Image.BILINEAR)
        
        # Ensure dimensions are divisible by patch_size
        w, h = pil_image.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        if new_w < self.patch_size or new_h < self.patch_size:
            # Too small, use minimum size
            new_w = max(new_w, self.patch_size * 4)  # At least 4x4 patches
            new_h = max(new_h, self.patch_size * 4)
        
        if (new_w, new_h) != (w, h):
            pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor
        tensor = transforms.ToTensor()(pil_image)
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # Get intermediate features
            features = self.dino_model.get_intermediate_layers(tensor, n=[11], return_class_token=True)
            patch_tokens = features[0][0]  # [1, num_patches, 384]
            cls_token = features[0][1]     # [1, 384]
            
            # Calculate patch grid size
            num_patches = patch_tokens.shape[1]
            grid_h = new_h // self.patch_size
            grid_w = new_w // self.patch_size
            
            # Verify dimensions match
            expected_patches = grid_h * grid_w
            if num_patches != expected_patches:
                print(f"Warning: patch count mismatch. Got {num_patches}, expected {expected_patches}")
                # Use square grid as fallback
                grid_size = int(np.sqrt(num_patches))
                grid_h = grid_w = grid_size
            
            # Reshape patch tokens to spatial grid
            patch_tokens = patch_tokens.reshape(1, grid_h, grid_w, -1)  # [1, H, W, 384]
            patch_tokens = patch_tokens.squeeze(0).cpu().numpy()  # [H, W, 384]
            
            cls_token = cls_token.squeeze(0).cpu().numpy()  # [384]
        
        return {
            'patch_features': patch_tokens,
            'global_feature': cls_token,
            'grid_h': grid_h,
            'grid_w': grid_w,
            'image_size': (new_w, new_h)
        }
    
    def extract_multi_scale_features(self, image, scales=None):
        """Extract features at multiple scales"""
        if scales is None:
            h, w = image.shape[:2]
            scales = self.get_adaptive_scales((w, h))
        
        all_features = []
        for scale in scales:
            features = self.extract_patch_features(image, scale)
            all_features.append(features)
        
        return all_features
    
    def aggregate_patch_features(self, patch_features, method='attention'):
        """Aggregate patch features using different methods"""
        # patch_features: [H, W, 384]
        h, w, d = patch_features.shape
        patches_flat = patch_features.reshape(-1, d)  # [H*W, 384]
        
        # Handle any NaN values
        patches_flat = np.nan_to_num(patches_flat, 0)
        
        if method == 'attention':
            # Use self-attention to weight patches
            # Simple version: use global feature similarity as attention
            global_feat = np.mean(patches_flat, axis=0)
            
            # Check if global_feat is valid
            if np.linalg.norm(global_feat) == 0:
                # Fallback to uniform weighting
                return patches_flat
            
            attention_weights = patches_flat @ global_feat
            
            # Prevent overflow in exp
            attention_weights = np.clip(attention_weights, -10, 10)
            attention_weights = np.exp(attention_weights)
            attention_sum = np.sum(attention_weights)
            
            if attention_sum > 0:
                attention_weights = attention_weights / attention_sum
            else:
                attention_weights = np.ones(len(attention_weights)) / len(attention_weights)
            
            weighted_features = patches_flat * attention_weights[:, np.newaxis]
            return weighted_features
        
        elif method == 'spatial_pool':
            # Divide into 2x2 spatial regions (or adapt based on shape)
            if h >= 2 and w >= 2:
                h_mid, w_mid = h // 2, w // 2
                regions = [
                    patch_features[:h_mid, :w_mid].reshape(-1, d),
                    patch_features[:h_mid, w_mid:].reshape(-1, d),
                    patch_features[h_mid:, :w_mid].reshape(-1, d),
                    patch_features[h_mid:, w_mid:].reshape(-1, d)
                ]
            else:
                # Too small for 2x2, just use all patches
                regions = [patches_flat]
            
            region_features = [np.mean(r, axis=0) if len(r) > 0 else np.zeros(d) for r in regions]
            return np.array(region_features)
        
        else:  # 'all'
            return patches_flat


class AdvancedOneShotObjectTrainer:
    """
    Advanced trainer using patch-level features and multiple prototypes.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.helper = AdvancedFeatureExtractorHelper(device)
        self.prototypes = {}
        self.config = {
            'similarity_threshold': 0.7,  # Slightly lower due to richer features
            'nms_distance_threshold': 50,
            'num_prototypes_per_class': 3,
            'patch_aggregation': 'attention',
            'use_multi_scale': True
        }
    
    def train(self, training_images_dict: Dict[str, List[str]], 
              objects_per_image: int = 1, 
              save_debug_images: bool = False):
        """Train with advanced features"""
        print("Starting advanced training process...")
        all_features = {obj_type: {'global': [], 'patch': []} for obj_type in training_images_dict.keys()}
        
        for obj_type, image_paths in training_images_dict.items():
            print(f"\nProcessing {obj_type} images...")
            
            for img_idx, img_path in enumerate(tqdm(image_paths, desc=f"Processing {obj_type}")):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Find green boxes
                boxes = self.helper.find_green_boxes(image)
                
                if len(boxes) == 0:
                    print(f"Warning: No green boxes found in {img_path}")
                    continue
                
                # Process each box
                for box_idx, box in enumerate(boxes[:objects_per_image]):
                    region = box['region']
                    
                    # Extract multi-scale features
                    if self.config['use_multi_scale']:
                        multi_features = self.helper.extract_multi_scale_features(region)
                        
                        # Aggregate across scales
                        global_features = []
                        patch_features_list = []
                        
                        for feat_dict in multi_features:
                            global_features.append(feat_dict['global_feature'])
                            
                            # Aggregate patches
                            patch_agg = self.helper.aggregate_patch_features(
                                feat_dict['patch_features'], 
                                method=self.config['patch_aggregation']
                            )
                            patch_features_list.append(patch_agg)
                        
                        # Average across scales
                        final_global = np.mean(global_features, axis=0)
                        # Ensure no NaN and normalize
                        final_global = np.nan_to_num(final_global, 0)
                        norm = np.linalg.norm(final_global)
                        if norm > 0:
                            final_global = final_global / norm
                        else:
                            final_global = np.ones(384) / np.sqrt(384)  # Fallback
                        
                        # For patches, concatenate from all scales
                        final_patches = np.concatenate(patch_features_list, axis=0)
                        final_patches = np.nan_to_num(final_patches, 0)
                        
                    else:
                        # Single scale
                        feat_dict = self.helper.extract_patch_features(region)
                        final_global = feat_dict['global_feature']
                        final_global = np.nan_to_num(final_global, 0)
                        norm = np.linalg.norm(final_global)
                        if norm > 0:
                            final_global = final_global / norm
                        else:
                            final_global = np.ones(384) / np.sqrt(384)  # Fallback
                        
                        final_patches = self.helper.aggregate_patch_features(
                            feat_dict['patch_features'],
                            method=self.config['patch_aggregation']
                        )
                        final_patches = np.nan_to_num(final_patches, 0)
                    
                    all_features[obj_type]['global'].append(final_global)
                    all_features[obj_type]['patch'].append(final_patches)
        
        # Create multiple prototypes using K-means
        print("\nCreating multiple prototypes per class...")
        for obj_type in all_features:
            if all_features[obj_type]['global']:
                # Global prototypes
                global_feats = np.array(all_features[obj_type]['global'])
                
                if len(global_feats) >= self.config['num_prototypes_per_class']:
                    # Use K-means to find diverse prototypes
                    kmeans = KMeans(n_clusters=self.config['num_prototypes_per_class'], 
                                   random_state=42, n_init=10)
                    kmeans.fit(global_feats)
                    global_prototypes = kmeans.cluster_centers_
                    
                    # Normalize
                    global_prototypes = global_prototypes / np.linalg.norm(global_prototypes, axis=1, keepdims=True)
                else:
                    # Too few samples, use all
                    global_prototypes = global_feats
                
                # Patch prototypes - store a subset of diverse patch features
                all_patches = []
                for patch_set in all_features[obj_type]['patch']:
                    # Ensure no NaN values
                    patch_set = np.nan_to_num(patch_set, 0)
                    all_patches.extend(patch_set)
                
                all_patches = np.array(all_patches)
                
                # Remove any zero vectors (from nan_to_num)
                non_zero_mask = np.any(all_patches != 0, axis=1)
                all_patches = all_patches[non_zero_mask]
                
                if len(all_patches) > 50:  # Limit number of patch prototypes
                    # Sample diverse patches
                    kmeans_patch = KMeans(n_clusters=50, random_state=42, n_init=3)
                    kmeans_patch.fit(all_patches)
                    patch_prototypes = kmeans_patch.cluster_centers_
                else:
                    patch_prototypes = all_patches
                
                # Normalize patch prototypes
                patch_norms = np.linalg.norm(patch_prototypes, axis=1, keepdims=True)
                patch_norms = np.where(patch_norms == 0, 1, patch_norms)  # Avoid division by zero
                patch_prototypes = patch_prototypes / patch_norms
                
                self.prototypes[obj_type] = {
                    'global': global_prototypes,
                    'patch': patch_prototypes,
                    'num_training_samples': len(global_feats)
                }
                
                print(f"{obj_type}: {len(global_prototypes)} global prototypes, {len(patch_prototypes)} patch prototypes")
            else:
                print(f"Warning: No features extracted for {obj_type}")
        
        return {
            'prototypes': self.prototypes,
            'config': self.config,
            'feature_counts': {k: v['num_training_samples'] for k, v in self.prototypes.items()}
        }
    
    def save_model(self, save_path: str):
        """Save trained model to disk"""
        model_data = {
            'prototypes': self.prototypes,
            'config': self.config,
            'model_info': {
                'feature_dim': 384,
                'dinov2_model': 'dinov2_vits14',
                'advanced_features': True
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Advanced model saved to {save_path}")


class AdvancedObjectDetector:
    """
    Advanced object detection using patch-level features and multiple prototypes.
    """
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.helper = AdvancedFeatureExtractorHelper(device)
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.prototypes = model_data['prototypes']
        self.config = model_data['config']
        self.object_types = list(self.prototypes.keys())
        
        print(f"Loaded advanced model with object types: {self.object_types}")
    
    def compute_similarity(self, features, obj_type):
        """Compute similarity using both global and patch features"""
        global_feat = features['global_feature']
        patch_feats = features['patch_features']
        
        # Global similarity - compare to all prototypes and take max
        global_prototypes = self.prototypes[obj_type]['global']
        global_sims = global_feat @ global_prototypes.T
        max_global_sim = np.max(global_sims)
        
        # Patch similarity - aggregate patch features first
        patch_agg = self.helper.aggregate_patch_features(
            patch_feats, 
            method=self.config['patch_aggregation']
        )
        
        # Compare aggregated patches to patch prototypes
        patch_prototypes = self.prototypes[obj_type]['patch']
        patch_sims = patch_agg @ patch_prototypes.T
        
        # Take top-K patch similarities
        top_k = min(5, len(patch_sims))
        top_patch_sims = np.sort(patch_sims.flatten())[-top_k:]
        avg_patch_sim = np.mean(top_patch_sims)
        
        # Combine global and patch similarities
        combined_sim = 0.6 * max_global_sim + 0.4 * avg_patch_sim
        
        return combined_sim, max_global_sim, avg_patch_sim
    
    def _apply_nms(self, detections, distance_threshold):
        """Apply non-maximum suppression based on center distance"""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for det in detections:
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
        """Detect objects using advanced features"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Find green boxes
        boxes = self.helper.find_green_boxes(image)
        
        if len(boxes) == 0:
            print("No green boxes found in the image.")
            return {obj_type: [] for obj_type in self.object_types}
        
        if debug:
            print(f"Found {len(boxes)} green boxes in the image")
        
        # Process each box
        detections = []
        for i, box in enumerate(boxes):
            region = box['region']
            
            # Extract features (multi-scale if configured)
            if self.config.get('use_multi_scale', True):
                multi_features = self.helper.extract_multi_scale_features(region)
                
                # Average features across scales for detection
                global_features = []
                for feat_dict in multi_features:
                    global_features.append(feat_dict['global_feature'])
                
                # Use middle scale for patch features
                middle_idx = len(multi_features) // 2
                features = multi_features[middle_idx]
                features['global_feature'] = np.mean(global_features, axis=0)
                features['global_feature'] = features['global_feature'] / np.linalg.norm(features['global_feature'])
            else:
                features = self.helper.extract_patch_features(region)
                features['global_feature'] = features['global_feature'] / np.linalg.norm(features['global_feature'])
            
            # Compare with all object types
            best_sim = -1
            best_type = None
            best_details = {}
            
            for obj_type in self.object_types:
                combined_sim, global_sim, patch_sim = self.compute_similarity(features, obj_type)
                
                if combined_sim > best_sim:
                    best_sim = combined_sim
                    best_type = obj_type
                    best_details = {
                        'global_sim': global_sim,
                        'patch_sim': patch_sim
                    }
            
            if best_sim > self.config['similarity_threshold']:
                detections.append({
                    'center': box['center'],
                    'bbox': box['bbox'],
                    'confidence': best_sim,
                    'object_type': best_type,
                    'details': best_details
                })
                
                if debug:
                    print(f"  Box {i}: {best_type} (conf: {best_sim:.3f}, "
                          f"global: {best_details['global_sim']:.3f}, "
                          f"patch: {best_details['patch_sim']:.3f})")
        
        # Group by object type and apply NMS
        results = {obj_type: [] for obj_type in self.object_types}
        
        for obj_type in self.object_types:
            obj_detections = [d for d in detections if d['object_type'] == obj_type]
            kept = self._apply_nms(obj_detections, self.config.get('nms_distance_threshold', 50))
            
            if max_objects_per_type and len(kept) > max_objects_per_type:
                kept = kept[:max_objects_per_type]
            
            results[obj_type] = [det['center'] for det in kept]
            
            if debug:
                print(f"{obj_type}: {len(kept)} objects detected")
        
        print(f"Advanced detection complete. Found {sum(len(v) for v in results.values())} objects.")
        
        if visualize:
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
                    
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"{obj_type}: {det['confidence']:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return results, vis_image
        
        return results


# Example usage
if __name__ == "__main__":
    # Training phase
    trainer = AdvancedOneShotObjectTrainer()
    
    # Prepare training images
    training_data = {
        'red_object': ['training/r1.png', 'training/r2.png', 'training/r3.png', 'training/r4.png', 'training/r5.png', 'training/r6.png', 'training/r7.png'],
        'blue_object': ['training/b1.png', 'training/b2.png', 'training/b3.png', 'training/b4.png', 'training/b5.png', 'training/b6.png', 'training/b7.png']
    }
    
    # Train model
    results = trainer.train(training_data, objects_per_image=1, save_debug_images=False)
    
    # Save model
    trainer.save_model('advanced_oneshot_model.pkl')
    
    # Inference phase
    detector = AdvancedObjectDetector('advanced_oneshot_model.pkl')
    
    # Detect objects
    detections = detector.detect('training/inference.png', debug=True)
    
    # Print results
    for obj_type, centers in detections.items():
        print(f"\n{obj_type}:")
        for i, (x, y) in enumerate(centers):
            print(f"  Object {i+1}: Center at ({x}, {y})")