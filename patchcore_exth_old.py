#!/usr/bin/env python3
"""
Advanced PatchCore Implementation with Performance "Secrets" + Elegant Heatmap Integration
These are the tricks that make PatchCore achieve 99.6% AUROC!
Newest Version: GPU Pytorch for coreset sampling instead of FAISS + Seamless Heatmap
  -> 25x faster + Beautiful anomaly visualization
"""

from xml.parsers.expat import model
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import time
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available - always using scipy for nearest neighbor search")
from pathlib import Path
from tqdm import tqdm
from bgmasker import BackgroundMasker


class SimplePatchCore:
    """Fixed PatchCore with consistent feature extraction"""
    
    def __init__(self, backbone='wide_resnet50_2', device='cpu', mask_method=None, mask_params=None):
        self.device = device

        "This version is optimized for INFERENCE: CPU FAISS and scipy"
        "                              TRAINING: GPU PyTorch if available, else CPU Torch"
        self.device = 'cpu'
        
        if backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.feature_layers = ['layer2', 'layer3']
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_layers = ['layer2', 'layer3']
        
        # FIX 1: Replace BatchNorm with GroupNorm to avoid batch size effects
        self._replace_batchnorm_with_groupnorm()
        
        # Setup feature extractor
        self.feature_extractor = self._setup_feature_extractor()
        self.model.to(device)
        self.model.eval()

        #masker
        self.mask_method = mask_method
        self.mask_params = mask_params
        self.masker = BackgroundMasker() if mask_method else None
        
        # FIX 2: Disable gradient computation and set to eval mode permanently
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Storage
        self.memory_bank = None
        self.global_threshold = None
        self.projection = None
        self.faiss_index = None
        
        # FIX 3: Store normalization parameters from training
        self.feature_mean = None
        self.feature_std = None
        
        # Store feature map size for heatmap generation
        self.feature_map_size = None
        
        # Image preprocessing
        self.image_size = 512
        self.transform_train = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _replace_batchnorm_with_groupnorm(self):
        """Replace BatchNorm layers with GroupNorm to avoid batch size dependency"""
        # Actually, don't replace anything - just freeze the BatchNorm layers
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                # Freeze BatchNorm parameters
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                # Set momentum to 0 to prevent running stats updates
                module.momentum = 0
    
    def _replace_batchnorm_recursive(self, module):
        """Not used anymore - kept for compatibility"""
        pass
    
    def _setup_feature_extractor(self):
        """Setup multi-layer feature extraction"""
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for multiple layers
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(hook_fn(name))
        
        return features
    
    def extract_features(self, images, return_spatial=False):
        """Extract multi-scale features"""
        features = []
        spatial_features = []
        
        with torch.no_grad():
            _ = self.model(images)
            
            reference_size = None
            
            for i, layer_name in enumerate(self.feature_layers):
                layer_features = self.feature_extractor[layer_name]
                b, c, h, w = layer_features.shape
                
                if reference_size is None:
                    reference_size = (h, w)
                    self.feature_map_size = reference_size
                
                if (h, w) != reference_size:
                    layer_features = torch.nn.functional.interpolate(
                        layer_features, 
                        size=reference_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                if return_spatial:
                    spatial_features.append(layer_features)
                
                layer_features = layer_features.permute(0, 2, 3, 1).reshape(b, reference_size[0]*reference_size[1], c)
                features.append(layer_features)
        
        features = torch.cat(features, dim=-1)
        
        if return_spatial:
            spatial_features = torch.cat(spatial_features, dim=1)
            return features, spatial_features
        
        return features
    
    def adaptive_sampling(self, features, sampling_ratio=0.01):
        """K-center greedy sampling"""
        n_samples = int(len(features) * sampling_ratio)
        if n_samples >= len(features):
            return np.arange(len(features))
        
        print(f"🚀 PyTorch GPU coreset sampling: {len(features)} -> {n_samples}")
        
        if torch.cuda.is_available():
            features_torch = torch.from_numpy(features).cuda()
            device = 'cuda'
        else:
            features_torch = torch.from_numpy(features)
            device = 'cpu'
        
        n_features = len(features)
        selected_indices = [np.random.randint(n_features)]
        
        min_distances = torch.full((n_features,), float('inf'), device=device)
        
        for i in tqdm(range(n_samples - 1), desc="PyTorch coreset sampling"):
            new_center_idx = selected_indices[-1]
            new_center = features_torch[new_center_idx:new_center_idx+1]
            
            distances = torch.cdist(features_torch, new_center).squeeze()
            min_distances = torch.minimum(min_distances, distances)
            
            temp_distances = min_distances.clone()
            temp_distances[selected_indices] = -1
            next_idx = torch.argmax(temp_distances).item()
            selected_indices.append(next_idx)
        
        return np.array(selected_indices)
    
    def setup_faiss_index(self, features):
        """Setup FAISS index"""
        if not FAISS_AVAILABLE:
            print("Using scipy instead of FAISS (slower but works)")
            return None
            
        dimension = features.shape[1]
        
        if self.device == 'cuda' and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
        
        index.add(features.astype(np.float32))
        
        return index
    
    def fit(self, train_dir, sample_ratio=0.01, threshold_percentile=99, val_dir=None):
        """Train with all the secrets
        
        Args:
            train_dir: Directory containing normal training images
            sample_ratio: Percentage of patches to keep in memory bank... 1%: ~95% AUROC//10%: ~99% AUROC//25%: ~99.5% AUROC//50%: ~99.6% AUROC
            threshold_percentile: Percentile for threshold calculation
            val_dir: Optional separate validation directory. If None, uses train_dir=>DANGEROUS!
        """
        print(f"Training Advanced PatchCore on: {train_dir}")
        
        # Create dataset
        dataset = SimpleImageDataset(train_dir, transform=self.transform_train, 
                                   mask_method=self.mask_method, mask_params=self.mask_params)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        
        all_features = []
        
        # Extract features
        start_time2 = time.time()
        print(f"Extracting multi-layer features from {len(dataloader)} batches...")
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            features = self.extract_features(images)
            all_features.append(features.cpu().numpy())
        total_time2 = time.time() - start_time2
        print(f"Extraction complete in {total_time2:.2f} seconds!")    
        
        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        print(f"Total features extracted: {all_features.shape}")
        
        # Dimensionality reduction (optional but helps)
        if all_features.shape[1] > 512:
            print("Applying dimensionality reduction...")
            self.projection = GaussianRandomProjection(n_components=512, random_state=42)
            all_features = self.projection.fit_transform(all_features)
        
        # Smart sampling
        print("Performing coreset sampling...")
        if sample_ratio < 0.51:  # Only use smart sampling for small ratios
            selected_indices = self.adaptive_sampling(all_features, sample_ratio)
        else:
            # Random sampling for larger ratios (faster)
            #n_samples = int(len(all_features) * sample_ratio)
            #selected_indices = np.random.choice(len(all_features), n_samples, replace=False)
            #Just take all!
            selected_indices = np.arange(len(all_features)) 
        
        self.memory_bank = all_features[selected_indices]
        print(f"Memory bank size: {self.memory_bank.shape}")
        
        # Setup FAISS index
        print("Setting up FAISS index for fast search...")
        self.faiss_index = self.setup_faiss_index(self.memory_bank)
        
        # Calculate threshold with validation split
        # Use val_dir if provided, otherwise use train_dir
        validation_dir = val_dir if val_dir is not None else train_dir
        if val_dir is not None:
            print(f"Using separate validation directory: {val_dir}")
        else:
            print("Warning: Using training directory for validation (not recommended)")
            
        self.calculate_threshold_validation(validation_dir, percentile=threshold_percentile)
        
        print("Training complete!")
    
    def calculate_threshold_validation(self, val_dir, percentile=99):
        """SECRET #7: Use validation split for threshold
        
        Args:
            val_dir: Directory containing validation images
            percentile: Percentile of scores to use as threshold
        """
        print(f"Calculating threshold from validation directory: {val_dir}")
        
        dataset = SimpleImageDataset(val_dir, transform=self.transform_test,
                                   mask_method=self.mask_method, mask_params=self.mask_params)
        
        # Use all validation images
        all_scores = []
        
        for idx in tqdm(range(len(dataset)), desc="Validation"):
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.extract_features(img_batch)
            features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
            
            # Project if needed
            if self.projection is not None:
                features_np = self.projection.transform(features_np)
            
            # Calculate anomaly score
            score = self.calculate_anomaly_score(features_np)
            all_scores.append(score)
        
        # Set threshold
        self.global_threshold = np.percentile(all_scores, percentile)
        
        print(f"\nValidation threshold calculated:")
        print(f"  - Based on {len(all_scores)} validation images")
        print(f"  - {percentile}th percentile: {self.global_threshold:.4f}")
        print(f"  - Score distribution: min={min(all_scores):.6f}, max={max(all_scores):.6f}, mean={np.mean(all_scores):.6f}")
        
        return self.global_threshold
    
    def calculate_anomaly_score(self, features, return_patch_scores=False):
        """Calculate anomaly score using FAISS or scipy
        
        Args:
            features: Feature array (n_patches, n_features)
            return_patch_scores: If True, returns per-patch scores for heatmap
        """
        if self.faiss_index is not None and FAISS_AVAILABLE:
            # Use FAISS for fast search
            start_time = time.perf_counter()
            distances, _ = self.faiss_index.search(features.astype(np.float32), k=1)
            #min_distances = distances.squeeze()
            min_distances = np.sqrt(distances.squeeze())
            execution_time = time.perf_counter() - start_time
            print(f"FAISS took {execution_time:.4f} seconds")
        else:
            # Fallback to scipy
            distances = cdist(features, self.memory_bank, metric='euclidean')
            min_distances = np.min(distances, axis=1)
        
        # Ensure min_distances is always 1D
        if len(min_distances.shape) == 0:
            min_distances = np.array([min_distances])
        
        if return_patch_scores:
            return min_distances
        
        # Use max for anomaly score
        anomaly_score = np.max(min_distances)
        
        return anomaly_score
    
    def generate_heatmap(self, image_path, alpha=0.5, colormap='jet', save_path=None):
        """Fast heatmap - main fix: remove spatial features extraction"""
        import time
        start_time = time.perf_counter()

        original_image = Image.open(image_path).convert('RGB')
        original_np = np.array(original_image)
        original_height, original_width = original_np.shape[:2]
        
        image_tensor = self.transform_test(original_image).unsqueeze(0).to(self.device)
        
        # FIX: Single feature extraction (not spatial)
        features = self.extract_features(image_tensor)  # Remove return_spatial=True
        features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
        
        if self.projection is not None:
            features_np = self.projection.transform(features_np)
        
        # Use existing method - should be fast with FAISS-GPU
        patch_scores = self.calculate_anomaly_score(features_np, return_patch_scores=True)
        
        h, w = self.feature_map_size
        score_map = patch_scores.reshape(h, w)
        
        # Faster image processing
        score_map_resized = cv2.resize(score_map.astype(np.float32), 
                                       (original_width, original_height), 
                                       interpolation=cv2.INTER_LINEAR)  # Faster than CUBIC
        
        score_map_smooth = gaussian_filter(score_map_resized, sigma=2.0)  # Faster than 4.0
        
        # Normalize and colorize
        score_min, score_max = score_map_smooth.min(), score_map_smooth.max()
        if score_max > score_min:
            score_map_norm = (score_map_smooth - score_min) / (score_max - score_min)
        else:
            score_map_norm = np.zeros_like(score_map_smooth)
        
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(score_map_norm)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Simple overlay
        original_float = original_np.astype(np.float32)
        heatmap_float = heatmap_colored.astype(np.float32)
        
        overlay = (original_float * (1.0 - alpha) + heatmap_float * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        if save_path:
            Image.fromarray(overlay).save(save_path)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Heatmap took {execution_time:.4f} seconds")    
        
        return overlay

    def predict(self, image_path, return_heatmap=True, min_region_size=None):
       """Predict with all optimizations + optional heatmap
       
       Args:
           image_path: Path to input image
           return_heatmap: If True, includes heatmap in return dict
           min_region_size: If not None, filters out anomaly regions smaller than this size in IMAGE pixels
                            (e.g., 25 for 25 pixels in image space, following FR PatchCore)
                            224x224 -> 25px, 512x512 -> 131px
       """
       # Load and preprocess
       image = Image.open(image_path).convert('RGB')
       original_size = image.size  # (width, height)
       
       # Apply background masking if configured
       if self.masker and self.mask_method:
           if self.mask_method == 'center_crop':
               image = self.masker.center_crop_percent(image, **self.mask_params)
           elif self.mask_method == 'edge_crop':
               image = self.masker.edge_based_crop(image, **self.mask_params)
       
       image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)
       
       # Extract features
       start_time2 = time.time()
       features = self.extract_features(image_tensor)
       total_time2 = time.time() - start_time2
       print(f"Extraction complete in {total_time2:.2f} seconds!")
       features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
       
       # Project if needed
       if self.projection is not None:
           features_np = self.projection.transform(features_np)
       
       # Calculate score
       anomaly_score = self.calculate_anomaly_score(features_np)
       is_anomaly = anomaly_score > self.global_threshold
       
       # Apply region-based filtering if requested
       if min_region_size is not None and is_anomaly:
           # Get patch-level scores
           patch_scores = self.calculate_anomaly_score(features_np, return_patch_scores=True)
           
           # Reshape to spatial dimensions
           h, w = self.feature_map_size
           score_map = patch_scores.reshape(h, w)
           
           # Upsample score map to image resolution
           # Use the transformed image size (512x512 typically)
           image_height, image_width = self.image_size, self.image_size 
           score_map_image = cv2.resize(score_map.astype(np.float32), 
                                       (image_width, image_height), 
                                       interpolation=cv2.INTER_LINEAR)
           
           # Create binary mask at image resolution
           binary_mask_image = (score_map_image > self.global_threshold).astype(np.uint8)
           
           # Find connected components at image resolution
           num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_image, connectivity=8)
           
           # Check if any region is larger than min_region_size (in image pixels)
           large_regions_exist = False
           for i in range(1, num_labels):  # Skip background (label 0)
               area = stats[i, cv2.CC_STAT_AREA]
               if area >= min_region_size:
                   large_regions_exist = True
                   x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                   print(f"Region {i}: area={area} px, bbox=({x},{y},{w},{h})")
                   #break #<- uncomment to stop after first large region and therefore speed up processing
           
           # Update anomaly decision based on region size
           if not large_regions_exist:
               is_anomaly = False
       
       result = {
           'anomaly_score': float(anomaly_score),
           'is_anomaly': bool(is_anomaly),
           'threshold': float(self.global_threshold)
       }
       
       # Add region filtering info if it was applied
       if min_region_size is not None:
           result['min_region_size'] = min_region_size
           result['region_filtered'] = not is_anomaly and anomaly_score > self.global_threshold
       
       # Add heatmap if requested
       if return_heatmap:
           try:
               heatmap = self.generate_heatmap(image_path)
               result['heatmap'] = heatmap
           except Exception as e:
               print(f"Warning: Could not generate heatmap: {e}")
               result['heatmap'] = None
       
       return result
    
    def predict_with_min_patch(self, image_path, return_heatmap=True, min_region_size=None):
        """Predict with all optimizations + optional heatmap
        
        Args:
            image_path: Path to input image
            return_heatmap: If True, includes heatmap in return dict
            min_region_size: If not None, filters out anomaly regions smaller than this size (in "pixels" in feature space (patch))
                             4 means that 4 patches must be connected to be considered an anomaly
                             224x224: 4 works well, but might be too aggressive
                             FR Patchcore uses 5x5 = 25px in image size for 224x224 images which is smaller than 1 patch!!!!
                             If i upscale to 512x512, then min_region_size=2 or 3 would simulate FR Patchcore's behavior, but will it work??
        """
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        
        # Apply background masking if configured
        if self.masker and self.mask_method:
            if self.mask_method == 'center_crop':
                image = self.masker.center_crop_percent(image, **self.mask_params)
            elif self.mask_method == 'edge_crop':
                image = self.masker.edge_based_crop(image, **self.mask_params)
        
        image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)
        
        # Extract features
        features = self.extract_features(image_tensor)
        features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
        
        # Project if needed
        if self.projection is not None:
            features_np = self.projection.transform(features_np)
        
        # Calculate score
        anomaly_score = self.calculate_anomaly_score(features_np)
        
        is_anomaly = anomaly_score > self.global_threshold
        
        # Apply region-based filtering if requested
        if min_region_size is not None and is_anomaly:
            # Get patch-level scores
            patch_scores = self.calculate_anomaly_score(features_np, return_patch_scores=True)
            
            # Reshape to spatial dimensions
            h, w = self.feature_map_size
            score_map = patch_scores.reshape(h, w)
            
            # Create binary mask using the same threshold
            binary_mask = (score_map > self.global_threshold).astype(np.uint8)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            # Check if any region is larger than min_region_size
            large_regions_exist = False
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_region_size:
                    large_regions_exist = True
                    break
            
            # Update anomaly decision based on region size
            if not large_regions_exist:
                is_anomaly = False
                # Keep the original score but mark as normal due to small region size
        
        result = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'threshold': float(self.global_threshold)
        }
        
        # Add region filtering info if it was applied
        if min_region_size is not None:
            result['min_region_size'] = min_region_size
            result['region_filtered'] = not is_anomaly and anomaly_score > self.global_threshold
        
        # Add heatmap if requested
        if return_heatmap:
            try:
                heatmap = self.generate_heatmap(image_path)
                result['heatmap'] = heatmap
            except Exception as e:
                print(f"Warning: Could not generate heatmap: {e}")
                result['heatmap'] = None
        
        return result
    
    def save(self, path):
        """Save model with all components"""
        torch.save({
            'memory_bank': self.memory_bank,
            'model_state': self.model.state_dict(),
            'global_threshold': self.global_threshold,
            'projection': self.projection,
            'feature_layers': self.feature_layers,
            'feature_map_size': self.feature_map_size,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'mask_method': self.mask_method,
            'mask_params': self.mask_params
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank = checkpoint['memory_bank']
        self.model.load_state_dict(checkpoint['model_state'])
        self.global_threshold = checkpoint['global_threshold']
        self.projection = checkpoint.get('projection', None)
        self.feature_layers = checkpoint.get('feature_layers', self.feature_layers)
        self.feature_map_size = checkpoint.get('feature_map_size', None)
        self.feature_mean = checkpoint.get('feature_mean', None)
        self.feature_std = checkpoint.get('feature_std', None)
        self.mask_method = checkpoint.get('mask_method', None)
        self.mask_params = checkpoint.get('mask_params', {})
        
        # Recreate masker if needed
        self.masker = BackgroundMasker() if self.mask_method else None
        
        # Smart FAISS switching at load time
        MIN_MEMORY_BANK_SIZE = 10000  #Only use FAISS for large data sets, else scipy is faster

        if self.memory_bank is not None:
            if FAISS_AVAILABLE and len(self.memory_bank) >= MIN_MEMORY_BANK_SIZE:
                print(f"Using FAISS (memory bank: {len(self.memory_bank)} patches)")
                self.faiss_index = self.setup_faiss_index(self.memory_bank)
            else:
                print(f"Using scipy (memory bank: {len(self.memory_bank)} patches)")
                self.faiss_index = None
        
        print(f"✓ Model loaded from: {path}")
        print("Index type:", type(self.faiss_index))
        print("Bank size :", len(self.memory_bank))
        print(f"self.device: {self.device}")

    def debug_region_sizes(self, image_path):
        """Debug function to understand the relationship between feature and image space"""
    
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)
    
        # Extract features
        features = self.extract_features(image_tensor)
        features_np = features.cpu().numpy().reshape(-1, features.shape[-1])
    
        # Get dimensions
        h, w = self.feature_map_size
        image_size = self.image_size  # or whatever your transform uses
    
        print(f"Image size: {image_size}×{image_size}")
        print(f"Feature map size: {h}×{w}")
        print(f"Feature-to-image ratio: {image_size/h:.2f}×{image_size/w:.2f}")
        print(f"Each feature pixel represents: {(image_size/h)*(image_size/w):.1f} image pixels")
        print()
    
        # Project if needed
        if self.projection is not None:
            features_np = self.projection.transform(features_np)
    
        # Calculate patch scores
        patch_scores = self.calculate_anomaly_score(features_np, return_patch_scores=True)
        score_map = patch_scores.reshape(h, w)
    
        # Create binary mask
        binary_mask = (score_map > self.global_threshold).astype(np.uint8)
    
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
        print(f"Number of anomaly regions found: {num_labels - 1}")  # -1 for background
    
        # Analyze each region
        for i in range(1, num_labels):
            area_feature = stats[i, cv2.CC_STAT_AREA]
            # Convert to approximate image pixels
            area_image = area_feature * (image_size/h) * (image_size/w)
        
            x, y, width, height = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
            print(f"\nRegion {i}:")
            print(f"  Feature space: {area_feature} pixels ({width}×{height})")
            print(f"  Image space (approx): {area_image:.0f} pixels")
            print(f"  Equivalent square size: {np.sqrt(area_image):.1f}×{np.sqrt(area_image):.1f} pixels")
        
        # Test different min_region_size values
        print("\n" + "="*50)
        print("Testing different min_region_size values:")
        print("="*50)
    
        for min_size in [1, 2, 4, 8, 16, 25, 50]:
            surviving_regions = 0
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    surviving_regions += 1
        
            equiv_image_pixels = min_size * (image_size/h) * (image_size/w)
            print(f"min_region_size={min_size:3d} (feature space) ≈ {equiv_image_pixels:6.0f} image pixels "
                  f"→ {surviving_regions} regions survive")

    # Add this method to your SimplePatchCore class and call it like:
    # model.debug_region_sizes("path/to/anomaly/image.jpg")

class SimpleImageDataset(Dataset):
    """Dataset for loading images"""
    def __init__(self, root_dir, transform=None, mask_method=None, mask_params=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_method = mask_method
        self.mask_params = mask_params or {}
        self.masker = BackgroundMasker() if mask_method else None
        
        # Collect all image files
        all_images = []
        all_images.extend(self.root_dir.glob("*.jpg"))
        all_images.extend(self.root_dir.glob("*.png"))
        all_images.extend(self.root_dir.glob("*.jpeg"))
        all_images.extend(self.root_dir.glob("*.JPEG"))
        
        # Remove duplicates (for case-insensitive filesystems)
        # Convert to resolved paths and use a set to remove duplicates
        unique_paths = list(set(img.resolve() for img in all_images))
        self.images = sorted(unique_paths)  # Sort for consistent ordering
        
        print(f"Found {len(self.images)} unique images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply background masking if configured
        if self.masker and self.mask_method:
            if self.mask_method == 'center_crop':
                image = self.masker.center_crop_percent(image, **self.mask_params)
            elif self.mask_method == 'edge_crop':
                image = self.masker.edge_based_crop(image, **self.mask_params)
        
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)