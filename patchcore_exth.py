#!/usr/bin/env python3
"""
Advanced PatchCore Implementation with Performance Optimizations
- PyTorch-based nearest neighbor search (replacing FAISS)
- Optimized training with mixed precision and better data loading
- Faster coreset sampling with GPU acceleration
"""

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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from pathlib import Path
from tqdm import tqdm
import os
import time
import cv2

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SimplePatchCore:
    """Optimized PatchCore with PyTorch-based inference and lazy initialization"""
    
    def __init__(self, backbone='wide_resnet50_2', device='cuda'):
        self.device = device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(f"self.device: {self.device} ")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(torch.cuda.get_device_name(0))  # Should show RTX 3060
        print(f"FAISS available: {FAISS_AVAILABLE}  ")
        if backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.feature_layers = ['layer2', 'layer3']
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_layers = ['layer2', 'layer3']
        
        # Replace BatchNorm with GroupNorm to avoid batch size effects
        self._replace_batchnorm_with_groupnorm()
        
        # Setup feature extractor
        self.feature_extractor = self._setup_feature_extractor()
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Storage
        self.memory_bank = None
        self.global_threshold = None
        self.projection = None
        self.faiss_index = None

        # Inferenz-Backend: Immer FAISS wenn verfügbar, sonst scipy
        self.use_faiss_inference = FAISS_AVAILABLE
        
        # Store normalization parameters
        self.feature_mean = None
        self.feature_std = None
        
        # Store feature map size
        self.feature_map_size = None
        
        # Image preprocessing
        self.image_size = 512
        self.transform_train = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _replace_batchnorm_with_groupnorm(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                module.momentum = 0

    def _replace_batchnorm_recursive(self, module):
        pass

    def _setup_feature_extractor(self):
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
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
        """Optimized K-center greedy sampling with GPU acceleration"""
        n_samples = int(len(features) * sampling_ratio)
        if n_samples >= len(features):
            return np.arange(len(features))
        
        print(f"🚀 Fast GPU coreset sampling: {len(features)} -> {n_samples}")
        
        # Convert to GPU tensor with appropriate precision
        if torch.cuda.is_available():
            # Use float16 for large datasets, float32 for smaller ones
            if len(features) > 100000:
                features_torch = torch.from_numpy(features).half().cuda()
                dtype = torch.float16
            else:
                features_torch = torch.from_numpy(features).float().cuda()
                dtype = torch.float32
        else:
            features_torch = torch.from_numpy(features).float()
            dtype = torch.float32
        
        n_features = len(features)
        selected_indices = [np.random.randint(n_features)]
        selected_mask = torch.zeros(n_features, dtype=torch.bool, device=features_torch.device)
        
        # Pre-allocate distance matrix
        min_distances = torch.full((n_features,), float('inf'), 
                                 dtype=dtype,
                                 device=features_torch.device)
        
        # Batch distance calculations
        batch_size = 50000 if torch.cuda.is_available() else 10000
        
        for i in tqdm(range(n_samples - 1), desc="Fast coreset sampling"):
            new_center_idx = selected_indices[-1]
            new_center = features_torch[new_center_idx:new_center_idx+1]
            
            # Calculate distances in batches
            for start_idx in range(0, n_features, batch_size):
                end_idx = min(start_idx + batch_size, n_features)
                batch_features = features_torch[start_idx:end_idx]
                
                distances = torch.cdist(batch_features, new_center).squeeze()
                min_distances[start_idx:end_idx] = torch.minimum(
                    min_distances[start_idx:end_idx], 
                    distances
                )
            
            # Mark selected indices
            selected_mask[selected_indices] = True
            
            # Find next index
            masked_distances = min_distances.clone()
            masked_distances[selected_mask] = -1
            next_idx = torch.argmax(masked_distances).item()
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
        """Optimized training with faster data loading and processing"""
        print(f"Training Optimized PatchCore on: {train_dir}")
        start_time = time.time()
        
        # Create dataset
        dataset = SimpleImageDataset(train_dir, transform=self.transform_train)
        
        print(f"Total images: {len(dataset)}")
        if len(dataset) < 100:
            num_workers = 0  # Disable workers for small datasets
        else: 
            num_workers = 2
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=num_workers)

        all_features = []
        
        # Extract features
        
        print(f"Extracting multi-layer features from {len(dataloader)} batches...")
        start_time2 = time.time()
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            features = self.extract_features(images)
            all_features.append(features.cpu().numpy())
        total_time2 = time.time() - start_time2
        print(f"Extracted {total_time2:.2f} seconds!")    

        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        print(f"Total features extracted: {all_features.shape}")
        
        # Dimensionality reduction
        if all_features.shape[1] > 512:
            print("Applying dimensionality reduction...")
            self.projection = GaussianRandomProjection(n_components=512, random_state=42)
            all_features = self.projection.fit_transform(all_features)
        
        # Smart sampling strategy
        if sample_ratio >= 0.5:
            print(f"High sample ratio ({sample_ratio}) - using simple sampling")
            if sample_ratio >= 0.99:
                selected_indices = np.arange(len(all_features))
            else:
                n_samples = int(len(all_features) * sample_ratio)
                selected_indices = np.random.choice(len(all_features), n_samples, replace=False)
        else:
            selected_indices = self.adaptive_sampling(all_features, sample_ratio)
        
        self.memory_bank = all_features[selected_indices]
        print(f"Memory bank size: {self.memory_bank.shape}")

        # FAISS-Index nur, wenn verfügbar
        if FAISS_AVAILABLE:
            self.faiss_index = self.setup_faiss_index(self.memory_bank)
        else:
            self.faiss_index = None

        # Kein memory_bank_torch mehr!
        # Calculate threshold
        validation_dir = val_dir if val_dir is not None else train_dir
        if val_dir is not None:
            print(f"Using separate validation directory: {val_dir}")
        else:
            print("Warning: Using training directory for validation")
            
        self.calculate_threshold_validation(validation_dir, percentile=threshold_percentile)
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f} seconds!")
    
    def calculate_threshold_validation(self, val_dir, percentile=99):
        """Calculate threshold from validation set"""
        print(f"Calculating threshold from validation directory: {val_dir}")
        
        dataset = SimpleImageDataset(val_dir, transform=self.transform_test)
        
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
        """Calculate 2D anomaly score with FAISS or scipy fallback"""
        start_time = time.perf_counter()
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if self.use_faiss_inference and self.faiss_index is not None:
            distances, _ = self.faiss_index.search(features.astype(np.float32), k=1)
            min_distances = np.sqrt(distances.squeeze())
        else:
            # Scipy fallback for CPU
            distances = cdist(features, self.memory_bank, metric='euclidean')
            min_distances = np.min(distances, axis=1)
        # 1D min_distances
        if len(min_distances.shape) == 0:
            min_distances = np.array([min_distances])

        execution_time = time.perf_counter() - start_time
        print(f"calculate anomaly took {execution_time:.4f} seconds")
        if return_patch_scores:
            return min_distances

        anomaly_score = np.max(min_distances)
        return anomaly_score
    
    def generate_heatmap(self, image_path, patch_scores, alpha=0.5, colormap='jet', save_path=None):
        """Fast heatmap generation"""
        start_timei = time.perf_counter()

        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        image_height, image_width = image_np.shape[:2]
        
        #image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)

        h, w = self.feature_map_size
        score_map = patch_scores.reshape(h, w)
        
        # Resize to original dimensions
        score_map_resized = cv2.resize(
            score_map.astype(np.float32), 
            (image_width, image_height), 
            interpolation=cv2.INTER_LINEAR
        )

        # Smooth the score map
        score_map_smooth = gaussian_filter(score_map_resized, sigma=2.0)
        
        # Normalize
        score_min, score_max = score_map_smooth.min(), score_map_smooth.max()
        if score_max > score_min:
            score_map_norm = (score_map_smooth - score_min) / (score_max - score_min)
        else:
            score_map_norm = np.zeros_like(score_map_smooth)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(score_map_norm)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay
        overlay = (image_np.astype(np.float32) * (1.0 - alpha) + 
                  heatmap_colored.astype(np.float32) * alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        if save_path:
            Image.fromarray(overlay).save(save_path)
        execution_timei = time.perf_counter() - start_timei
        print(f"Heatmap generation  {execution_timei:.4f} seconds")
        return overlay

    def predict(self, image_path, return_heatmap=True, min_region_size=None):
        """Predict with region filtering"""
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)
        
  
        # Extract features will take the most time 
        features = self.extract_features(image_tensor)

        # Doesnt take too much time
        features_np = features.cpu().numpy().reshape(-1, features.shape[-1])

        # Project if needed
        if self.projection is not None:
            features_np = self.projection.transform(features_np) 

        # Calculate score
        patch_scores = self.calculate_anomaly_score(features_np, return_patch_scores=True)
        anomaly_score = np.max(patch_scores)
        is_anomaly = anomaly_score > self.global_threshold
        
        # Apply region-based filtering if requested
        if min_region_size is not None and is_anomaly:
            # Reshape to spatial dimensions
            h, w = self.feature_map_size
            score_map = patch_scores.reshape(h, w)
            
            # Upsample to image resolution
            image_height, image_width = self.image_size, self.image_size 
            score_map_image = cv2.resize(
                score_map.astype(np.float32), 
                (image_width, image_height), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Create binary mask
            binary_mask_image = (score_map_image > self.global_threshold).astype(np.uint8)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_image, connectivity=8)
            
            # Check for regions larger than threshold
            large_regions_exist = False
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_region_size:
                    large_regions_exist = True
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    print(f"Region {i}: area={area} px, bbox=({x},{y},{w},{h})")
            
            if not large_regions_exist:
                is_anomaly = False
        
        result = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'threshold': float(self.global_threshold)
        }
        
        if min_region_size is not None:
            result['min_region_size'] = min_region_size
            result['region_filtered'] = not is_anomaly and anomaly_score > self.global_threshold
        
        if return_heatmap:
            try:
                heatmap = self.generate_heatmap(image_path, patch_scores)
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
            'feature_std': self.feature_std
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

        # FAISS-Index immer neu bauen, wenn verfügbar
        if self.memory_bank is not None and FAISS_AVAILABLE:
            self.faiss_index = self.setup_faiss_index(self.memory_bank)
            print(f"✓ Memory bank loaded: {self.memory_bank.shape}")
        else:
            self.faiss_index = None

        print(f"✓ Model loaded from: {path}")
        print("Index type:", type(self.faiss_index))
        print("Bank size :", len(self.memory_bank))

class SimpleImageDataset(Dataset):
   """Optimized dataset for loading images"""
   def __init__(self, root_dir, transform=None):
       self.root_dir = Path(root_dir)
       self.transform = transform
       
       # Collect all image files
       all_images = []
       extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPEG', '*.JPG', '*.PNG']
       for ext in extensions:
           all_images.extend(self.root_dir.glob(ext))
       
       # Remove duplicates
       unique_paths = list(set(img.resolve() for img in all_images))
       self.images = sorted(unique_paths)
       
       print(f"Found {len(self.images)} unique images in {root_dir}")
   
   def __len__(self):
       return len(self.images)
   
   def __getitem__(self, idx):
       img_path = self.images[idx]
       
       image = Image.open(img_path).convert('RGB')
       
       if self.transform:
           image = self.transform(image)
       
       return image, str(img_path)
