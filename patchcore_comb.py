#!/usr/bin/env python3
"""PatchCore Hybrid - Combining WideResNet and DINOv2 features"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from pathlib import Path
from tqdm import tqdm
import time
import cv2

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class PatchCore:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(f"self.device: {self.device} ")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(torch.cuda.get_device_name(0))
        print(f"FAISS available: {FAISS_AVAILABLE}  ")
        
        # Initialize WideResNet
        print("Loading WideResNet50_2...")
        self.wr_model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.wr_feature_layers = ['layer2', 'layer3']
        self._setup_wideresnet()
        
        # Initialize DINOv2
        print("Loading DINOv2 vits14...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_model.to(self.device)
        self.dino_model.eval()
        
        # Disable gradients for both models
        for param in self.wr_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        # Feature dimensions
        self.wr_feature_dim = 1792  # 768 + 1024 from layer2 + layer3
        self.dino_feature_dim = 768
        self.combined_feature_dim = self.wr_feature_dim + self.dino_feature_dim
        
        print(f"Combined feature dimension: {self.combined_feature_dim}")
        
        # Storage
        self.memory_bank = None
        self.global_threshold = None
        self.projection = None
        self.faiss_index = None
        
        # Store feature map size
        self.feature_map_size = None
        
        # Image preprocessing
        self.image_size = 518  # Divisible by 14 for DINOv2
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
    
    def _setup_wideresnet(self):
        """Setup WideResNet feature extraction"""
        # Replace BatchNorm with fixed behavior
        for module in self.wr_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                module.momentum = 0
        
        # Setup feature extractor
        self.wr_features = {}
        def hook_fn(name):
            def hook(module, input, output):
                self.wr_features[name] = output
            return hook
        
        for name, module in self.wr_model.named_modules():
            if name in self.wr_feature_layers:
                module.register_forward_hook(hook_fn(name))
        
        self.wr_model.to(self.device)
        self.wr_model.eval()
    
    def extract_wr_features(self, images, use_neighborhood_aggregation=True):
        """Extract WideResNet features with neighborhood aggregation"""
        features = []
        neighborhood_size = 3
        
        with torch.no_grad():
            _ = self.wr_model(images)
            reference_size = None
            
            for layer_name in self.wr_feature_layers:
                layer_features = self.wr_features[layer_name]
                b, c, h, w = layer_features.shape
                
                if reference_size is None:
                    reference_size = (h, w)
                
                if (h, w) != reference_size:
                    layer_features = F.interpolate(
                        layer_features, 
                        size=reference_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Apply neighborhood aggregation for WideResNet
                if use_neighborhood_aggregation:
                    layer_features = F.avg_pool2d(
                        layer_features,
                        kernel_size=neighborhood_size,
                        stride=1,
                        padding=neighborhood_size // 2
                    )
                
                layer_features = layer_features.permute(0, 2, 3, 1).reshape(b, reference_size[0]*reference_size[1], c)
                features.append(layer_features)
        
        return torch.cat(features, dim=-1), reference_size
    
    def extract_dino_features(self, images):
        """Extract DINOv2 features without neighborhood aggregation"""
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(images)
            features = features_dict['x_norm_patchtokens']
            
            # features shape: (B, N_patches, D)
            b, n_patches, d = features.shape
            
            # Calculate spatial dimensions
            h = w = int(np.sqrt(n_patches))
            
            return features, (h, w)
    
    def extract_features(self, images, return_spatial=False):
        """Extract and combine features from both models"""
        # Extract WideResNet features with neighborhood aggregation
        wr_features, wr_spatial_size = self.extract_wr_features(images, use_neighborhood_aggregation=True)
        
        # Extract DINOv2 features without neighborhood aggregation
        dino_features, dino_spatial_size = self.extract_dino_features(images)
        
        # Verify spatial dimensions match
        wr_patches = wr_features.shape[1]
        dino_patches = dino_features.shape[1]
        
        if wr_patches != dino_patches:
            # Interpolate WideResNet features to match DINOv2 spatial size
            b = wr_features.shape[0]
            h_wr, w_wr = wr_spatial_size
            h_dino, w_dino = dino_spatial_size
            
            wr_features_spatial = wr_features.reshape(b, h_wr, w_wr, -1).permute(0, 3, 1, 2)
            wr_features_spatial = F.interpolate(
                wr_features_spatial,
                size=(h_dino, w_dino),
                mode='bilinear',
                align_corners=False
            )
            wr_features = wr_features_spatial.permute(0, 2, 3, 1).reshape(b, h_dino * w_dino, -1)
            
            self.feature_map_size = dino_spatial_size
        else:
            self.feature_map_size = dino_spatial_size
        
        # Simply concatenate features
        combined_features = torch.cat([wr_features, dino_features], dim=-1)
        
        if return_spatial:
            # For heatmap generation
            b, n_patches, d = combined_features.shape
            h, w = self.feature_map_size
            spatial_features = combined_features.reshape(b, h, w, d).permute(0, 3, 1, 2)
            return combined_features, spatial_features
        
        return combined_features
    
    def adaptive_sampling(self, features, sampling_ratio=0.01, callback=None):
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
            
            # Update Progress to caller
            if callback and i % 100 == 0:  # Call every 100 iterations
                callback(i, n_samples - 1)
        
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
    
    def fit(self, train_dir, sample_ratio=0.01, threshold_percentile=99, val_dir=None, callback=None):
        """Optimized training with faster data loading and processing"""
        print(f"Training Hybrid PatchCore (WideResNet + DINOv2) on: {train_dir}")
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
        print(f"Extracting hybrid features from {len(dataloader)} batches...")
        start_time2 = time.time()
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            features = self.extract_features(images)
            all_features.append(features.cpu().numpy())
        total_time2 = time.time() - start_time2
        print(f"Extracted in {total_time2:.2f} seconds!")    

        # Concatenate all features
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        print(f"Total features extracted: {all_features.shape}")
        
        # Dimensionality reduction (optional for combined features)
        if all_features.shape[1] > 1024:
            print("Applying dimensionality reduction...")
            self.projection = GaussianRandomProjection(n_components=1024, random_state=42)
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
            selected_indices = self.adaptive_sampling(all_features, sample_ratio, callback=callback)

        self.memory_bank = all_features[selected_indices]
        print(f"Memory bank size: {self.memory_bank.shape}")

        # FAISS-Index
        if FAISS_AVAILABLE:
            self.faiss_index = self.setup_faiss_index(self.memory_bank)
        else:
            self.faiss_index = None

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
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if FAISS_AVAILABLE and self.faiss_index is not None:
            distances, _ = self.faiss_index.search(features.astype(np.float32), k=1)
            min_distances = np.sqrt(distances.squeeze())
        else:
            # Scipy fallback for CPU
            distances = cdist(features, self.memory_bank, metric='euclidean')
            min_distances = np.min(distances, axis=1)
        
        # 1D min_distances
        if len(min_distances.shape) == 0:
            min_distances = np.array([min_distances])
        
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
        print(f"Heatmap generation {execution_timei:.4f} seconds")
        return overlay

    def predict(self, image_path, return_heatmap=True, min_region_size=None):
        """Predict with region filtering"""
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform_test(image).unsqueeze(0).to(self.device)        
  
        # Extract features 
        features = self.extract_features(image_tensor)
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
            'wr_model_state': self.wr_model.state_dict(),
            'dino_model_state': self.dino_model.state_dict(),
            'global_threshold': self.global_threshold,
            'projection': self.projection,
            'feature_map_size': self.feature_map_size,
            'combined_feature_dim': self.combined_feature_dim
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.memory_bank = checkpoint['memory_bank']
        self.wr_model.load_state_dict(checkpoint['wr_model_state'])
        self.dino_model.load_state_dict(checkpoint['dino_model_state'])
        self.global_threshold = checkpoint['global_threshold']
        self.projection = checkpoint.get('projection', None)
        self.feature_map_size = checkpoint.get('feature_map_size', None)
        self.combined_feature_dim = checkpoint.get('combined_feature_dim', self.combined_feature_dim)

        # Re-build FAISS-Index if available
        if self.memory_bank is not None and FAISS_AVAILABLE:
            self.faiss_index = self.setup_faiss_index(self.memory_bank)
            print(f"✓ Memory bank loaded: {self.memory_bank.shape}")
        else:
            self.faiss_index = None

        print(f"✓ Model loaded from: {path}")
        print("Index type:", type(self.faiss_index))
        print("Bank size:", len(self.memory_bank))


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