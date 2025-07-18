#!/usr/bin/env python3
"""Feature Space Analysis Benchmark for PatchCore Model Selection"""
import torch
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import shutil
import os
import random
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import entropy
from scipy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import umap
import warnings
warnings.filterwarnings('ignore')

# Import both PatchCore implementations
from patchcore_exth import PatchCore as PatchCoreResNet
from patchcore_dino import PatchCore as PatchCoreDINO


class FeatureSpaceAnalyzer:
    """Analyzes feature space properties of trained PatchCore models"""
    
    def __init__(self, memory_bank, model_type, object_name):
        self.memory_bank = memory_bank
        self.model_type = model_type
        self.object_name = object_name
        self.n_features, self.feature_dim = memory_bank.shape
        print(f"    Memory bank shape: {self.n_features} x {self.feature_dim}")
        
    def compute_all_metrics(self):
        """Compute comprehensive feature space metrics"""
        print(f"  Analyzing {self.model_type} feature space...")
        
        metrics = {}
        
        # Basic properties
        print("    Computing basic properties...")
        metrics.update(self._compute_basic_properties())
        
        # Diversity metrics
        print("    Computing diversity metrics...")
        metrics.update(self._compute_diversity_metrics())
        
        # Clustering metrics
        print("    Computing clustering metrics...")
        metrics.update(self._compute_clustering_metrics())
        
        # Dimensionality metrics
        print("    Computing dimensionality metrics...")
        metrics.update(self._compute_dimensionality_metrics())
        
        # Coverage metrics
        print("    Computing coverage metrics...")
        metrics.update(self._compute_coverage_metrics())
        
        # Manifold metrics - OPTIMIZED
        print("    Computing manifold metrics...")
        metrics.update(self._compute_manifold_metrics())
        
        print("    Analysis complete!")
        return metrics
    
    def _compute_basic_properties(self):
        """Basic statistical properties"""
        metrics = {
            'n_features': self.n_features,
            'feature_dim': self.feature_dim,
            'mean_norm': float(np.mean(np.linalg.norm(self.memory_bank, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(self.memory_bank, axis=1))),
            'mean_feature': float(np.mean(self.memory_bank)),
            'std_feature': float(np.std(self.memory_bank)),
        }
        return metrics
    
    def _compute_diversity_metrics(self):
        """Measure feature diversity"""
        # Sample for large memory banks
        if self.n_features > 5000:
            print(f"      Sampling {5000} features from {self.n_features} for diversity metrics...")
            indices = np.random.choice(self.n_features, 5000, replace=False)
            sampled_bank = self.memory_bank[indices]
        else:
            sampled_bank = self.memory_bank
        
        # Compute pairwise distances
        distances = pairwise_distances(sampled_bank)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        # Nearest neighbor distances
        np.fill_diagonal(distances, np.inf)
        nn_distances = np.min(distances, axis=1)
        
        metrics = {
            'mean_pairwise_distance': float(np.mean(upper_tri)),
            'std_pairwise_distance': float(np.std(upper_tri)),
            'min_pairwise_distance': float(np.min(upper_tri)),
            'max_pairwise_distance': float(np.max(upper_tri)),
            'mean_nn_distance': float(np.mean(nn_distances)),
            'std_nn_distance': float(np.std(nn_distances)),
            'distance_cv': float(np.std(upper_tri) / np.mean(upper_tri)) if np.mean(upper_tri) > 0 else 0,
        }
        
        # Distance distribution entropy
        hist, _ = np.histogram(upper_tri, bins=50)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        metrics['distance_entropy'] = float(entropy(hist))
        
        # Feature correlation - sample dimensions if too many
        if self.feature_dim > 1000:
            dim_indices = np.random.choice(self.feature_dim, 1000, replace=False)
            corr_data = sampled_bank[:, dim_indices]
        else:
            corr_data = sampled_bank
            
        if corr_data.shape[0] > 1:
            corr_matrix = np.corrcoef(corr_data.T)
            metrics['mean_feature_correlation'] = float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
        else:
            metrics['mean_feature_correlation'] = 0
        
        return metrics
    
    def _compute_clustering_metrics(self):
        """Analyze clustering structure"""
        metrics = {}
        
        # Sample for large memory banks
        if self.n_features > 5000:
            indices = np.random.choice(self.n_features, 5000, replace=False)
            sampled_bank = self.memory_bank[indices]
        else:
            sampled_bank = self.memory_bank
        
        # DBSCAN clustering with adaptive eps
        mean_nn_dist = np.mean(np.sort(pairwise_distances(sampled_bank), axis=1)[:, 1])
        
        # Try different eps values
        eps_values = [mean_nn_dist * 0.5, mean_nn_dist, mean_nn_dist * 1.5]
        for eps_mult, eps in zip([0.5, 1.0, 1.5], eps_values):
            clustering = DBSCAN(eps=eps, min_samples=3)
            labels = clustering.fit_predict(sampled_bank)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            metrics[f'n_clusters_eps{eps_mult}'] = n_clusters
            metrics[f'noise_ratio_eps{eps_mult}'] = float(n_noise / len(labels))
        
        # K-means for fixed number of clusters
        if len(sampled_bank) >= 10:
            k = min(10, len(sampled_bank) // 10)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(sampled_bank)
            
            # Cluster sizes
            cluster_sizes = np.bincount(kmeans_labels)
            metrics['cluster_size_std'] = float(np.std(cluster_sizes))
            metrics['cluster_size_cv'] = float(np.std(cluster_sizes) / np.mean(cluster_sizes))
            
            # Within-cluster sum of squares
            metrics['kmeans_inertia'] = float(kmeans.inertia_ / len(sampled_bank))
        
        return metrics
    
    def _compute_dimensionality_metrics(self):
        """Analyze intrinsic dimensionality"""
        metrics = {}
        
        # Sample for large memory banks
        if self.n_features > 10000:
            indices = np.random.choice(self.n_features, 10000, replace=False)
            sampled_bank = self.memory_bank[indices]
        else:
            sampled_bank = self.memory_bank
        
        # PCA analysis
        max_components = min(sampled_bank.shape[0], sampled_bank.shape[1], 100)  # Limit components
        pca = PCA(n_components=max_components)
        pca.fit(sampled_bank)
        
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        # Number of components for different variance thresholds
        for threshold in [0.8, 0.9, 0.95, 0.99]:
            n_components = np.argmax(cumsum_var >= threshold) + 1
            metrics[f'pca_{int(threshold*100)}_components'] = int(n_components)
        
        # Effective rank (components with >1% variance)
        metrics['effective_rank'] = int(np.sum(explained_var > 0.01))
        
        # Spectral entropy
        var_normalized = explained_var / explained_var.sum()
        metrics['spectral_entropy'] = float(-np.sum(var_normalized * np.log(var_normalized + 1e-10)))
        
        # Participation ratio
        metrics['participation_ratio'] = float(np.sum(explained_var)**2 / np.sum(explained_var**2))
        
        # SVD-based metrics - use randomized SVD for speed
        if min(sampled_bank.shape) > 50:
            from sklearn.utils.extmath import randomized_svd
            _, s, _ = randomized_svd(sampled_bank, n_components=50, random_state=42)
            s_normalized = s / s[0]
            
            # Decay rate (slope of log singular values)
            log_s = np.log(s_normalized + 1e-10)
            x = np.arange(len(log_s))
            slope = np.polyfit(x, log_s, 1)[0]
            metrics['singular_value_decay'] = float(abs(slope))
        
        return metrics
    
    def _compute_coverage_metrics(self):
        """Analyze feature space coverage - FIXED VERSION"""
        metrics = {}
        
        # More aggressive sampling for coverage metrics
        max_coverage_samples = 1000  # Reduced from 5000
        if self.n_features > max_coverage_samples:
            print(f"      Sampling {max_coverage_samples} from {self.n_features} features for coverage...")
            indices = np.random.choice(self.n_features, max_coverage_samples, replace=False)
            sampled_bank = self.memory_bank[indices]
        else:
            sampled_bank = self.memory_bank
        
        # Skip convex hull for high dimensions - it's expensive and often fails
        if self.feature_dim <= 5 and len(sampled_bank) > self.feature_dim + 1:
            try:
                hull = ConvexHull(sampled_bank)
                metrics['convex_hull_volume'] = float(hull.volume)
            except:
                metrics['convex_hull_volume'] = None
        else:
            # Use PCA projection only if really needed
            if self.feature_dim > 10 and len(sampled_bank) > 100:
                # Use even fewer components and samples
                n_hull_samples = min(100, len(sampled_bank))
                hull_indices = np.random.choice(len(sampled_bank), n_hull_samples, replace=False)
                hull_data = sampled_bank[hull_indices]
                
                pca = PCA(n_components=min(5, self.feature_dim, len(hull_data)-1))
                projected = pca.fit_transform(hull_data)
                
                if projected.shape[0] > projected.shape[1] + 1:
                    try:
                        hull = ConvexHull(projected)
                        metrics['convex_hull_volume_pca'] = float(hull.volume)
                    except:
                        metrics['convex_hull_volume_pca'] = None
                else:
                    metrics['convex_hull_volume_pca'] = None
            else:
                metrics['convex_hull_volume_pca'] = None
        
        # K-NN with approximate nearest neighbors for speed
        print(f"      Computing k-NN statistics on {len(sampled_bank)} samples...")
        
        # Use smaller k for faster computation
        k = min(5, len(sampled_bank) - 1)  # Reduced from 10
        
        # For very high dimensions, use approximate NN
        if self.feature_dim > 1000 or len(sampled_bank) > 2000:
            # Use approximate nearest neighbors with sklearn's BallTree
            from sklearn.neighbors import BallTree
            
            # Further sample if still too large
            if len(sampled_bank) > 500:
                knn_indices = np.random.choice(len(sampled_bank), 500, replace=False)
                knn_data = sampled_bank[knn_indices]
            else:
                knn_data = sampled_bank
            
            print(f"      Using BallTree approximate NN on {len(knn_data)} samples...")
            tree = BallTree(knn_data, leaf_size=40)
            distances, _ = tree.query(knn_data, k=k+1)
        else:
            # Use exact NN for smaller datasets
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
            nbrs.fit(sampled_bank)
            distances, _ = nbrs.kneighbors(sampled_bank)
            knn_data = sampled_bank
        
        # Average k-NN distance (exclude self which is at index 0)
        knn_distances = distances[:, 1:].mean(axis=1)
        metrics['mean_knn_distance'] = float(np.mean(knn_distances))
        metrics['std_knn_distance'] = float(np.std(knn_distances))
        metrics['knn_distance_cv'] = float(np.std(knn_distances) / np.mean(knn_distances)) if np.mean(knn_distances) > 0 else 0
        
        # Maximum empty sphere radius (largest gap)
        all_nn_distances = distances[:, 1]  # Distance to nearest neighbor
        metrics['max_empty_sphere_radius'] = float(np.max(all_nn_distances))
        
        return metrics
    
    def _compute_manifold_metrics(self):
        """Analyze manifold properties - OPTIMIZED VERSION"""
        metrics = {}
        
        # Sample heavily for manifold analysis
        max_samples = 500  # Reduced from potentially thousands
        if self.n_features > max_samples:
            indices = np.random.choice(self.n_features, max_samples, replace=False)
            sampled_bank = self.memory_bank[indices]
        else:
            sampled_bank = self.memory_bank
        
        # Local dimension estimation using PCA on neighborhoods
        k = min(20, len(sampled_bank) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1)
        nbrs.fit(sampled_bank)
        
        local_dims = []
        n_samples_for_local = min(50, len(sampled_bank))  # Reduced from 100
        sample_indices = np.random.choice(len(sampled_bank), n_samples_for_local, replace=False)
        
        for i in sample_indices:
            _, indices = nbrs.kneighbors([sampled_bank[i]])
            neighbors = sampled_bank[indices[0][1:]]  # Exclude self
            
            # Local PCA
            if len(neighbors) > 3:
                pca = PCA(n_components=min(len(neighbors), 10))  # Limit components
                pca.fit(neighbors)
                # Local dimension = components explaining 90% variance
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                local_dim = np.argmax(cumsum >= 0.9) + 1
                local_dims.append(local_dim)
        
        if local_dims:
            metrics['mean_local_dimension'] = float(np.mean(local_dims))
            metrics['std_local_dimension'] = float(np.std(local_dims))
        
        # Smoothness: variance of distances to k nearest neighbors
        distances, _ = nbrs.kneighbors(sampled_bank)
        distance_vars = np.var(distances[:, 1:], axis=1)
        metrics['mean_distance_variance'] = float(np.mean(distance_vars))
        metrics['manifold_smoothness'] = float(1.0 / (1.0 + np.mean(distance_vars)))
        
        return metrics
    
    def visualize_features(self, save_path):
        """Create visualization of feature space - OPTIMIZED"""
        print(f"    Creating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sample for visualization if too many points
        max_viz_points = 5000
        if self.n_features > max_viz_points:
            viz_indices = np.random.choice(self.n_features, max_viz_points, replace=False)
            viz_bank = self.memory_bank[viz_indices]
        else:
            viz_bank = self.memory_bank
        
        # 1. PCA projection
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(viz_bank)
        
        axes[0, 0].scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.6, s=30)
        axes[0, 0].set_title(f'{self.model_type} - PCA Projection')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        
        # 2. t-SNE projection
        tsne_max = 1000  # Further reduced for speed
        if len(viz_bank) > tsne_max:
            tsne_indices = np.random.choice(len(viz_bank), tsne_max, replace=False)
            tsne_data = viz_bank[tsne_indices]
        else:
            tsne_data = viz_bank
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)-1))
        tsne_features = tsne.fit_transform(tsne_data)
        
        axes[0, 1].scatter(tsne_features[:, 0], tsne_features[:, 1], alpha=0.6, s=30)
        axes[0, 1].set_title(f'{self.model_type} - t-SNE Projection')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        
        # 3. Distance distribution - sample for speed
        sample_for_dist = min(2000, len(viz_bank))
        if len(viz_bank) > sample_for_dist:
            dist_indices = np.random.choice(len(viz_bank), sample_for_dist, replace=False)
            dist_bank = viz_bank[dist_indices]
        else:
            dist_bank = viz_bank
        
        distances = pairwise_distances(dist_bank)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        axes[1, 0].hist(upper_tri, bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title(f'{self.model_type} - Pairwise Distance Distribution')
        axes[1, 0].set_xlabel('Distance')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].axvline(np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.3f}')
        axes[1, 0].legend()
        
        # 4. Explained variance curve
        max_components = min(50, viz_bank.shape[0], viz_bank.shape[1])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(viz_bank)
        
        n_show = len(pca_full.explained_variance_ratio_)
        axes[1, 1].plot(range(1, n_show+1), np.cumsum(pca_full.explained_variance_ratio_), 'b-', linewidth=2)
        axes[1, 1].set_title(f'{self.model_type} - Cumulative Explained Variance')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
        axes[1, 1].axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
        axes[1, 1].legend()
        
        plt.suptitle(f'{self.object_name} - Feature Space Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class FeatureSpaceBenchmark:
    def __init__(self, mvtec_root, output_dir="feature_space_analysis", categories=None, sample_ratio=0.01):
        self.mvtec_root = Path(mvtec_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Random seeds
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        self.categories = categories
        self.sample_ratio = sample_ratio
        
        self.results = {
            'benchmark_start': datetime.now().isoformat(),
            'objects': {}
        }
        
        # Temp directory
        self.temp_dir = Path("temp_models")
        self.temp_dir.mkdir(exist_ok=True)
    
    def discover_objects(self):
        """Find all object folders in MVTec dataset"""
        objects = []
        for item in self.mvtec_root.iterdir():
            if item.is_dir():
                train_path = item / "train" / "good"
                test_path = item / "test"
                if train_path.exists() and test_path.exists():
                    objects.append(item.name)
        
        if self.categories:
            objects = [obj for obj in objects if obj in self.categories]
        
        print(f"Found {len(objects)} objects: {', '.join(sorted(objects))}")
        return sorted(objects)
    
    def train_and_analyze(self, model_type, object_name):
        """Train model and analyze its feature space"""
        print(f"\nTraining {model_type} on {object_name}...")
        
        # Prepare paths
        train_dir = self.mvtec_root / object_name / "train" / "good"
        
        # Initialize model
        if model_type == "ResNet":
            model = PatchCoreResNet(backbone='wide_resnet50_2')
        else:  # DINOv2
            model = PatchCoreDINO(backbone='dinov2_vits14')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train model
        start_time = time.time()
        model.fit(
            train_dir=str(train_dir),
            val_dir=None,  # No validation split needed
            sample_ratio=self.sample_ratio,
            threshold_percentile=95
        )
        train_time = time.time() - start_time
        
        # Extract memory bank - handle both tensor and numpy cases
        memory_bank = model.memory_bank
        if isinstance(memory_bank, torch.Tensor):
            memory_bank = memory_bank.cpu().numpy()
        elif not isinstance(memory_bank, np.ndarray):
            # If it's neither tensor nor numpy array, try to convert
            memory_bank = np.array(memory_bank)
        
        # Analyze feature space
        analyzer = FeatureSpaceAnalyzer(memory_bank, model_type, object_name)
        feature_metrics = analyzer.compute_all_metrics()
        
        # Visualize
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        analyzer.visualize_features(viz_dir / f'{object_name}_{model_type.lower()}.png')
        
        results = {
            'train_time': train_time,
            'feature_metrics': feature_metrics,
            'memory_bank_shape': list(memory_bank.shape)
        }
        
        return model, results
    
    def evaluate_model(self, model, test_dir):
        """Evaluate model AUROC"""
        test_path = Path(test_dir)
        
        all_scores = []
        all_labels = []
        
        for category_dir in test_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            for img_path in category_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    try:
                        result = model.predict(str(img_path), return_heatmap=False)
                        all_scores.append(result['anomaly_score'])
                        all_labels.append(0 if category_name == 'good' else 1)
                    except:
                        continue
        
        if len(set(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_scores)
        else:
            auroc = None
        
        return auroc
    
    def run_benchmark(self):
        """Run the benchmark"""
        objects = self.discover_objects()
        
        for obj_idx, object_name in enumerate(objects):
            print(f"\n{'='*80}")
            print(f"Object {obj_idx+1}/{len(objects)}: {object_name}")
            print(f"{'='*80}")
            
            obj_results = {}
            test_dir = self.mvtec_root / object_name / "test"
            
            # Process ResNet
            try:
                resnet_model, resnet_analysis = self.train_and_analyze("ResNet", object_name)
                resnet_auroc = self.evaluate_model(resnet_model, test_dir)
                
                obj_results['resnet'] = {
                    'auroc': resnet_auroc,
                    'analysis': resnet_analysis
                }
                
                print(f"ResNet AUROC: {resnet_auroc:.3f}")
                
                del resnet_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with ResNet: {e}")
                obj_results['resnet'] = {'error': str(e)}
            
            # Process DINOv2
            try:
                dino_model, dino_analysis = self.train_and_analyze("DINOv2", object_name)
                dino_auroc = self.evaluate_model(dino_model, test_dir)
                
                obj_results['dinov2'] = {
                    'auroc': dino_auroc,
                    'analysis': dino_analysis
                }
                
                print(f"DINOv2 AUROC: {dino_auroc:.3f}")
                
                del dino_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with DINOv2: {e}")
                obj_results['dinov2'] = {'error': str(e)}
            
            # Determine winner
            if 'error' not in obj_results['resnet'] and 'error' not in obj_results['dinov2']:
                resnet_auroc = obj_results['resnet']['auroc']
                dino_auroc = obj_results['dinov2']['auroc']
                
                if resnet_auroc and dino_auroc:
                    obj_results['winner'] = 'resnet' if resnet_auroc > dino_auroc else 'dinov2'
                    obj_results['auroc_difference'] = abs(resnet_auroc - dino_auroc)
            
            self.results['objects'][object_name] = obj_results
            self._save_results()
        
        self.results['benchmark_end'] = datetime.now().isoformat()
        self._save_results()
        self.generate_report()
        
        # Cleanup
        print("\nCleaning up...")
        shutil.rmtree(self.temp_dir)
    
    def _save_results(self):
        """Save results to JSON"""
        with open(self.output_dir / 'feature_space_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self):
        """Generate analysis report"""
        print("\nGenerating feature space analysis report...")
        
        # Collect data
        data = self._collect_analysis_data()
        
        if data.empty:
            print("No valid data for analysis")
            return
        
        # Generate visualizations
        self._generate_comparison_plots(data)
        self._generate_correlation_heatmap(data)
        self._generate_decision_analysis(data)
        
        # Generate text report
        self._generate_text_report(data)
    
    def _collect_analysis_data(self):
        """Organize data for analysis"""
        rows = []
        
        for obj_name, obj_results in self.results['objects'].items():
            if 'winner' not in obj_results:
                continue
            
            row = {
                'object': obj_name,
                'winner': obj_results['winner'],
                'auroc_difference': obj_results['auroc_difference'],
                'resnet_auroc': obj_results['resnet']['auroc'],
                'dinov2_auroc': obj_results['dinov2']['auroc']
            }
            
            # Add feature metrics
            for model in ['resnet', 'dinov2']:
                if model in obj_results and 'analysis' in obj_results[model]:
                    metrics = obj_results[model]['analysis']['feature_metrics']
                    for metric_name, value in metrics.items():
                        if value is not None:
                            row[f'{model}_{metric_name}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_comparison_plots(self, df):
        """Generate comparison visualizations"""
        # Select key metrics for visualization
        key_metrics = [
            'effective_rank', 'mean_pairwise_distance', 'distance_entropy',
            'n_clusters_eps1.0', 'spectral_entropy', 'manifold_smoothness',
            'mean_local_dimension', 'knn_distance_cv'
        ]
        
        # Create figure
        n_metrics = len(key_metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Get data for both models
            resnet_col = f'resnet_{metric}'
            dino_col = f'dinov2_{metric}'
            
            if resnet_col not in df.columns or dino_col not in df.columns:
                ax.set_visible(False)
                continue
            
            # Create scatter plot
            for winner, color in [('resnet', 'blue'), ('dinov2', 'orange')]:
                winner_data = df[df['winner'] == winner]
                ax.scatter(winner_data[resnet_col], winner_data[dino_col], 
                          color=color, alpha=0.6, s=100, label=f'{winner.upper()} wins')
                
                # Add object labels
                for _, row in winner_data.iterrows():
                    ax.annotate(row['object'][:3], 
                               (row[resnet_col], row[dino_col]),
                               fontsize=8, alpha=0.7)
            
            # Add diagonal line
            min_val = min(df[resnet_col].min(), df[dino_col].min())
            max_val = max(df[resnet_col].max(), df[dino_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel(f'ResNet {metric}')
            ax.set_ylabel(f'DINOv2 {metric}')
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused
        for idx in range(len(key_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_space_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_correlation_heatmap(self, df):
        """Generate correlation heatmap"""
        # Calculate correlations between feature metrics and AUROC difference
        metric_columns = [col for col in df.columns if col.startswith(('resnet_', 'dinov2_')) 
                         and not col.endswith('_auroc')]
        
        # Create difference metrics
        diff_metrics = {}
        for col in metric_columns:
            if col.startswith('resnet_'):
                metric_name = col[7:]  # Remove 'resnet_'
                dino_col = f'dinov2_{metric_name}'
                if dino_col in df.columns:
                    diff_metrics[f'diff_{metric_name}'] = df[col] - df[dino_col]
        
        diff_df = pd.DataFrame(diff_metrics)
        
        # Calculate which model wins (1 for ResNet, 0 for DINOv2)
        df['resnet_wins'] = (df['winner'] == 'resnet').astype(int)
        
        # Correlations with winning model
        correlations = {}
        for col in diff_df.columns:
            corr = diff_df[col].corr(df['resnet_wins'])
            if not np.isnan(corr):
                correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Plot top correlations
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_n = min(20, len(sorted_corrs))
        metrics = [x[0].replace('diff_', '') for x in sorted_corrs[:top_n]]
        values = [x[1] for x in sorted_corrs[:top_n]]
        
        colors = ['blue' if v > 0 else 'orange' for v in values]
        
        ax.barh(range(len(metrics)), values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Correlation with ResNet Victory')
        ax.set_title('Feature Metrics Most Predictive of Model Performance')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(0, color='black', linewidth=0.5)
        
        # Add text annotations
        ax.text(0.5, 0.98, 'Higher values favor ResNet →', transform=ax.transAxes, 
                ha='right', va='top', color='blue', fontweight='bold')
        ax.text(0.5, 0.98, '← Higher values favor DINOv2', transform=ax.transAxes, 
                ha='left', va='top', color='orange', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return sorted_corrs
    
    def _generate_decision_analysis(self, df):
        """Generate decision boundary analysis"""
        # UMAP projection of all feature metrics
        metric_columns = [col for col in df.columns if col.startswith(('resnet_', 'dinov2_')) 
                         and not col.endswith('_auroc')]
        
        # Create feature matrix
        feature_data = []
        for _, row in df.iterrows():
            features = []
            for col in sorted(metric_columns):
                if pd.notna(row[col]):
                    features.append(row[col])
            feature_data.append(features)
        
        feature_matrix = np.array(feature_data)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # UMAP projection
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: UMAP of all features
        if len(df) > 3:  # UMAP needs at least 4 points
            reducer = umap.UMAP(random_state=42, n_neighbors=min(len(df)-1, 15))
            embedding = reducer.fit_transform(feature_matrix_scaled)
            
            for winner, color in [('resnet', 'blue'), ('dinov2', 'orange')]:
                mask = df['winner'] == winner
                ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                           color=color, alpha=0.6, s=100, label=f'{winner.upper()} wins')
                
                # Add object labels
                for i, (_, row) in enumerate(df[mask].iterrows()):
                    ax1.annotate(row['object'], 
                               (embedding[mask][i-sum(mask[:i]), 0], 
                                embedding[mask][i-sum(mask[:i]), 1]),
                               fontsize=8, alpha=0.7)
            
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.set_title('UMAP Projection of All Feature Metrics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance difference vs feature complexity
        # Use effective rank as complexity measure
        complexity_metric = 'effective_rank'
        if f'resnet_{complexity_metric}' in df.columns:
            df['complexity_diff'] = df[f'resnet_{complexity_metric}'] - df[f'dinov2_{complexity_metric}']
            df['performance_diff'] = df['resnet_auroc'] - df['dinov2_auroc']
            
            ax2.scatter(df['complexity_diff'], df['performance_diff'], 
                       c=['blue' if w == 'resnet' else 'orange' for w in df['winner']], 
                       alpha=0.6, s=100)
            
            # Add object labels
            for _, row in df.iterrows():
                ax2.annotate(row['object'], 
                           (row['complexity_diff'], row['performance_diff']),
                           fontsize=8, alpha=0.7)
            
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax2.axvline(0, color='gray', linestyle='-', alpha=0.5)
            ax2.set_xlabel('ResNet - DINOv2: Effective Rank')
            ax2.set_ylabel('ResNet - DINOv2: AUROC')
            ax2.set_title('Feature Complexity vs Performance Difference')
            ax2.grid(True, alpha=0.3)
            
            # Add quadrant labels
            ax2.text(0.02, 0.98, 'DINOv2 Better\nSimpler Features', transform=ax2.transAxes,
                    ha='left', va='top', fontsize=10, alpha=0.5, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.2))
            ax2.text(0.98, 0.98, 'DINOv2 Better\nComplex Features', transform=ax2.transAxes,
                    ha='right', va='top', fontsize=10, alpha=0.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.2))
            ax2.text(0.02, 0.02, 'ResNet Better\nSimpler Features', transform=ax2.transAxes,
                    ha='left', va='bottom', fontsize=10, alpha=0.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
            ax2.text(0.98, 0.02, 'ResNet Better\nComplex Features', transform=ax2.transAxes,
                    ha='right', va='bottom', fontsize=10, alpha=0.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.2))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, df):
        """Generate comprehensive text report"""
        sorted_corrs = self._generate_correlation_heatmap(df)
        
        report_lines = [
            "Feature Space Analysis Report",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary",
            "-" * 40,
            f"Total objects analyzed: {len(df)}",
            f"ResNet wins: {len(df[df['winner'] == 'resnet'])} ({100*len(df[df['winner'] == 'resnet'])/len(df):.1f}%)",
            f"DINOv2 wins: {len(df[df['winner'] == 'dinov2'])} ({100*len(df[df['winner'] == 'dinov2'])/len(df):.1f}%)",
            f"Average AUROC difference: {df['auroc_difference'].mean():.3f}",
            "",
            "Per-Object Results",
            "-" * 40
        ]
        
        # Sort by AUROC difference
        df_sorted = df.sort_values('auroc_difference', ascending=False)
        
        for _, row in df_sorted.iterrows():
            report_lines.append(
                f"{row['object']:15s} | Winner: {row['winner']:6s} | "
                f"ResNet: {row['resnet_auroc']:.3f} | DINOv2: {row['dinov2_auroc']:.3f} | "
                f"Diff: {row['auroc_difference']:.3f}"
            )
        
        # Key findings
        report_lines.extend([
            "",
            "Key Feature Space Insights",
            "-" * 40,
            "Top 10 Most Predictive Feature Differences (ResNet - DINOv2):"
        ])
        
        for i, (metric, corr) in enumerate(sorted_corrs[:10]):
            metric_clean = metric.replace('diff_', '').replace('_', ' ').title()
            direction = "Higher favors ResNet" if corr > 0 else "Higher favors DINOv2"
            report_lines.append(f"{i+1:2d}. {metric_clean:40s} | Correlation: {corr:+.3f} | {direction}")
        
        # Feature space characteristics
        report_lines.extend([
            "",
            "Feature Space Characteristics by Model",
            "-" * 40
        ])
        
        # Average metrics for winners
        key_metrics = ['effective_rank', 'mean_pairwise_distance', 'spectral_entropy', 
                      'manifold_smoothness', 'n_clusters_eps1.0']
        
        for metric in key_metrics:
            resnet_col = f'resnet_{metric}'
            dino_col = f'dinov2_{metric}'
            
            if resnet_col in df.columns and dino_col in df.columns:
                resnet_wins = df[df['winner'] == 'resnet']
                dino_wins = df[df['winner'] == 'dinov2']
                
                if len(resnet_wins) > 0 and len(dino_wins) > 0:
                    metric_clean = metric.replace('_', ' ').title()
                    report_lines.append(
                        f"\n{metric_clean}:"
                    )
                    report_lines.append(
                        f"  When ResNet wins: ResNet={resnet_wins[resnet_col].mean():.3f}, "
                        f"DINOv2={resnet_wins[dino_col].mean():.3f}"
                    )
                    report_lines.append(
                        f"  When DINOv2 wins: ResNet={dino_wins[resnet_col].mean():.3f}, "
                        f"DINOv2={dino_wins[dino_col].mean():.3f}"
                    )
        
        # Decision rules
        report_lines.extend([
            "",
            "Discovered Decision Patterns",
            "-" * 40
        ])
        
        # Simple rules based on correlations
        strong_predictors = [(m, c) for m, c in sorted_corrs if abs(c) > 0.3]
        
        if strong_predictors:
            for metric, corr in strong_predictors[:5]:
                metric_clean = metric.replace('diff_', '')
                if corr > 0:
                    report_lines.append(
                        f"• When ResNet has higher {metric_clean} than DINOv2, "
                        f"it tends to perform better (correlation: {corr:.3f})"
                    )
                else:
                    report_lines.append(
                        f"• When DINOv2 has higher {metric_clean} than ResNet, "
                        f"it tends to perform better (correlation: {abs(corr):.3f})"
                    )
        else:
            report_lines.append("No strong predictive patterns found in feature space metrics.")
        
        # Conclusions
        report_lines.extend([
            "",
            "Conclusions",
            "-" * 40
        ])
        
        if len(strong_predictors) > 0:
            report_lines.append(
                "Feature space analysis reveals some predictive patterns, particularly in:"
            )
            for metric, _ in strong_predictors[:3]:
                report_lines.append(f"  - {metric.replace('diff_', '').replace('_', ' ').title()}")
            report_lines.append(
                "\nHowever, the correlations are moderate, suggesting that feature space "
                "properties alone may not fully predict anomaly detection performance."
            )
        else:
            report_lines.append(
                "Feature space metrics show weak correlations with model performance, "
                "suggesting that the choice between ResNet and DINOv2 may depend more on "
                "the specific types of anomalies present rather than general feature space properties."
            )
        
        # Save report
        report_file = self.output_dir / 'feature_space_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nReport saved to: {report_file}")
        print(f"Visualizations saved to: {self.output_dir}")


def main():
    """Run the feature space analysis benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Space Analysis for PatchCore Model Selection')
    parser.add_argument('--mvtec-root', type=str, default='mvtec',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--output-dir', type=str, default='feature_space_analysis',
                        help='Output directory for results')
    parser.add_argument('--category', type=str, nargs='*', default=None,
                        help='Subset of categories to analyze (e.g. --category leather bottle)')
    parser.add_argument('--sample-ratio', type=float, default=0.01,
                        help='Sample ratio for PatchCore memory bank (default: 0.01)')
    
    args = parser.parse_args()
    
    # Check MVTec directory
    if not Path(args.mvtec_root).exists():
        print(f"Error: MVTec directory not found at {args.mvtec_root}")
        return
    
    # Run benchmark
    benchmark = FeatureSpaceBenchmark(
        args.mvtec_root,
        args.output_dir,
        categories=args.category,
        sample_ratio=args.sample_ratio
    )
    benchmark.run_benchmark()
    
    print("\nFeature space analysis completed!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"Check the visualizations and report for insights.")


if __name__ == "__main__":
    main()