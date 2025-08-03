#!/usr/bin/env python3
"""Compare different normalization methods for confusion score - Optimized with Pytorch"""
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.stats import gaussian_kde
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from patchcore_dino import PatchCore as PatchCoreDINO


def compute_pairwise_distances_torch_gpu(memory_bank, n_bins=1000):
    """Compute distance statistics using PyTorch on GPU with exact histogram"""
    n_features = len(memory_bank)
    
    # Convert to torch tensor
    if torch.cuda.is_available():
        memory_torch = torch.from_numpy(memory_bank).float().cuda()
        device = 'cuda'
    else:
        memory_torch = torch.from_numpy(memory_bank).float()
        device = 'cpu'
    
    print(f"  Computing distance statistics for {n_features} points using PyTorch {device.upper()}...")
    
    # ========== FIRST PASS: Find min/max and compute statistics ==========
    print("  First pass: Finding range and computing statistics...")
    
    distance_sum = 0.0
    distance_sum_sq = 0.0
    n_distances = 0
    min_dist = float('inf')
    max_dist = float('-inf')
    
    # Batch size based on available memory
    if device == 'cuda':
        if n_features > 30000:
            batch_size = 5000
        elif n_features > 20000:
            batch_size = 8000
        else:
            batch_size = 10000
    else:
        batch_size = 2000
    
    # First pass: get statistics and range
    total_batches = (n_features + batch_size - 1) // batch_size
    with tqdm(total=total_batches * (total_batches + 1) // 2, desc="First pass") as pbar:
        for i in range(0, n_features, batch_size):
            batch1 = memory_torch[i:min(i+batch_size, n_features)]
            
            for j in range(i, n_features, batch_size):
                batch2 = memory_torch[j:min(j+batch_size, n_features)]
                
                # Compute distances
                distances = torch.cdist(batch1, batch2)
                
                # If same batch, only take upper triangular
                if i == j:
                    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                    batch_distances = distances[mask]
                else:
                    batch_distances = distances.flatten()
                
                if len(batch_distances) > 0:
                    # Update statistics
                    distance_sum += batch_distances.sum().item()
                    distance_sum_sq += (batch_distances ** 2).sum().item()
                    n_distances += len(batch_distances)
                    
                    # Update min/max
                    batch_min = batch_distances.min().item()
                    batch_max = batch_distances.max().item()
                    min_dist = min(min_dist, batch_min)
                    max_dist = max(max_dist, batch_max)
                
                pbar.update(1)
    
    # Calculate statistics
    mean_dist = distance_sum / n_distances
    variance = (distance_sum_sq / n_distances) - (mean_dist ** 2)
    std_dist = np.sqrt(variance)
    
    print(f"  Range: [{min_dist:.4f}, {max_dist:.4f}]")
    print(f"  Mean: {mean_dist:.4f}, Std: {std_dist:.4f}")
    
    # ========== SECOND PASS: Build exact histogram ==========
    print(f"  Second pass: Building histogram with {n_bins} bins...")
    
    # Create histogram bins
    # Add small epsilon to max to ensure all values fall within bins
    eps = (max_dist - min_dist) * 1e-6
    bin_edges = np.linspace(min_dist, max_dist + eps, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    
    # Convert bin edges to torch tensor
    bin_edges_torch = torch.from_numpy(bin_edges).to(device)
    
    # Second pass: fill histogram
    with tqdm(total=total_batches * (total_batches + 1) // 2, desc="Second pass") as pbar:
        for i in range(0, n_features, batch_size):
            batch1 = memory_torch[i:min(i+batch_size, n_features)]
            
            for j in range(i, n_features, batch_size):
                batch2 = memory_torch[j:min(j+batch_size, n_features)]
                
                # Compute distances
                distances = torch.cdist(batch1, batch2)
                
                # If same batch, only take upper triangular
                if i == j:
                    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                    batch_distances = distances[mask]
                else:
                    batch_distances = distances.flatten()
                
                if len(batch_distances) > 0:
                    # Compute which bin each distance belongs to
                    # searchsorted gives the insertion point, subtract 1 for bin index
                    bin_indices = torch.searchsorted(bin_edges_torch[:-1], batch_distances, right=False)
                    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)  # Safety clamp
                    
                    # Count occurrences in each bin
                    for bin_idx in range(n_bins):
                        count = (bin_indices == bin_idx).sum().item()
                        bin_counts[bin_idx] += count
                
                pbar.update(1)
    
    # Clear GPU memory
    if device == 'cuda':
        del memory_torch
        torch.cuda.empty_cache()
    
    print(f"  Processed {n_distances:,} distances")
    print(f"  Built exact histogram with {n_bins} bins")
    
    # Find mode from histogram
    mode_bin_idx = np.argmax(bin_counts)
    mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2
    
    print(f"  Mode (from histogram): {mode_value:.4f}")
    
    return {
        'mean': mean_dist,
        'std': std_dist,
        'min': min_dist,
        'max': max_dist,
        'mode': mode_value,
        'n_distances': n_distances,
        'histogram': {
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2
        }
    }


def compute_confusion_scores(distance_stats):
    """Compute confusion scores using exact histogram data"""
    mean_dist = distance_stats['mean']
    std_dist = distance_stats['std']
    min_dist = distance_stats['min']
    max_dist = distance_stats['max']
    mode_dist = distance_stats['mode']
    
    histogram = distance_stats['histogram']
    bin_centers = histogram['bin_centers']
    bin_counts = histogram['bin_counts']
    bin_width = bin_centers[1] - bin_centers[0]
    
    # Raw difference
    mode_mean_diff = mode_dist - mean_dist
    
    # Method 1: Normalize by mean
    if mean_dist > 0:
        confusion_by_mean = mode_mean_diff / mean_dist
    else:
        confusion_by_mean = 0
    
    # Only positive values (mode > mean indicates confusion)
    confusion_by_mean = max(0, confusion_by_mean)
    
    # Calculate mean position in distribution
    range_dist = max_dist - min_dist
    
    # Mean position (0 = left edge, 0.5 = center, 1 = right edge)
    if range_dist > 0:
        mean_position = (mean_dist - min_dist) / range_dist
    else:
        mean_position = 0.5
    
    # Position confusion: how far mean is from center (0.5)
    position_confusion = max(0, mean_position - 0.5) * 4  # Scale to 0-2
    
    # Density gap confusion using histogram counts
    density_gap_confusion = 0
    if mode_dist > mean_dist:
        # Find bins for mean and mode
        mean_bin_idx = np.searchsorted(histogram['bin_edges'][:-1], mean_dist, side='right') - 1
        mean_bin_idx = np.clip(mean_bin_idx, 0, len(bin_counts) - 1)
        mode_bin_idx = np.argmax(bin_counts)
        
        # Get "density" (count) at mean and mode
        density_at_mean = bin_counts[mean_bin_idx]
        density_at_mode = bin_counts[mode_bin_idx]
        
        # Calculate density gap
        if density_at_mode > 0:
            density_gap_confusion = 1 - (density_at_mean / density_at_mode)
        density_gap_confusion = max(0, density_gap_confusion)
    
    # Combined score with 3 components (equal weighting)
    confusion_combined = (confusion_by_mean + position_confusion + density_gap_confusion) / 3
    
    return {
        'mode': mode_dist,
        'mean': mean_dist,
        'std': std_dist,
        'mode_mean_diff': mode_mean_diff,
        'confusion_by_mean': confusion_by_mean,
        'mean_position': mean_position,
        'position_confusion': position_confusion,
        'density_gap_confusion': density_gap_confusion,
        'confusion_combined': confusion_combined
    }


def plot_distance_distribution(distance_stats, scores, object_name, output_dir, auroc=None):
    """Plot exact distance distribution from histogram data"""
    plt.figure(figsize=(10, 6))
    
    # Extract histogram data
    histogram = distance_stats['histogram']
    bin_centers = histogram['bin_centers']
    bin_counts = histogram['bin_counts']
    bin_edges = histogram['bin_edges']
    n_distances = distance_stats['n_distances']
    
    # Convert counts to density
    bin_width = bin_edges[1] - bin_edges[0]
    densities = bin_counts / (n_distances * bin_width)
    
    # Plot histogram as bar chart
    plt.bar(bin_centers, densities, width=bin_width, alpha=0.6, 
            color='blue', edgecolor='black', linewidth=0.5, label='Exact distribution')
    
    # Fit and plot Gaussian for comparison
    mean, std = scores['mean'], scores['std']
    x = np.linspace(distance_stats['min'], distance_stats['max'], 1000)
    gaussian = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)
    plt.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit (μ={mean:.3f}, σ={std:.3f})')
    
    # Add vertical lines
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    plt.axvline(scores['mode'], color='green', linestyle='--', linewidth=2, label=f'Mode: {scores["mode"]:.3f}')
    
    # Highlight confusion region if mode > mean
    if scores['mode'] > mean:
        plt.axvspan(mean, scores['mode'], alpha=0.2, color='orange', label='Confusion region')
    
    # Labels and formatting
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    auroc_str = f'{auroc:.3f}' if auroc is not None else 'N/A'
    plt.title(f'{object_name} - NN Distance Distribution (Exact)\n' + 
              f'Combined Confusion Score: {scores["confusion_combined"]:.3f}, ' +
              f'AUROC: {auroc_str}', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_dir / f'{object_name}_distance_distribution.png', dpi=150)
    plt.close()


class NormalizationTester:
    def __init__(self, mvtec_root, output_dir="normalization_comparison", sample_ratio=0.01):
        self.mvtec_root = Path(mvtec_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_ratio = sample_ratio
        
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def discover_objects(self):
        """Find all object folders in MVTec dataset"""
        objects = []
        for item in self.mvtec_root.iterdir():
            if item.is_dir():
                train_path = item / "train" / "good"
                test_path = item / "test"
                if train_path.exists() and test_path.exists():
                    objects.append(item.name)
        return sorted(objects)
    
    def analyze_object(self, object_name):
        """Analyze a single object"""
        print(f"\nAnalyzing {object_name}...")
        
        # Train model
        train_dir = self.mvtec_root / object_name / "train" / "good"
        model = PatchCoreDINO(backbone='dinov2_vits14')
        
        model.fit(
            train_dir=str(train_dir),
            val_dir=None,
            sample_ratio=self.sample_ratio,
            threshold_percentile=95
        )
        
        # Extract memory bank
        memory_bank = model.memory_bank
        if isinstance(memory_bank, torch.Tensor):
            memory_bank = memory_bank.cpu().numpy()
        
        print(f"  Memory bank size: {memory_bank.shape}")
        
        # Use PyTorch GPU for exact histogram computation - NO SAMPLING!
        distance_stats = compute_pairwise_distances_torch_gpu(memory_bank, n_bins=1000)
        
        # Compute all confusion scores using exact histogram
        scores = compute_confusion_scores(distance_stats)
        
        # Evaluate AUROC
        test_dir = self.mvtec_root / object_name / "test"
        auroc = self.evaluate_model(model, test_dir)
        
        # Plot distribution
        plot_distance_distribution(distance_stats, scores, object_name, self.output_dir, auroc)
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'auroc': auroc,
            'distance_stats': distance_stats,  # Keep full stats for analysis
            **scores
        }
    
    def evaluate_model(self, model, test_dir):
        """Evaluate model AUROC"""
        all_scores = []
        all_labels = []
        
        for category_dir in test_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            for img_path in category_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    try:
                        result = model.predict(str(img_path), return_heatmap=False)
                        all_scores.append(result['anomaly_score'])
                        all_labels.append(0 if category_name == 'good' else 1)
                    except:
                        continue
        
        if len(set(all_labels)) > 1:
            return roc_auc_score(all_labels, all_scores)
        return None
    
    def run_comparison(self):
        """Run comparison analysis"""
        objects = self.discover_objects()
        results = {}
        
        print(f"Found {len(objects)} objects to analyze")
        print(f"Using PyTorch {'GPU' if torch.cuda.is_available() else 'CPU'} for distance calculations")
        print("Using exact histogram approach (no sampling) for accurate mode and density calculations")
        
        for obj in tqdm(objects, desc="Analyzing objects"):
            obj_results = self.analyze_object(obj)
            if obj_results['auroc'] is not None:
                results[obj] = obj_results
                print(f"  {obj}: AUROC={obj_results['auroc']:.3f}")
                # Print confusion score details after each run
                print("  Confusion Score Details:")
                print(f"    mode: {obj_results['mode']:.4f}")
                print(f"    mean: {obj_results['mean']:.4f}")
                print(f"    std: {obj_results['std']:.4f}")
                print(f"    mode_mean_diff: {obj_results['mode_mean_diff']:.4f}")
                print(f"    confusion_by_mean: {obj_results['confusion_by_mean']:.4f}")
                print(f"    mean_position: {obj_results['mean_position']:.4f}")
                print(f"    position_confusion: {obj_results['position_confusion']:.4f}")
                print(f"    density_gap_confusion: {obj_results['density_gap_confusion']:.4f}")
                print(f"    confusion_combined: {obj_results['confusion_combined']:.4f}")
        
        # Save raw results - convert numpy types to Python native types
        save_results = {}
        for obj_name, obj_data in results.items():
            save_results[obj_name] = {}
            for key, value in obj_data.items():
                if key not in ['distance_stats']:  # Skip the large stats dictionary
                    # Convert numpy types to Python native types
                    if isinstance(value, (np.float32, np.float64)):
                        save_results[obj_name][key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        save_results[obj_name][key] = int(value)
                    elif isinstance(value, np.ndarray):
                        save_results[obj_name][key] = value.tolist()
                    else:
                        save_results[obj_name][key] = value
        
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Create visualizations
        self.create_comparison_plots(results)
        
        return results
        
    def create_comparison_plots(self, results):
        """Create comprehensive comparison visualizations"""
        # Extract data
        aurocs = []
        confusion_by_mean = []
        position_confusion = []
        density_gap_confusion = []
        confusion_combined = []
        mode_mean_diff = []
        mean_positions = []
        labels = []
        
        for obj, res in results.items():
            aurocs.append(res['auroc'])
            confusion_by_mean.append(res['confusion_by_mean'])
            position_confusion.append(res['position_confusion'])
            density_gap_confusion.append(res['density_gap_confusion'])
            confusion_combined.append(res['confusion_combined'])
            mode_mean_diff.append(res['mode_mean_diff'])
            mean_positions.append(res['mean_position'])
            labels.append(obj)
        
        # Convert to numpy arrays
        aurocs = np.array(aurocs)
        confusion_by_mean = np.array(confusion_by_mean)
        position_confusion = np.array(position_confusion)
        density_gap_confusion = np.array(density_gap_confusion)
        confusion_combined = np.array(confusion_combined)
        mode_mean_diff = np.array(mode_mean_diff)
        mean_positions = np.array(mean_positions)
        
        # 1. Main comparison plot - all methods
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Mean normalization
        ax = axes[0, 0]
        ax.scatter(confusion_by_mean, aurocs, alpha=0.6, s=100, color='green')
        corr_mean = np.corrcoef(confusion_by_mean, aurocs)[0, 1]
        z = np.polyfit(confusion_by_mean, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confusion_by_mean.min(), confusion_by_mean.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Confusion Score (by Mean)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Normalized by Mean\nCorrelation: {corr_mean:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Position confusion
        ax = axes[0, 1]
        ax.scatter(position_confusion, aurocs, alpha=0.6, s=100, color='orange')
        corr_pos = np.corrcoef(position_confusion, aurocs)[0, 1]
        z = np.polyfit(position_confusion, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(position_confusion.min(), position_confusion.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Position Confusion', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Mean Position\nCorrelation: {corr_pos:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Density gap confusion (now using histogram)
        ax = axes[1, 0]
        ax.scatter(density_gap_confusion, aurocs, alpha=0.6, s=100, color='purple')
        corr_density = np.corrcoef(density_gap_confusion, aurocs)[0, 1]
        z = np.polyfit(density_gap_confusion, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(density_gap_confusion.min(), density_gap_confusion.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Density Gap Confusion (Histogram)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Density Gap from Exact Histogram\nCorrelation: {corr_density:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Combined confusion score
        ax = axes[1, 1]
        ax.scatter(confusion_combined, aurocs, alpha=0.6, s=100, color='red')
        corr_combined = np.corrcoef(confusion_combined, aurocs)[0, 1]
        z = np.polyfit(confusion_combined, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confusion_combined.min(), confusion_combined.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Combined Confusion Score', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Combined (Mean + Position + Density Gap)\nCorrelation: {corr_combined:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Confusion Score Methods Comparison (Using Exact Histograms)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_methods_comparison.png', dpi=300)
        plt.close()
        
        # 2. Correlation strength comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['By Mean', 'Position', 'Density Gap\n(Histogram)', 'Combined']
        correlations = [corr_mean, corr_pos, corr_density, corr_combined]
        colors = ['green', 'orange', 'purple', 'red']
        
        bars = ax.bar(methods, correlations, alpha=0.7, color=colors)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Correlation with AUROC', fontsize=12)
        ax.set_title('Correlation Strength Comparison (Exact Histogram)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(min(correlations) - 0.1, 0)
        
        # Highlight best method
        best_idx = np.argmax(np.abs(correlations))
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_comparison.png', dpi=300)
        plt.close()
        
        # 3. Scatter matrix for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Components relationship
        scatter = axes[0, 0].scatter(confusion_by_mean, density_gap_confusion, 
                                    alpha=0.6, c=aurocs, cmap='viridis', s=100)
        axes[0, 0].set_xlabel('Mode-Mean Confusion (by Mean)')
        axes[0, 0].set_ylabel('Density Gap Confusion (Histogram)')
        axes[0, 0].set_title('Component Relationship (color = AUROC)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Mean position vs density gap
        scatter = axes[0, 1].scatter(position_confusion, density_gap_confusion, 
                                    alpha=0.6, c=aurocs, cmap='viridis', s=100)
        axes[0, 1].set_xlabel('Position Confusion')
        axes[0, 1].set_ylabel('Density Gap Confusion (Histogram)')
        axes[0, 1].set_title('Position vs Density Gap (color = AUROC)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Mean position distribution
        axes[1, 0].hist(mean_positions, bins=20, alpha=0.7, color='teal')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Center (0.5)')
        axes[1, 0].set_xlabel('Mean Position')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Mean Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Density gap distribution
        axes[1, 1].hist(density_gap_confusion, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Density Gap Confusion (Histogram)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Density Gap Values')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Detailed Component Analysis (Exact Histograms)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_component_analysis.png', dpi=300)
        plt.close()
        
        # 4. Generate comparison report
        self.generate_comparison_report(results, corr_mean, corr_pos, corr_density, corr_combined)
        
    def generate_comparison_report(self, results, corr_mean, corr_pos, corr_density, corr_combined):
        """Generate detailed comparison report"""
        report_lines = [
            "Confusion Score Normalization Comparison Report",
            "=" * 80,
            "",
            "Summary",
            "-" * 40,
            f"Objects analyzed: {len(results)}",
            f"Using: PyTorch {'GPU' if torch.cuda.is_available() else 'CPU'} for distance calculations",
            "",
            "Correlation with AUROC:",
            f"  - Normalized by Mean:         {corr_mean:.3f}",
            f"  - Position Confusion:         {corr_pos:.3f}",
            f"  - Density Gap Confusion:      {corr_density:.3f}",
            f"  - Combined Score:             {corr_combined:.3f}",
            "",
        ]
        
        # Find best method
        all_corrs = {
            'Normalized by Mean': abs(corr_mean),
            'Position Confusion': abs(corr_pos),
            'Density Gap': abs(corr_density),
            'Combined': abs(corr_combined)
        }
        best_method = max(all_corrs, key=all_corrs.get)
        best_corr = all_corrs[best_method]
        
        report_lines.extend([
            f"{'Best method:':<20} {best_method} (correlation: {best_corr:.3f})",
            "",
            "Method Descriptions:",
            "-" * 40,
            "1. Normalized by Mean: (mode - mean) / mean when mode > mean",
            "2. Position Confusion: How far mean is shifted right from center (0.5)",
            "3. Density Gap: 1 - (density_at_mean / density_at_mode) when mode > mean",
            "4. Combined: Equal average of all three metrics",
            "",
            "Detailed Results",
            "-" * 40,
            f"{'Object':<15} {'AUROC':<8} {'By Mean':<10} {'Position':<10} {'Density Gap':<12} {'Combined':<10}",
            "-" * 75
        ])
        
        # Sort by AUROC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
        
        for obj, res in sorted_results:
            report_lines.append(
                f"{obj:<15} {res['auroc']:<8.3f} {res['confusion_by_mean']:<10.3f} "
                f"{res['position_confusion']:<10.3f} {res['density_gap_confusion']:<12.3f} "
                f"{res['confusion_combined']:<10.3f}"
            )
        
        # Add statistical analysis
        report_lines.extend([
            "",
            "Statistical Analysis",
            "-" * 40
        ])
        
        # Calculate R-squared values
        r2_values = {
            'By Mean': corr_mean ** 2,
            'Position': corr_pos ** 2,
            'Density Gap': corr_density ** 2,
            'Combined': corr_combined ** 2
        }
        
        report_lines.extend([
            "R-squared (variance explained):",
        ])
        for method, r2 in r2_values.items():
            report_lines.append(f"  - {method:<20} {r2:.3f} ({r2*100:.1f}%)")
        
        # Mean position analysis
        mean_positions = [r['mean_position'] for r in results.values()]
        report_lines.extend([
            "",
            "Mean Position Analysis:",
            f"  - Average position: {np.mean(mean_positions):.3f} (ideal: 0.5)",
            f"  - Std deviation:    {np.std(mean_positions):.3f}",
            f"  - Range:            [{min(mean_positions):.3f}, {max(mean_positions):.3f}]",
            "",
            "Objects with mean position far from center (>0.65 or <0.35):"
        ])
        
        outlier_positions = [(obj, res['mean_position'], res['auroc']) 
                           for obj, res in results.items() 
                           if res['mean_position'] > 0.65 or res['mean_position'] < 0.35]
        
        if outlier_positions:
            for obj, pos, auroc in sorted(outlier_positions, key=lambda x: x[1]):
                report_lines.append(f"  - {obj}: position={pos:.3f}, AUROC={auroc:.3f}")
        else:
            report_lines.append("  - None found")
        
        # Density gap analysis
        density_gaps = [r['density_gap_confusion'] for r in results.values()]
        report_lines.extend([
            "",
            "Density Gap Analysis:",
            f"  - Average gap:      {np.mean(density_gaps):.3f}",
            f"  - Std deviation:    {np.std(density_gaps):.3f}",
            f"  - Range:            [{min(density_gaps):.3f}, {max(density_gaps):.3f}]",
            "",
            "Objects with high density gap (>0.7):"
        ])
        
        high_density_gaps = [(obj, res['density_gap_confusion'], res['auroc']) 
                            for obj, res in results.items() 
                            if res['density_gap_confusion'] > 0.7]
        
        if high_density_gaps:
            for obj, gap, auroc in sorted(high_density_gaps, key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {obj}: gap={gap:.3f}, AUROC={auroc:.3f}")
        else:
            report_lines.append("  - None found")
        
        # Component contribution analysis
        report_lines.extend([
            "",
            "Component Contribution Analysis:",
            "-" * 40
        ])
        
        # Analyze which component is most predictive
        component_importance = []
        
        # For each object, find which component has highest value
        for obj, res in results.items():
            components = {
                'mean': res['confusion_by_mean'],
                'position': res['position_confusion'],
                'density': res['density_gap_confusion']
            }
            dominant = max(components, key=components.get)
            component_importance.append(dominant)
        
        from collections import Counter
        component_counts = Counter(component_importance)
        
        report_lines.extend([
            "Dominant component per object:",
            f"  - Mode-Mean confusion: {component_counts['mean']} objects",
            f"  - Position confusion:  {component_counts['position']} objects",
            f"  - Density gap:         {component_counts['density']} objects",
            "",
            "Recommendation:",
            "-" * 40
        ])
        
        if best_method == 'Combined':
            report_lines.extend([
                f"Use the Combined confusion score for best overall performance.",
                "This metric captures all three aspects of distribution problems:",
                "  1. Mode-mean separation (normalized by mean)",
                "  2. Mean position shift from center",
                "  3. Density valley between mean and mode",
                "",
                "The combined approach is more robust than individual components."
            ])
        else:
            report_lines.extend([
                f"Use {best_method} confusion score for best AUROC prediction.",
                "However, consider using the Combined score for more robust results",
                "across different object types."
            ])
        
        # Add insights about problematic objects
        report_lines.extend([
            "",
            "Problematic Objects (Combined Score > 0.5):",
            "-" * 40
        ])
        
        problematic = [(obj, res['confusion_combined'], res['auroc'])
                        for obj, res in results.items()
                        if res['confusion_combined'] > 0.5]
        
        if problematic:
            for obj, score, auroc in sorted(problematic, key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {obj}: score={score:.3f}, AUROC={auroc:.3f}")
        else:
            report_lines.append("  - None found (all objects have good feature distributions)")
        
        # Save report
        with open(self.output_dir / 'normalization_comparison_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nReport saved to: {self.output_dir / 'normalization_comparison_report.txt'}")
        print(f"\nKey Finding: {best_method} shows strongest correlation ({best_corr:.3f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare confusion score normalization methods')
    parser.add_argument('--mvtec-root', type=str, default='mvtec',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--output-dir', type=str, default='normalization_comparison',
                        help='Output directory for results')
    parser.add_argument('--sample-ratio', type=float, default=0.01,
                        help='Sample ratio for PatchCore memory bank')
    
    args = parser.parse_args()
    
    if not Path(args.mvtec_root).exists():
        print(f"Error: MVTec directory not found at {args.mvtec_root}")
        return
    
    tester = NormalizationTester(
        args.mvtec_root,
        args.output_dir,
        sample_ratio=args.sample_ratio
    )
    
    results = tester.run_comparison()
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print(f"Results saved to: {args.output_dir}/")
    print("Check the visualizations and report for the best normalization method.")


if __name__ == "__main__":
    main()