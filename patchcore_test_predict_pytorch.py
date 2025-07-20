#!/usr/bin/env python3
"""Compare different normalization methods for confusion score - Optimized with Pytorch"""
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from patchcore_dino import PatchCore as PatchCoreDINO


def find_highest_peak(distances, bins=50):
    """Find the mode (highest peak) in distance distribution"""
    hist, bin_edges = np.histogram(distances, bins=bins)
    max_bin_idx = np.argmax(hist)
    mode_value = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    return mode_value


def compute_pairwise_distances_torch_gpu(memory_bank):
    """Compute distance statistics using PyTorch on GPU - no k limitations!"""
    n_features = len(memory_bank)
    
    # Convert to torch tensor
    if torch.cuda.is_available():
        memory_torch = torch.from_numpy(memory_bank).float().cuda()
        device = 'cuda'
    else:
        memory_torch = torch.from_numpy(memory_bank).float()
        device = 'cpu'
    
    print(f"  Computing distance statistics for {n_features} points using PyTorch {device.upper()}...")
    
    # Initialize statistics
    distance_samples = []  # For histogram computation
    distance_sum = 0.0
    distance_sum_sq = 0.0
    n_distances = 0
    min_dist = float('inf')
    max_dist = float('-inf')
    
    # Batch size based on available memory
    if device == 'cuda':
        # Estimate safe batch size based on memory bank size and GPU memory
        # Rough estimate: each distance matrix element takes 4 bytes (float32)
        # We need batch_size × n_features × 4 bytes
        # Use conservative estimate to avoid OOM
        if n_features > 30000:
            batch_size = 5000
        elif n_features > 20000:
            batch_size = 8000
        else:
            batch_size = 10000
    else:
        batch_size = 2000  # Smaller for CPU
    
    # Process in batches
    total_batches = (n_features + batch_size - 1) // batch_size
    with tqdm(total=total_batches * (total_batches + 1) // 2, desc="Distance computation") as pbar:
        for i in range(0, n_features, batch_size):
            batch1 = memory_torch[i:min(i+batch_size, n_features)]
            
            for j in range(i, n_features, batch_size):  # Start from i to avoid duplicates
                batch2 = memory_torch[j:min(j+batch_size, n_features)]
                
                # Compute batch1 × batch2 distances
                distances = torch.cdist(batch1, batch2)
                
                # If same batch, only take upper triangular (excluding diagonal)
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
                    
                    # Keep samples for histogram
                    if len(distance_samples) < 1_000_000:
                        sample_size = min(1000, len(batch_distances), 1_000_000 - len(distance_samples))
                        if sample_size > 0:
                            # Take evenly spaced samples
                            indices = torch.linspace(0, len(batch_distances)-1, sample_size, dtype=torch.long, device=device)
                            sampled = batch_distances[indices].cpu().numpy()
                            distance_samples.extend(sampled)
                
                pbar.update(1)
    
    # Clear GPU memory
    if device == 'cuda':
        del memory_torch
        torch.cuda.empty_cache()
    
    # Calculate final statistics
    mean_dist = distance_sum / n_distances
    variance = (distance_sum_sq / n_distances) - (mean_dist ** 2)
    std_dist = np.sqrt(variance)
    
    # Convert samples to array
    distance_samples = np.array(distance_samples)
    
    print(f"  Processed {n_distances:,} distances using PyTorch {device.upper()}")
    print(f"  Mean: {mean_dist:.4f}, Std: {std_dist:.4f}, Min: {min_dist:.4f}, Max: {max_dist:.4f}")
    print(f"  Histogram sample size: {len(distance_samples):,}")
    
    return distance_samples, mean_dist, std_dist, min_dist, max_dist

def compute_confusion_scores(distance_samples, mean_dist, std_dist, min_dist, max_dist):
    """Compute confusion scores with different normalizations"""
    # Find mode from samples
    mode_dist = find_highest_peak(distance_samples)
    
    # Raw difference
    mode_mean_diff = mode_dist - mean_dist
    
    # Method 1: Normalize by standard deviation
    if std_dist > 0:
        confusion_by_std = mode_mean_diff / std_dist
    else:
        confusion_by_std = 0
    
    # Method 2: Normalize by mean
    if mean_dist > 0:
        confusion_by_mean = mode_mean_diff / mean_dist
    else:
        confusion_by_mean = 0
    
    # Only positive values (mode > mean indicates confusion)
    confusion_by_std = max(0, confusion_by_std)
    confusion_by_mean = max(0, confusion_by_mean)
    
    # Calculate mean position in distribution
    range_dist = max_dist - min_dist
    
    # Mean position (0 = left edge, 0.5 = center, 1 = right edge)
    if range_dist > 0:
        mean_position = (mean_dist - min_dist) / range_dist
    else:
        mean_position = 0.5
    
    # Position confusion: how far mean is from center (0.5)
    # Higher score when mean is shifted right
    position_confusion = max(0, mean_position - 0.5) * 2  # Scale to 0-1
    
    # Combined scores (50/50 weighting)
    confusion_std_with_position = 0.5 * confusion_by_std + 0.5 * position_confusion
    confusion_mean_with_position = 0.5 * confusion_by_mean + 0.5 * position_confusion
    
    return {
        'mode': mode_dist,
        'mean': mean_dist,
        'std': std_dist,
        'mode_mean_diff': mode_mean_diff,
        'confusion_by_std': confusion_by_std,
        'confusion_by_mean': confusion_by_mean,
        'mean_position': mean_position,
        'position_confusion': position_confusion,
        'confusion_std_with_position': confusion_std_with_position,
        'confusion_mean_with_position': confusion_mean_with_position
    }


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

    def plot_distance_distribution(self,distance_samples, scores, object_name, output_dir):
        """Plot distance distribution with Gaussian fit - isolated visualization function"""
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(distance_samples, bins=100, density=True, alpha=0.6, 
                                    color='blue', edgecolor='black', linewidth=0.5)
        
        # Fit and plot Gaussian
        mean, std = scores['mean'], scores['std']
        x = np.linspace(distance_samples.min(), distance_samples.max(), 1000)
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
        plt.title(f'{object_name} - NN Distance Distribution\n' + 
                f'Confusion Score (STD): {scores["confusion_by_std"]:.3f}, ' +
                f'AUROC: {scores.get("auroc", "N/A")}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        print(f"Saving distance distribution plot for {object_name}...")
        plt.savefig(output_dir / f'{object_name}_distance_distribution.png', dpi=150)
        plt.close()
        
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
        
        # Use PyTorch GPU for fast distance computation - NO SAMPLING, NO LIMITS!
        distance_samples, mean_dist, std_dist, min_dist, max_dist = compute_pairwise_distances_torch_gpu(memory_bank)
        
        # Compute all confusion scores
        scores = compute_confusion_scores(distance_samples, mean_dist, std_dist, min_dist, max_dist)
        print("Start plot")
        self.plot_distance_distribution(distance_samples, scores, object_name, self.output_dir)
        # Evaluate AUROC
        test_dir = self.mvtec_root / object_name / "test"
        auroc = self.evaluate_model(model, test_dir)
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'auroc': auroc,
            'distance_samples': distance_samples,  # Keep samples for visualization
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
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
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
                print(f"    confusion_by_std: {obj_results['confusion_by_std']:.4f}")
                print(f"    confusion_by_mean: {obj_results['confusion_by_mean']:.4f}")
                print(f"    mean_position: {obj_results['mean_position']:.4f}")
                print(f"    position_confusion: {obj_results['position_confusion']:.4f}")
                print(f"    confusion_std_with_position: {obj_results['confusion_std_with_position']:.4f}")
                print(f"    confusion_mean_with_position: {obj_results['confusion_mean_with_position']:.4f}")
        
        # Save raw results - convert numpy types to Python native types
        save_results = {}
        for obj_name, obj_data in results.items():
            save_results[obj_name] = {}
            for key, value in obj_data.items():
                if key not in ['distance_samples']:  # Skip the large distance array
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
        confusion_by_std = []
        confusion_by_mean = []
        position_confusion = []
        confusion_std_with_position = []
        confusion_mean_with_position = []
        mode_mean_diff = []
        mean_positions = []
        labels = []
        
        for obj, res in results.items():
            aurocs.append(res['auroc'])
            confusion_by_std.append(res['confusion_by_std'])
            confusion_by_mean.append(res['confusion_by_mean'])
            position_confusion.append(res['position_confusion'])
            confusion_std_with_position.append(res['confusion_std_with_position'])
            confusion_mean_with_position.append(res['confusion_mean_with_position'])
            mode_mean_diff.append(res['mode_mean_diff'])
            mean_positions.append(res['mean_position'])
            labels.append(obj)
        
        # Convert to numpy arrays
        aurocs = np.array(aurocs)
        confusion_by_std = np.array(confusion_by_std)
        confusion_by_mean = np.array(confusion_by_mean)
        position_confusion = np.array(position_confusion)
        confusion_std_with_position = np.array(confusion_std_with_position)
        confusion_mean_with_position = np.array(confusion_mean_with_position)
        mode_mean_diff = np.array(mode_mean_diff)
        mean_positions = np.array(mean_positions)
        
        # 1. Main comparison plot - all methods
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Original STD normalization
        ax = axes[0, 0]
        ax.scatter(confusion_by_std, aurocs, alpha=0.6, s=100, color='blue')
        corr_std = np.corrcoef(confusion_by_std, aurocs)[0, 1]
        z = np.polyfit(confusion_by_std, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confusion_by_std.min(), confusion_by_std.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Confusion Score (by STD)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Normalized by STD\nCorrelation: {corr_std:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Original Mean normalization
        ax = axes[0, 1]
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
        
        # Plot 3: Position confusion alone
        ax = axes[0, 2]
        ax.scatter(position_confusion, aurocs, alpha=0.6, s=100, color='orange')
        corr_pos = np.corrcoef(position_confusion, aurocs)[0, 1]
        z = np.polyfit(position_confusion, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(position_confusion.min(), position_confusion.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Position Confusion', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Mean Position Only\nCorrelation: {corr_pos:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: STD + Position (50/50)
        ax = axes[1, 0]
        ax.scatter(confusion_std_with_position, aurocs, alpha=0.6, s=100, color='purple')
        corr_std_pos = np.corrcoef(confusion_std_with_position, aurocs)[0, 1]
        z = np.polyfit(confusion_std_with_position, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confusion_std_with_position.min(), confusion_std_with_position.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Confusion Score (STD + Position)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'STD + Position (50/50)\nCorrelation: {corr_std_pos:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Mean + Position (50/50)
        ax = axes[1, 1]
        ax.scatter(confusion_mean_with_position, aurocs, alpha=0.6, s=100, color='brown')
        corr_mean_pos = np.corrcoef(confusion_mean_with_position, aurocs)[0, 1]
        z = np.polyfit(confusion_mean_with_position, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confusion_mean_with_position.min(), confusion_mean_with_position.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel('Confusion Score (Mean + Position)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'Mean + Position (50/50)\nCorrelation: {corr_mean_pos:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Mean position visualization
        ax = axes[1, 2]
        ax.scatter(mean_positions, aurocs, alpha=0.6, s=100, color='teal')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Center')
        ax.set_xlabel('Mean Position in Distribution', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('Mean Position vs Performance', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.suptitle('Confusion Score Methods Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_methods_comparison.png', dpi=300)
        plt.close()
        
        # 2. Correlation strength comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['By STD', 'By Mean', 'Position Only', 'STD + Position', 'Mean + Position']
        correlations = [corr_std, corr_mean, corr_pos, corr_std_pos, corr_mean_pos]
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        bars = ax.bar(methods, correlations, alpha=0.7, color=colors)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Correlation with AUROC', fontsize=12)
        ax.set_title('Correlation Strength Comparison - All Methods', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(min(correlations) - 0.1, 0)
        
        # Highlight best method
        best_idx = np.argmax(np.abs(correlations))
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_comparison_all_methods.png', dpi=300)
        plt.close()
        
        # 3. Scatter matrix for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Components breakdown
        axes[0, 0].scatter(confusion_by_std, position_confusion, alpha=0.6, c=aurocs, cmap='viridis')
        axes[0, 0].set_xlabel('Mode-Mean Confusion (by STD)')
        axes[0, 0].set_ylabel('Position Confusion')
        axes[0, 0].set_title('Component Relationship (color = AUROC)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Best two methods comparison
        axes[0, 1].scatter(confusion_mean_with_position, confusion_std_with_position, alpha=0.6)
        axes[0, 1].set_xlabel('Mean + Position')
        axes[0, 1].set_ylabel('STD + Position')
        axes[0, 1].set_title('Combined Methods Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        # Add diagonal line
        lims = [
            np.min([axes[0, 1].get_xlim(), axes[0, 1].get_ylim()]),
            np.max([axes[0, 1].get_xlim(), axes[0, 1].get_ylim()]),
        ]
        axes[0, 1].plot(lims, lims, 'k--', alpha=0.5)
        
        # Mean position distribution
        axes[1, 0].hist(mean_positions, bins=20, alpha=0.7, color='teal')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Center (0.5)')
        axes[1, 0].set_xlabel('Mean Position')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Mean Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Position confusion vs mode-mean difference
        axes[1, 1].scatter(mode_mean_diff, position_confusion, alpha=0.6, c=aurocs, cmap='viridis')
        axes[1, 1].set_xlabel('Mode - Mean (raw difference)')
        axes[1, 1].set_ylabel('Position Confusion')
        axes[1, 1].set_title('Raw Difference vs Position (color = AUROC)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Detailed Component Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_component_analysis.png', dpi=300)
        plt.close()
        
        # 4. Generate comparison report
        self.generate_comparison_report(results, corr_std, corr_mean, corr_pos, corr_std_pos, corr_mean_pos)
    
    def generate_comparison_report(self, results, corr_std, corr_mean, corr_pos, corr_std_pos, corr_mean_pos):
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
            f"  - Normalized by STD:          {corr_std:.3f}",
            f"  - Normalized by Mean:         {corr_mean:.3f}",
            f"  - Position Confusion Only:    {corr_pos:.3f}",
            f"  - STD + Position (50/50):     {corr_std_pos:.3f}",
            f"  - Mean + Position (50/50):    {corr_mean_pos:.3f}",
            "",
        ]
        
        # Find best method
        all_corrs = {
            'Normalized by STD': abs(corr_std),
            'Normalized by Mean': abs(corr_mean),
            'Position Only': abs(corr_pos),
            'STD + Position': abs(corr_std_pos),
            'Mean + Position': abs(corr_mean_pos)
        }
        best_method = max(all_corrs, key=all_corrs.get)
        best_corr = all_corrs[best_method]
        
        report_lines.extend([
            f"{'Best method:':<20} {best_method} (correlation: {best_corr:.3f})",
            "",
            "Detailed Results",
            "-" * 40,
            f"{'Object':<15} {'AUROC':<8} {'By STD':<10} {'By Mean':<10} {'Position':<10} {'STD+Pos':<10} {'Mean+Pos':<10}",
            "-" * 80
        ])
        
        # Sort by AUROC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
        
        for obj, res in sorted_results:
            report_lines.append(
                f"{obj:<15} {res['auroc']:<8.3f} {res['confusion_by_std']:<10.3f} "
                f"{res['confusion_by_mean']:<10.3f} {res['position_confusion']:<10.3f} "
                f"{res['confusion_std_with_position']:<10.3f} {res['confusion_mean_with_position']:<10.3f}"
            )
        
        # Add statistical analysis
        report_lines.extend([
            "",
            "Statistical Analysis",
            "-" * 40
        ])
        
        # Calculate R-squared values
        aurocs = [r['auroc'] for r in results.values()]
        
        r2_values = {
            'By STD': corr_std ** 2,
            'By Mean': corr_mean ** 2,
            'Position Only': corr_pos ** 2,
            'STD + Position': corr_std_pos ** 2,
            'Mean + Position': corr_mean_pos ** 2
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
        
        # Component contribution analysis
        report_lines.extend([
            "",
            "Component Contribution Analysis:",
            "-" * 40
        ])
        
        # Check if position improves predictions
        improvement_std = abs(corr_std_pos) - abs(corr_std)
        improvement_mean = abs(corr_mean_pos) - abs(corr_mean)
        
        report_lines.extend([
            f"Adding position confusion to STD:  {'+' if improvement_std > 0 else ''}{improvement_std:.3f} correlation change",
            f"Adding position confusion to Mean: {'+' if improvement_mean > 0 else ''}{improvement_mean:.3f} correlation change",
            "",
            "Recommendation:",
            "-" * 40
        ])
        
        if best_method in ['STD + Position', 'Mean + Position']:
            report_lines.append(
                f"Use {best_method} confusion score, as the mean position significantly "
                "improves prediction accuracy."
            )
        else:
            report_lines.append(
                f"Use {best_method} confusion score for best AUROC prediction."
            )
        
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
    