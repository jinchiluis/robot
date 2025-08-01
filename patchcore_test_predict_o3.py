#!/usr/bin/env python3
"""Compare different normalization methods for confusion score - Optimized with Pytorch
   + Robust tail/multimodality confusion metrics (MOR, tail stretch, right mass, optional peaks)
"""
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


# ================================================================
# Distance computation (unverändert)
# ================================================================
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
    
    total_batches = (n_features + batch_size - 1) // batch_size
    with tqdm(total=total_batches * (total_batches + 1) // 2, desc="First pass") as pbar:
        for i in range(0, n_features, batch_size):
            batch1 = memory_torch[i:min(i+batch_size, n_features)]
            for j in range(i, n_features, batch_size):
                batch2 = memory_torch[j:min(j+batch_size, n_features)]
                distances = torch.cdist(batch1, batch2)
                if i == j:
                    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                    batch_distances = distances[mask]
                else:
                    batch_distances = distances.flatten()
                if len(batch_distances) > 0:
                    distance_sum += batch_distances.sum().item()
                    distance_sum_sq += (batch_distances ** 2).sum().item()
                    n_distances += len(batch_distances)
                    batch_min = batch_distances.min().item()
                    batch_max = batch_distances.max().item()
                    min_dist = min(min_dist, batch_min)
                    max_dist = max(max_dist, batch_max)
                pbar.update(1)
    
    mean_dist = distance_sum / n_distances
    variance = (distance_sum_sq / n_distances) - (mean_dist ** 2)
    std_dist = np.sqrt(max(variance, 0.0))
    
    print(f"  Range: [{min_dist:.4f}, {max_dist:.4f}]")
    print(f"  Mean: {mean_dist:.4f}, Std: {std_dist:.4f}")
    
    # ========== SECOND PASS: Build exact histogram ==========
    print(f"  Second pass: Building histogram with {n_bins} bins...")
    eps = (max_dist - min_dist) * 1e-6
    bin_edges = np.linspace(min_dist, max_dist + eps, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_edges_torch = torch.from_numpy(bin_edges).to(device)
    
    with tqdm(total=total_batches * (total_batches + 1) // 2, desc="Second pass") as pbar:
        for i in range(0, n_features, batch_size):
            batch1 = memory_torch[i:min(i+batch_size, n_features)]
            for j in range(i, n_features, batch_size):
                batch2 = memory_torch[j:min(j+batch_size, n_features)]
                distances = torch.cdist(batch1, batch2)
                if i == j:
                    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
                    batch_distances = distances[mask]
                else:
                    batch_distances = distances.flatten()
                if len(batch_distances) > 0:
                    bin_indices = torch.searchsorted(bin_edges_torch[:-1], batch_distances, right=False)
                    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
                    # Vectorisierte Zählung statt Schleife:
                    binc = torch.bincount(bin_indices, minlength=n_bins).cpu().numpy()
                    bin_counts += binc
                pbar.update(1)
    
    if device == 'cuda':
        del memory_torch
        torch.cuda.empty_cache()
    
    print(f"  Processed {n_distances:,} distances")
    print(f"  Built exact histogram with {n_bins} bins")
    
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


# ================================================================
# Alte (mode-basierte) Kennzahlen – behalten falls du Vergleich brauchst
# ================================================================
def compute_confusion_scores_mode_based(distance_stats):
    """Original mode/mean basierte Heuristik (rename, um beide Versionen nutzen zu können)."""
    mean_dist = distance_stats['mean']
    std_dist = distance_stats['std']
    min_dist = distance_stats['min']
    max_dist = distance_stats['max']
    mode_dist = distance_stats['mode']
    histogram = distance_stats['histogram']
    bin_counts = histogram['bin_counts']
    
    mode_mean_diff = mode_dist - mean_dist
    confusion_by_mean = (mode_mean_diff / mean_dist) if mean_dist > 0 else 0
    confusion_by_mean = max(0, confusion_by_mean)
    range_dist = max_dist - min_dist
    mean_position = (mean_dist - min_dist) / range_dist if range_dist > 0 else 0.5
    position_confusion = max(0, mean_position - 0.5) * 4
    density_gap_confusion = 0
    if mode_dist > mean_dist:
        mean_bin_idx = np.searchsorted(histogram['bin_edges'][:-1], mean_dist, side='right') - 1
        mean_bin_idx = np.clip(mean_bin_idx, 0, len(bin_counts) - 1)
        mode_bin_idx = np.argmax(bin_counts)
        density_at_mean = bin_counts[mean_bin_idx]
        density_at_mode = bin_counts[mode_bin_idx]
        if density_at_mode > 0:
            density_gap_confusion = 1 - (density_at_mean / density_at_mode)
        density_gap_confusion = max(0, density_gap_confusion)
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

def compute_confusion_scores(distance_stats):
    """
    Original mode/mean basierte Scores (unverändert) + einfache rechte-Masse-Heuristik
    jetzt basierend auf linkem Kern (ref_mean/ref_std) statt globalem Mean.
    """
    mean_dist = distance_stats['mean']
    std_dist  = distance_stats['std']
    min_dist  = distance_stats['min']
    max_dist  = distance_stats['max']
    mode_dist = distance_stats['mode']
    
    histogram   = distance_stats['histogram']
    bin_edges   = histogram['bin_edges']
    bin_counts  = histogram['bin_counts']
    bin_centers = histogram['bin_centers']
    
    # ==================== (A) ALTE LOGIK (unverändert) ====================
    mode_mean_diff = mode_dist - mean_dist
    if mean_dist > 0:
        confusion_by_mean = mode_mean_diff / mean_dist
    else:
        confusion_by_mean = 0.0
    confusion_by_mean = max(0.0, confusion_by_mean)
    
    range_dist = max_dist - min_dist
    if range_dist > 0:
        mean_position = (mean_dist - min_dist) / range_dist
    else:
        mean_position = 0.5
    position_confusion = max(0.0, mean_position - 0.5) * 4  # 0..2
    
    density_gap_confusion = 0.0
    if mode_dist > mean_dist:
        mean_bin_idx = np.searchsorted(bin_edges[:-1], mean_dist, side='right') - 1
        mean_bin_idx = np.clip(mean_bin_idx, 0, len(bin_counts) - 1)
        mode_bin_idx = np.argmax(bin_counts)
        density_at_mean = bin_counts[mean_bin_idx]
        density_at_mode = bin_counts[mode_bin_idx]
        if density_at_mode > 0:
            density_gap_confusion = 1 - (density_at_mean / density_at_mode)
        density_gap_confusion = max(0.0, density_gap_confusion)
    
    confusion_combined = (confusion_by_mean + position_confusion + density_gap_confusion) / 3.0
    
    # ==================== (B) NEUE REFERENZ: linker Kern (<= Median) ====================
    # Rekonstruiere (sub-)Sample für Kernabschätzung via Histogramm:
    # Wir approximieren ref_mean/ref_std durch gewichtetes Mittel/Varianz aller Bins,
    # deren Bin-Mitte <= Median (Median schätzen wir aus Histogramm).
    
    # Median-Schätzung aus Histogramm:
    total = bin_counts.sum()
    if total == 0:
        # Degenerierter Fall
        simple_shift = simple_shift_norm = 0.0
        simple_right_mass = simple_excess_right_mass = 0.0
    else:
        cdf = np.cumsum(bin_counts)
        median_target = 0.5 * total
        med_bin_idx = np.searchsorted(cdf, median_target, side='left')
        med_bin_idx = min(med_bin_idx, len(bin_counts)-1)
        # Interpolation innerhalb des Median-Bins (optional – hier reicht Bin-Zentrum)
        median_est = (bin_edges[med_bin_idx] + bin_edges[med_bin_idx+1]) / 2.0
        
        # Linker Kern: alle Bins mit Zentrum <= median_est
        core_mask = bin_centers <= median_est
        core_counts = bin_counts[core_mask]
        core_centers = bin_centers[core_mask]
        core_total = core_counts.sum()
        if core_total <= 0:
            ref_mean = mean_dist
            ref_std  = std_dist if std_dist > 0 else 1.0
        else:
            ref_mean = (core_centers * core_counts).sum() / core_total
            # Varianz des Kerns
            var_core = (core_counts * (core_centers - ref_mean)**2).sum() / core_total
            ref_std = np.sqrt(var_core)
            if ref_std == 0:
                ref_std = std_dist if std_dist > 0 else 1.0
        
        # ==================== (C) EINFACHE LOGIK mit ref_mean / ref_std ====================
        # 1. Shift relativ zu ref_mean/ref_std
        if ref_std > 0:
            simple_shift = max(0.0, (mode_dist - ref_mean) / ref_std)
        else:
            simple_shift = 0.0
        simple_shift_norm = simple_shift / (1.0 + simple_shift)
        
        # 2. Right mass gegen ref_mean (nicht globalen Mean)
        ref_mean_bin_idx = np.searchsorted(bin_edges, ref_mean, side='right') - 1
        ref_mean_bin_idx = np.clip(ref_mean_bin_idx, 0, len(bin_counts)-1)
        simple_right_mass = bin_counts[ref_mean_bin_idx+1:].sum() / total
        
        # 3. Excess (Basis 0.5)
        simple_excess_right_mass = max(0.0, simple_right_mass - 0.5) / 0.5
        
    # Kombination (Gewichte anpassbar – aktuell gleich)
    simple_confusion_score = 0.5 * simple_shift_norm + 0.5 * simple_excess_right_mass
    
    return {
        # Alte Keys
        'mode': mode_dist,
        'mean': mean_dist,
        'std': std_dist,
        'mode_mean_diff': mode_mean_diff,
        'confusion_by_mean': confusion_by_mean,
        'mean_position': mean_position,
        'position_confusion': position_confusion,
        'density_gap_confusion': density_gap_confusion,
        'confusion_combined': confusion_combined,
        # Neue / angepasste einfachen Keys
        'simple_shift': simple_shift,
        'simple_shift_norm': simple_shift_norm,
        'simple_right_mass': simple_right_mass,
        'simple_excess_right_mass': simple_excess_right_mass,
        'simple_confusion_score': simple_confusion_score,
        # Referenzwerte zur Kontrolle
        'ref_mean': ref_mean if total > 0 else mean_dist,
        'ref_std': ref_std if total > 0 else std_dist
    }


# ================================================================
# Plot: erweitert um robuste Kennzahlen
# ================================================================
def plot_distance_distribution(distance_stats, scores, object_name, output_dir, auroc=None):
    """Plot exact distance distribution from histogram data + simple confusion score."""
    plt.figure(figsize=(10, 6))
    
    # Histogram data
    histogram   = distance_stats['histogram']
    bin_centers = histogram['bin_centers']
    bin_counts  = histogram['bin_counts']
    bin_edges   = histogram['bin_edges']
    n_distances = distance_stats['n_distances']
    
    bin_width = bin_edges[1] - bin_edges[0]
    densities = bin_counts / (n_distances * bin_width)
    
    # Histogram
    plt.bar(bin_centers, densities, width=bin_width, alpha=0.55,
            color='steelblue', edgecolor='black', linewidth=0.4,
            label='Exact distribution')
    
    # Gaussian fit
    mean_ = scores['mean']
    std_  = scores['std']
    x = np.linspace(distance_stats['min'], distance_stats['max'], 1000)
    if std_ > 0:
        gaussian = (1/(std_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_)/std_)**2)
        plt.plot(x, gaussian, 'r-', linewidth=1.8,
                 label=f'Gaussian (μ={mean_:.3f}, σ={std_:.3f})')
    
    # Mean & Mode lines
    plt.axvline(mean_, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_:.3f}')
    plt.axvline(scores['mode'], color='green', linestyle='--', linewidth=1.5,
                label=f"Mode: {scores['mode']:.3f}")
    
    # Simple confusion region (Mean -> Mode) falls Mode rechts
    if scores['mode'] > mean_:
        plt.axvspan(mean_, scores['mode'], alpha=0.20, color='orange',
                    label='Mode > Mean region')
    
    # Textbox mit Kennzahlen (rechts oben)
    auroc_str = f'{auroc:.3f}' if auroc is not None else 'N/A'
    simple_score = scores.get('simple_confusion_score', None)
    if simple_score is not None:
        text_lines = [
            f"AUROC: {auroc_str}",
            f"SIMPLE Score: {simple_score:.3f}",
            f"  shift_norm: {scores['simple_shift_norm']:.3f}",
            f"  right_mass: {scores['simple_right_mass']:.3f}",
            f"ModeScore(old): {scores['confusion_combined']:.3f}"
        ]
    else:
        text_lines = [
            f"AUROC: {auroc_str}",
            f"ModeScore(old): {scores['confusion_combined']:.3f}"
        ]
    
    textbox = "\n".join(text_lines)
    plt.gca().text(0.97, 0.97, textbox,
                   ha='right', va='top',
                   transform=plt.gca().transAxes,
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5))
    
    plt.title(f'{object_name} - NN Distance Distribution', fontsize=14)
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
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
        
        # Compute robust confusion score only
        scores = compute_confusion_scores(distance_stats)
        robust_confusion = scores.get('robust_confusion', None)

        # Evaluate AUROC
        test_dir = self.mvtec_root / object_name / "test"
        auroc = self.evaluate_model(model, test_dir)

        # Plot distribution (pass only robust_confusion)
        plot_distance_distribution(distance_stats, scores, object_name, self.output_dir, auroc)

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return all new simple_* keys as well
        result = {
            'auroc': auroc,
            'distance_stats': distance_stats,  # Keep full stats for analysis
            'robust_confusion': robust_confusion
        }
        # Add new simple_* keys if present
        for k in ['simple_shift', 'simple_shift_norm', 'simple_right_mass', 'simple_excess_right_mass', 'simple_confusion_score']:
            if k in scores:
                result[k] = scores[k]
        return result
    
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
        print("Using exact histogram approach (no sampling) for accurate mode and density calculations")
        
        for obj in tqdm(objects, desc="Analyzing objects"):
            obj_results = self.analyze_object(obj)
            if obj_results['auroc'] is not None:
                results[obj] = obj_results
                print(f"  {obj}: AUROC={obj_results['auroc']:.3f}")
                # Print robust confusion stats
                distance_stats = obj_results['distance_stats']
                robust_scores = compute_confusion_scores(distance_stats)
                print("  Robust Confusion Details:")
                print(f"    simple_confusion: {robust_scores.get('simple_confusion_score', None):.4f}")

                # Print new simple_* keys
                for k in ['simple_shift', 'simple_shift_norm', 'simple_right_mass', 'simple_excess_right_mass', 'simple_confusion_score']:
                    if k in robust_scores:
                        print(f"    {k}: {robust_scores[k]:.4f}")
                # Print all requested values as a dict after each run
                print({
                    'simple_shift': robust_scores.get('simple_shift', None),
                    'simple_shift_norm': robust_scores.get('simple_shift_norm', None),
                    'simple_right_mass': robust_scores.get('simple_right_mass', None),
                    'simple_excess_right_mass': robust_scores.get('simple_excess_right_mass', None),
                    'simple_confusion_score': robust_scores.get('simple_confusion_score', None)
                })
        
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
        """Create visualizations for robust confusion only"""
        # Extract data
        aurocs = []
        robust_confusions = []
        labels = []

        for obj, res in results.items():
            aurocs.append(res['auroc'])
            robust_confusions.append(res['robust_confusion'])
            labels.append(obj)

        aurocs = np.array(aurocs)
        robust_confusions = np.array(robust_confusions)

        # Main plot: robust_confusion vs AUROC
        plt.figure(figsize=(10, 7))
        plt.scatter(robust_confusions, aurocs, alpha=0.7, s=120, color='blue')
        corr_robust = np.corrcoef(robust_confusions, aurocs)[0, 1]
        z = np.polyfit(robust_confusions, aurocs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(robust_confusions.min(), robust_confusions.max(), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8)
        plt.xlabel('Robust Confusion Score', fontsize=13)
        plt.ylabel('AUROC', fontsize=13)
        plt.title(f'Robust Confusion vs AUROC\nCorrelation: {corr_robust:.3f}', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robust_confusion_vs_auroc.png', dpi=300)
        plt.close()

        # Generate report
        self.generate_comparison_report(results, corr_robust)
        
    def generate_comparison_report(self, results, corr_robust):
        """Generate report for robust confusion only"""
        report_lines = [
            "Robust Confusion Score Comparison Report",
            "=" * 80,
            "",
            "Summary",
            "-" * 40,
            f"Objects analyzed: {len(results)}",
            f"Using: PyTorch {'GPU' if torch.cuda.is_available() else 'CPU'} for distance calculations",
            "",
            f"Correlation with AUROC: {corr_robust:.3f}",
            "",
            "Method Description:",
            "-" * 40,
            "Robust confusion score combines Bowley skew, tail stretch, right mass shift, mean-median shift, and right peak bonus.",
            "",
            "Detailed Results",
            "-" * 40,
            f"{'Object':<15} {'AUROC':<8} {'RobustConf':<12}",
            "-" * 40
        ]

        # Sort by AUROC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
        for obj, res in sorted_results:
            report_lines.append(f"{obj:<15} {res['auroc']:<8.3f} {res['robust_confusion']:<12.3f}")

        # Statistical analysis
        robust_confusions = [r['robust_confusion'] for r in results.values()]
        report_lines.extend([
            "",
            "Statistical Analysis",
            "-" * 40,
            f"  - Average robust confusion: {np.mean(robust_confusions):.3f}",
            f"  - Std deviation:           {np.std(robust_confusions):.3f}",
            f"  - Range:                   [{min(robust_confusions):.3f}, {max(robust_confusions):.3f}]",
            "",
            "Objects with high robust confusion (>0.5):"
        ])
        high_robust = [(obj, res['robust_confusion'], res['auroc'])
                       for obj, res in results.items()
                       if res['robust_confusion'] > 0.5]
        if high_robust:
            for obj, score, auroc in sorted(high_robust, key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {obj}: score={score:.3f}, AUROC={auroc:.3f}")
        else:
            report_lines.append("  - None found (all objects have good feature distributions)")

        # Save report
        with open(self.output_dir / 'normalization_comparison_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\nReport saved to: {self.output_dir / 'normalization_comparison_report.txt'}")
        print(f"\nKey Finding: Robust confusion shows correlation ({corr_robust:.3f})")


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