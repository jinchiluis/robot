#!/usr/bin/env python3
"""MVTec AD Benchmark - Validation Data Analysis for Model Selection"""
import torch
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import shutil
import os
import random
from sklearn.metrics import roc_auc_score, silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import GPUtil
from collections import defaultdict

# Import both PatchCore implementations
from patchcore_exth import PatchCore as PatchCoreResNet
from patchcore_dino import PatchCore as PatchCoreDINO


class ValidationAnalysisBenchmark:
    def __init__(self, mvtec_root, output_dir="validation_analysis_results", categories=None, sample_ratio=0.01):
        self.mvtec_root = Path(mvtec_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set random seeds for reproducibility
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.categories = categories
        self.sample_ratio = sample_ratio

        self.results = {
            'system_info': self._get_system_info(),
            'benchmark_start': datetime.now().isoformat(),
            'objects': {}
        }

        # Temp directory for model checkpoints
        self.temp_dir = Path("temp_models")
        self.temp_dir.mkdir(exist_ok=True)
        
    def _get_system_info(self):
        """Collect system information"""
        info = {
            'python_version': os.sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_name'] = gpus[0].name
                info['gpu_memory_mb'] = gpus[0].memoryTotal
        
        return info
    
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
    
    def split_training_data(self, train_dir, split_ratio=0.99):
        """Split training data into train and validation sets"""
        train_path = Path(train_dir)
        
        # Get all image files
        image_files = set()
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.update(train_path.glob(ext))
            image_files.update(train_path.glob(ext.upper()))
        
        image_files = list(image_files)
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create temporary directories
        temp_train = self.temp_dir / "train_split" / "good"
        temp_val = self.temp_dir / "val_split" / "good"
        
        if temp_train.parent.parent.exists():
            shutil.rmtree(temp_train.parent.parent)
        
        temp_train.mkdir(parents=True)
        temp_val.mkdir(parents=True)
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, temp_train / f.name)
        for f in val_files:
            shutil.copy2(f, temp_val / f.name)
        
        return str(temp_train), str(temp_val), len(train_files), len(val_files)
    
    def calculate_distribution_metrics(self, scores):
        """Calculate comprehensive distribution metrics"""
        scores = np.array(scores)
        
        metrics = {}
        
        # Basic statistics
        metrics['mean'] = float(np.mean(scores))
        metrics['median'] = float(np.median(scores))
        metrics['std'] = float(np.std(scores))
        metrics['min'] = float(np.min(scores))
        metrics['max'] = float(np.max(scores))
        metrics['range'] = float(metrics['max'] - metrics['min'])
        
        # Coefficient of variation
        metrics['cv'] = float(metrics['std'] / metrics['mean']) if metrics['mean'] > 0 else 0
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'p{p}'] = float(np.percentile(scores, p))
        
        # IQR and MAD
        metrics['iqr'] = float(metrics['p75'] - metrics['p25'])
        metrics['mad'] = float(np.median(np.abs(scores - metrics['median'])))
        
        # Shape metrics
        metrics['skewness'] = float(stats.skew(scores))
        metrics['kurtosis'] = float(stats.kurtosis(scores))
        
        # Bimodality coefficient
        # BC = (skewness^2 + 1) / (kurtosis + 3)
        # BC > 0.555 suggests bimodality
        bc = (metrics['skewness']**2 + 1) / (metrics['kurtosis'] + 3)
        metrics['bimodality_coefficient'] = float(bc)
        
        # Gap analysis
        sorted_scores = np.sort(scores)
        gaps = np.diff(sorted_scores)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            metrics['largest_gap'] = float(gaps[max_gap_idx])
            metrics['gap_location'] = float((sorted_scores[max_gap_idx] + sorted_scores[max_gap_idx + 1]) / 2)
            metrics['gap_relative_position'] = float(max_gap_idx / len(gaps))
        else:
            metrics['largest_gap'] = 0
            metrics['gap_location'] = 0
            metrics['gap_relative_position'] = 0
        
        # Entropy (binned)
        hist, _ = np.histogram(scores, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        metrics['entropy'] = float(-np.sum(hist * np.log(hist)))
        
        # Clustering metrics
        if len(scores) > 10:
            try:
                # Try to find 2 clusters
                scores_reshaped = scores.reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scores_reshaped)
                
                # Only calculate silhouette if we have both clusters
                if len(np.unique(labels)) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(scores_reshaped, labels))
                    metrics['calinski_harabasz'] = float(calinski_harabasz_score(scores_reshaped, labels))
                    
                    # Cluster separation
                    cluster_centers = kmeans.cluster_centers_.flatten()
                    metrics['cluster_separation'] = float(abs(cluster_centers[1] - cluster_centers[0]))
                    
                    # Cluster balance
                    cluster_sizes = np.bincount(labels)
                    metrics['cluster_balance'] = float(min(cluster_sizes) / max(cluster_sizes))
                else:
                    metrics['silhouette_score'] = 0
                    metrics['calinski_harabasz'] = 0
                    metrics['cluster_separation'] = 0
                    metrics['cluster_balance'] = 1
            except:
                metrics['silhouette_score'] = 0
                metrics['calinski_harabasz'] = 0
                metrics['cluster_separation'] = 0
                metrics['cluster_balance'] = 1
        else:
            metrics['silhouette_score'] = 0
            metrics['calinski_harabasz'] = 0
            metrics['cluster_separation'] = 0
            metrics['cluster_balance'] = 1
        
        # Outlier ratio (using IQR method)
        q1, q3 = metrics['p25'], metrics['p75']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((scores < lower_bound) | (scores > upper_bound))
        metrics['outlier_ratio'] = float(outliers / len(scores))
        
        # Percentile stability (how much percentiles change in upper range)
        metrics['percentile_stability'] = float(metrics['p95'] - metrics['p90'])
        
        # Dynamic range utilization
        theoretical_range = 1.0  # Assuming scores are normalized to [0, 1]
        metrics['range_utilization'] = float(metrics['range'] / theoretical_range)
        
        return metrics
    
    def train_model_with_validation_tracking(self, model_type, object_name):
        """Train model and collect validation scores"""
        print(f"\n{'='*60}")
        print(f"Training {model_type} on {object_name}")
        print(f"{'='*60}")
        
        # Prepare paths
        train_dir = self.mvtec_root / object_name / "train" / "good"
        train_split_dir, val_split_dir, n_train, n_val = self.split_training_data(train_dir)
        
        print(f"Split: {n_train} train, {n_val} validation images")
        
        # Initialize model
        if model_type == "ResNet":
            model = PatchCoreResNet(backbone='wide_resnet50_2')
        else:  # DINOv2
            model = PatchCoreDINO(backbone='dinov2_vits14')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train model
        model.fit(
            train_dir=train_split_dir,
            val_dir=val_split_dir,
            sample_ratio=self.sample_ratio,
            threshold_percentile=95
        )
        
        # Collect validation scores
        val_scores = []
        val_path = Path(val_split_dir)
        
        for img_path in val_path.glob('*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    result = model.predict(str(img_path), return_heatmap=False)
                    val_scores.append(result['anomaly_score'])
                except:
                    continue
        
        # Calculate distribution metrics
        distribution_metrics = self.calculate_distribution_metrics(val_scores)
        
        # Clean up split directories
        shutil.rmtree(Path(train_split_dir).parent)
        
        results = {
            'n_train_images': n_train,
            'n_val_images': n_val,
            'validation_scores': val_scores,
            'distribution_metrics': distribution_metrics,
            'threshold': float(model.global_threshold)
        }
        
        return model, results
    
    def evaluate_model(self, model, test_dir):
        """Evaluate model and return AUROC"""
        test_path = Path(test_dir)
        
        all_scores = []
        all_labels = []
        
        # Process each category
        for category_dir in test_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # Get all images in category
            for img_path in category_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    try:
                        result = model.predict(str(img_path), return_heatmap=False)
                        all_scores.append(result['anomaly_score'])
                        all_labels.append(0 if category_name == 'good' else 1)
                    except:
                        continue
        
        # Calculate AUROC
        if len(set(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_scores)
        else:
            auroc = None
        
        return auroc
    
    def run_benchmark(self):
        """Run benchmark focused on validation analysis"""
        objects = self.discover_objects()
        
        for obj_idx, object_name in enumerate(objects):
            print(f"\n{'#'*80}")
            print(f"Processing object {obj_idx+1}/{len(objects)}: {object_name}")
            print(f"{'#'*80}")
            
            obj_results = {}
            test_dir = self.mvtec_root / object_name / "test"
            
            # Train and evaluate ResNet
            try:
                resnet_model, resnet_train_results = self.train_model_with_validation_tracking("ResNet", object_name)
                resnet_auroc = self.evaluate_model(resnet_model, test_dir)
                
                obj_results['resnet'] = {
                    'auroc': resnet_auroc,
                    'validation_analysis': resnet_train_results
                }
                
                print(f"ResNet AUROC: {resnet_auroc:.3f}")
                
                del resnet_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with ResNet on {object_name}: {e}")
                obj_results['resnet'] = {'error': str(e)}
            
            # Train and evaluate DINOv2
            try:
                dino_model, dino_train_results = self.train_model_with_validation_tracking("DINOv2", object_name)
                dino_auroc = self.evaluate_model(dino_model, test_dir)
                
                obj_results['dinov2'] = {
                    'auroc': dino_auroc,
                    'validation_analysis': dino_train_results
                }
                
                print(f"DINOv2 AUROC: {dino_auroc:.3f}")
                
                del dino_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with DINOv2 on {object_name}: {e}")
                obj_results['dinov2'] = {'error': str(e)}
            
            # Determine winner
            if 'error' not in obj_results['resnet'] and 'error' not in obj_results['dinov2']:
                resnet_auroc = obj_results['resnet']['auroc']
                dino_auroc = obj_results['dinov2']['auroc']
                
                if resnet_auroc and dino_auroc:
                    obj_results['winner'] = 'resnet' if resnet_auroc > dino_auroc else 'dinov2'
                    obj_results['auroc_difference'] = abs(resnet_auroc - dino_auroc)
                    
                    print(f"Winner: {obj_results['winner'].upper()} (diff: {obj_results['auroc_difference']:.3f})")
            
            self.results['objects'][object_name] = obj_results
            self._save_results()
        
        self.results['benchmark_end'] = datetime.now().isoformat()
        self._save_results()
        self.generate_analysis_report()
        
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(self.temp_dir)
    
    def _save_results(self):
        """Save results to JSON"""
        results_file = self.output_dir / "validation_analysis_results.json"
        
        # Create a copy without the raw validation scores for saving
        results_copy = json.loads(json.dumps(self.results))
        
        # Remove raw validation scores to reduce file size
        for obj_name, obj_data in results_copy['objects'].items():
            for model in ['resnet', 'dinov2']:
                if model in obj_data and 'validation_analysis' in obj_data[model]:
                    if 'validation_scores' in obj_data[model]['validation_analysis']:
                        # Just keep the count
                        n_scores = len(self.results['objects'][obj_name][model]['validation_analysis']['validation_scores'])
                        results_copy['objects'][obj_name][model]['validation_analysis']['n_validation_scores'] = n_scores
                        del results_copy['objects'][obj_name][model]['validation_analysis']['validation_scores']
        
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
    
    def generate_analysis_report(self):
        """Generate analysis report and visualizations"""
        print("\nGenerating validation analysis report...")
        
        # Collect all data for analysis
        analysis_data = self._collect_analysis_data()
        
        if not analysis_data:
            print("No valid data for analysis")
            return
        
        # Generate visualizations
        self._generate_distribution_plots(analysis_data)
        self._generate_correlation_analysis(analysis_data)
        self._generate_decision_analysis(analysis_data)
        
        # Generate text report
        self._generate_text_report(analysis_data)
    
    def _collect_analysis_data(self):
        """Collect and organize data for analysis"""
        data = []
        
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
            
            # Add distribution metrics for each model
            for model in ['resnet', 'dinov2']:
                if model in obj_results and 'validation_analysis' in obj_results[model]:
                    metrics = obj_results[model]['validation_analysis']['distribution_metrics']
                    for metric_name, value in metrics.items():
                        row[f'{model}_{metric_name}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _generate_distribution_plots(self, df):
        """Generate distribution visualization plots"""
        # Create figure with subplots for each object
        n_objects = len(df)
        n_cols = 3
        n_rows = (n_objects + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_objects > 1 else [axes]
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            obj_name = row['object']
            winner = row['winner']
            
            # Get validation scores for both models
            resnet_scores = self.results['objects'][obj_name]['resnet']['validation_analysis']['validation_scores']
            dino_scores = self.results['objects'][obj_name]['dinov2']['validation_analysis']['validation_scores']
            
            # Plot distributions
            ax.hist(resnet_scores, bins=30, alpha=0.5, label='ResNet', color='blue', density=True)
            ax.hist(dino_scores, bins=30, alpha=0.5, label='DINOv2', color='orange', density=True)
            
            # Add KDE
            from scipy.stats import gaussian_kde
            if len(resnet_scores) > 1:
                kde_resnet = gaussian_kde(resnet_scores)
                x_range = np.linspace(min(resnet_scores), max(resnet_scores), 100)
                ax.plot(x_range, kde_resnet(x_range), 'b-', linewidth=2)
            
            if len(dino_scores) > 1:
                kde_dino = gaussian_kde(dino_scores)
                x_range = np.linspace(min(dino_scores), max(dino_scores), 100)
                ax.plot(x_range, kde_dino(x_range), 'r-', linewidth=2)
            
            # Mark thresholds
            resnet_threshold = self.results['objects'][obj_name]['resnet']['validation_analysis']['threshold']
            dino_threshold = self.results['objects'][obj_name]['dinov2']['validation_analysis']['threshold']
            
            ax.axvline(resnet_threshold, color='blue', linestyle='--', alpha=0.7, label='ResNet threshold')
            ax.axvline(dino_threshold, color='orange', linestyle='--', alpha=0.7, label='DINOv2 threshold')
            
            # Title with winner info
            winner_color = 'blue' if winner == 'resnet' else 'orange'
            ax.set_title(f'{obj_name}\nWinner: {winner.upper()}', color=winner_color, fontweight='bold')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(df), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'validation_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_correlation_analysis(self, df):
        """Generate correlation analysis between metrics and model performance"""
        # Calculate which metrics differ most between winners
        resnet_wins = df[df['winner'] == 'resnet']
        dino_wins = df[df['winner'] == 'dinov2']
        
        # Get all metric columns
        metric_columns = [col for col in df.columns if ('resnet_' in col or 'dinov2_' in col) and '_auroc' not in col]
        
        # Calculate feature importance (difference in means between winners)
        feature_importance = []
        
        for col in metric_columns:
            if len(resnet_wins) > 0 and len(dino_wins) > 0:
                resnet_mean = resnet_wins[col].mean()
                dino_mean = dino_wins[col].mean()
                importance = abs(resnet_mean - dino_mean)
                
                # Also calculate which direction favors which model
                favors_model = 'resnet' if resnet_mean > dino_mean else 'dinov2'
                
                feature_importance.append({
                    'metric': col,
                    'importance': importance,
                    'resnet_mean': resnet_mean,
                    'dino_mean': dino_mean,
                    'favors': favors_model
                })
        
        # Sort by importance
        feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
        
        # Plot top 20 most important features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Feature importance plot
        top_features = feature_importance[:20]
        metrics = [f['metric'] for f in top_features]
        importances = [f['importance'] for f in top_features]
        colors = ['blue' if f['favors'] == 'resnet' else 'orange' for f in top_features]
        
        ax1.barh(range(len(metrics)), importances, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(metrics)))
        ax1.set_yticklabels(metrics)
        ax1.set_xlabel('Mean Difference Between Winners')
        ax1.set_title('Most Discriminative Validation Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Higher values favor ResNet'),
            Patch(facecolor='orange', alpha=0.7, label='Higher values favor DINOv2')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Scatter plot of top 2 features
        if len(top_features) >= 2:
            feat1, feat2 = top_features[0]['metric'], top_features[1]['metric']
            
            for model, color in [('resnet', 'blue'), ('dinov2', 'orange')]:
                model_data = df[df['winner'] == model]
                ax2.scatter(model_data[feat1], model_data[feat2], 
                           color=color, alpha=0.6, s=100, label=f'{model.upper()} wins')
            
            ax2.set_xlabel(feat1)
            ax2.set_ylabel(feat2)
            ax2.set_title('Top 2 Discriminative Features')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_importance
    
    def _generate_decision_analysis(self, df):
        """Generate decision boundary visualizations"""
        # Create multiple scatter plots for different metric pairs
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Define metric pairs to visualize
        metric_pairs = [
            ('skewness', 'kurtosis'),
            ('cv', 'entropy'),
            ('silhouette_score', 'cluster_separation'),
            ('bimodality_coefficient', 'outlier_ratio'),
            ('percentile_stability', 'range_utilization'),
            ('p95', 'iqr')
        ]
        
        for idx, (metric1, metric2) in enumerate(metric_pairs):
            ax = axes[idx]
            
            # Create full metric names
            resnet_metric1 = f'resnet_{metric1}'
            resnet_metric2 = f'resnet_{metric2}'
            dino_metric1 = f'dinov2_{metric1}'
            dino_metric2 = f'dinov2_{metric2}'
            
            # Check if metrics exist
            if all(col in df.columns for col in [resnet_metric1, resnet_metric2, dino_metric1, dino_metric2]):
                # Calculate difference metrics
                df[f'diff_{metric1}'] = df[resnet_metric1] - df[dino_metric1]
                df[f'diff_{metric2}'] = df[resnet_metric2] - df[dino_metric2]
                
                # Plot
                for winner, color, marker in [('resnet', 'blue', 'o'), ('dinov2', 'orange', 's')]:
                    winner_data = df[df['winner'] == winner]
                    ax.scatter(winner_data[f'diff_{metric1}'], winner_data[f'diff_{metric2}'],
                             color=color, alpha=0.6, s=100, marker=marker, label=f'{winner.upper()} wins')
                
                ax.set_xlabel(f'ResNet - DINOv2: {metric1}')
                ax.set_ylabel(f'ResNet - DINOv2: {metric2}')
                ax.set_title(f'{metric1} vs {metric2} Differences')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_boundaries.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, df):
        """Generate comprehensive text report"""
        feature_importance = self._generate_correlation_analysis(df)
        
        report_lines = [
            "Validation Analysis Report for Model Selection",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary Statistics",
            "-" * 40,
            f"Total objects analyzed: {len(df)}",
            f"ResNet wins: {len(df[df['winner'] == 'resnet'])}",
            f"DINOv2 wins: {len(df[df['winner'] == 'dinov2'])}",
            f"Average AUROC difference: {df['auroc_difference'].mean():.3f}",
            "",
            "Object-wise Results",
            "-" * 40
        ]
        
        # Add per-object results
        for _, row in df.iterrows():
            report_lines.append(
                f"{row['object']}: Winner={row['winner'].upper()}, "
                f"ResNet AUROC={row['resnet_auroc']:.3f}, "
                f"DINOv2 AUROC={row['dinov2_auroc']:.3f}, "
                f"Difference={row['auroc_difference']:.3f}"
            )
        
        # Add key insights
        report_lines.extend([
            "",
            "Key Insights from Validation Data",
            "-" * 40,
            "Top 10 Most Discriminative Metrics:"
        ])
        
        for i, feat in enumerate(feature_importance[:10]):
            direction = "higher" if feat['favors'] == 'resnet' else "lower"
            report_lines.append(
                f"{i+1}. {feat['metric']}: {direction} values favor ResNet "
                f"(importance={feat['importance']:.3f})"
            )
        
        # Add decision rules
        report_lines.extend([
            "",
            "Discovered Decision Rules",
            "-" * 40
        ])
        
        # Simple decision rules based on top features
        rules = self._extract_decision_rules(df, feature_importance)
        for rule in rules:
            report_lines.append(rule)
        
        # Add detailed statistics
        report_lines.extend([
            "",
            "Detailed Validation Metrics Comparison",
            "-" * 40
        ])
        
        # Create comparison table for key metrics
        key_metrics = ['skewness', 'kurtosis', 'cv', 'entropy', 'silhouette_score', 
                      'bimodality_coefficient', 'percentile_stability']
        
        for metric in key_metrics:
            resnet_col = f'resnet_{metric}'
            dino_col = f'dinov2_{metric}'
            
            if resnet_col in df.columns and dino_col in df.columns:
                resnet_wins_metric = df[df['winner'] == 'resnet'][resnet_col].mean()
                dino_wins_metric = df[df['winner'] == 'dinov2'][dino_col].mean()
                
                report_lines.append(
                    f"{metric}: ResNet wins avg={resnet_wins_metric:.3f}, "
                    f"DINOv2 wins avg={dino_wins_metric:.3f}"
                )
        
        # Save report
        report_file = self.output_dir / "validation_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nReport saved to: {report_file}")
    
    def _extract_decision_rules(self, df, feature_importance):
        """Extract simple decision rules from the data"""
        rules = []
        
        # Rule 1: Based on skewness
        if 'resnet_skewness' in df.columns and 'dinov2_skewness' in df.columns:
            resnet_wins = df[df['winner'] == 'resnet']
            dino_wins = df[df['winner'] == 'dinov2']
            
            if len(resnet_wins) > 0 and len(dino_wins) > 0:
                # Calculate average difference in skewness
                resnet_skew_diff = (resnet_wins['resnet_skewness'] - resnet_wins['dinov2_skewness']).mean()
                dino_skew_diff = (dino_wins['resnet_skewness'] - dino_wins['dinov2_skewness']).mean()
                
                if resnet_skew_diff > 0 and dino_skew_diff < 0:
                    rules.append(
                        "Rule 1: If ResNet validation skewness > DINOv2 validation skewness, "
                        "prefer ResNet (holds in {:.0f}% of cases)".format(
                            100 * len(resnet_wins[resnet_wins['resnet_skewness'] > resnet_wins['dinov2_skewness']]) / len(resnet_wins)
                        )
                    )
        
        # Rule 2: Based on coefficient of variation
        if 'resnet_cv' in df.columns and 'dinov2_cv' in df.columns:
            resnet_high_cv = df[(df['winner'] == 'resnet') & (df['resnet_cv'] > df['resnet_cv'].median())]
            dino_high_cv = df[(df['winner'] == 'dinov2') & (df['dinov2_cv'] > df['dinov2_cv'].median())]
            
            if len(resnet_high_cv) > len(dino_high_cv):
                rules.append(
                    "Rule 2: High coefficient of variation (> {:.3f}) in validation scores "
                    "favors ResNet".format(df['resnet_cv'].median())
                )
        
        # Rule 3: Based on clustering metrics
        if 'resnet_silhouette_score' in df.columns:
            high_silhouette = df[df['resnet_silhouette_score'] > 0.3]
            if len(high_silhouette) > 0:
                resnet_ratio = len(high_silhouette[high_silhouette['winner'] == 'resnet']) / len(high_silhouette)
                if resnet_ratio > 0.7:
                    rules.append(
                        "Rule 3: High silhouette score (> 0.3) in validation data "
                        "indicates ResNet advantage ({:.0f}% accuracy)".format(100 * resnet_ratio)
                    )
        
        # Rule 4: Based on entropy
        if 'resnet_entropy' in df.columns and 'dinov2_entropy' in df.columns:
            entropy_diff = df['resnet_entropy'] - df['dinov2_entropy']
            high_entropy_diff = df[entropy_diff > entropy_diff.median()]
            
            if len(high_entropy_diff) > 0:
                dino_ratio = len(high_entropy_diff[high_entropy_diff['winner'] == 'dinov2']) / len(high_entropy_diff)
                if dino_ratio > 0.6:
                    rules.append(
                        "Rule 4: When ResNet entropy > DINOv2 entropy, "
                        "consider DINOv2 ({:.0f}% accuracy)".format(100 * dino_ratio)
                    )
        
        # Rule 5: Composite rule
        if all(col in df.columns for col in ['resnet_cv', 'dinov2_cv', 'resnet_skewness', 'dinov2_skewness']):
            # High CV difference AND positive skewness
            df['cv_diff'] = df['resnet_cv'] - df['dinov2_cv']
            df['skew_diff'] = df['resnet_skewness'] - df['dinov2_skewness']
            
            composite = df[(df['cv_diff'] > 0.1) & (df['skew_diff'] > 0.5)]
            if len(composite) > 0:
                resnet_accuracy = len(composite[composite['winner'] == 'resnet']) / len(composite)
                if resnet_accuracy > 0.8:
                    rules.append(
                        "Rule 5 (Composite): If (ResNet CV - DINOv2 CV > 0.1) AND "
                        "(ResNet skewness - DINOv2 skewness > 0.5), "
                        "strongly prefer ResNet ({:.0f}% accuracy)".format(100 * resnet_accuracy)
                    )
        
        if not rules:
            rules.append("No strong decision rules found - models perform similarly across objects")
        
        return rules


def main():
    """Run the validation analysis benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MVTec AD Validation Analysis Benchmark')
    parser.add_argument('--mvtec-root', type=str, default='mvtec',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--output-dir', type=str, default='validation_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--category', type=str, nargs='*', default=None,
                        help='Subset of categories to benchmark (e.g. --category leather bottle)')
    parser.add_argument('--sample-ratio', type=float, default=0.01,
                        help='Sample ratio for PatchCore fit (e.g. 0.01 for 1%)')
    
    args = parser.parse_args()
    
    # Check if MVTec directory exists
    if not Path(args.mvtec_root).exists():
        print(f"Error: MVTec directory not found at {args.mvtec_root}")
        return
    
    # Run benchmark
    benchmark = ValidationAnalysisBenchmark(
        args.mvtec_root, 
        args.output_dir, 
        categories=args.category,
        sample_ratio=args.sample_ratio
    )
    benchmark.run_benchmark()
    
    print("\nValidation analysis completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()