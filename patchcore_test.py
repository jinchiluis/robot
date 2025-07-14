#!/usr/bin/env python3
"""MVTec AD Benchmark - Compare PatchCore ResNet vs DINOv2"""
import torch
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import shutil
import os
import random
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import GPUtil

# Import both PatchCore implementations
from patchcore_exth import PatchCore as PatchCoreResNet
from patchcore_dino import PatchCore as PatchCoreDINO


class MVTecBenchmark:
    def __init__(self, mvtec_root, output_dir="benchmark_results", categories=None, sample_ratio=0.01):
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
        """Find all object folders in MVTec dataset, optionally filter by categories"""
        objects = []
        for item in self.mvtec_root.iterdir():
            if item.is_dir():
                train_path = item / "train" / "good"
                test_path = item / "test"
                if train_path.exists() and test_path.exists():
                    objects.append(item.name)

        if self.categories:
            # Filter to only requested categories
            objects = [obj for obj in objects if obj in self.categories]

        print(f"Found {len(objects)} objects: {', '.join(sorted(objects))}")
        return sorted(objects)
    
    def split_training_data(self, train_dir, split_ratio=0.5):
        """Split training data into train and validation sets"""
        train_path = Path(train_dir)
        
        # Get all image files (case-insensitive but avoiding duplicates)
        image_files = set()  # Use set to avoid duplicates
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            # Add both lowercase and uppercase versions
            image_files.update(train_path.glob(ext))
            image_files.update(train_path.glob(ext.upper()))
        
        # Convert back to list for shuffling
        image_files = list(image_files)
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create temporary directories - maintain the good subdirectory structure
        temp_train = self.temp_dir / "train_split" / "good"
        temp_val = self.temp_dir / "val_split" / "good"
        
        # Clean up if exists
        if temp_train.parent.parent.exists():
            shutil.rmtree(temp_train.parent.parent)
        
        temp_train.mkdir(parents=True)
        temp_val.mkdir(parents=True)
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, temp_train / f.name)
        for f in val_files:
            shutil.copy2(f, temp_val / f.name)
        
        # Return direct paths to the directories containing images
        # The SimpleImageDataset expects to find images directly in the provided directory
        return str(temp_train), str(temp_val), len(train_files), len(val_files)
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0
    
    def find_optimal_threshold(self, scores, labels):
        """Find threshold that maximizes F1 score"""
        # Get unique thresholds to try
        thresholds = np.unique(scores)
        
        # If too many thresholds, sample them
        if len(thresholds) > 1000:
            thresholds = np.percentile(thresholds, np.linspace(0, 100, 1000))
        
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            
            # Skip if all predictions are the same
            if len(np.unique(predictions)) == 1:
                continue
                
            f1 = f1_score(labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
                # Calculate additional metrics at this threshold
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                tn = np.sum((predictions == 0) & (labels == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                best_metrics = {
                    'threshold': float(best_threshold),
                    'f1_score': float(best_f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'accuracy': float((tp + tn) / len(labels)),
                    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
                }
        
        return best_metrics
    
    def train_model(self, model_type, object_name, sample_ratio=0.01):
        """Train a model and measure performance"""
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
            model = PatchCoreDINO(backbone='dinov2_vitb14')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure training
        mem_before = self.get_memory_usage()
        start_time = time.time()
        
        # Train with fixed parameters
        model.fit(
            train_dir=train_split_dir,
            val_dir=val_split_dir,
            sample_ratio=sample_ratio,
            threshold_percentile=95
        )
        
        train_time = time.time() - start_time
        mem_after = self.get_memory_usage()
        peak_memory = mem_after - mem_before
        
        # Save model
        model_path = self.temp_dir / f"{object_name}_{model_type.lower()}.pth"
        model.save(str(model_path))
        model_size_mb = model_path.stat().st_size / (1024**2)
        
        # Clean up split directories
        shutil.rmtree(Path(train_split_dir).parent)
        
        results = {
            'train_time': train_time,
            'n_train_images': n_train,
            'n_val_images': n_val,
            'peak_memory_gb': peak_memory,
            'model_size_mb': model_size_mb,
            'threshold': float(model.global_threshold),
            'memory_bank_shape': model.memory_bank.shape
        }
        
        print(f"Training completed in {train_time:.2f}s")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Threshold: {model.global_threshold:.6f}")
        
        return model, results
    
    def evaluate_model(self, model, test_dir, object_name, model_type):
        """Evaluate model on test data"""
        test_path = Path(test_dir)
        
        results = {
            'categories': {},
            'overall': {
                'total_images': 0,
                'inference_times': [],
                'all_scores': [],
                'all_labels': [],
                'model_threshold': float(model.global_threshold),  # Store the 95-percentile threshold
            }
        }
        
        # Process each category
        for category_dir in sorted(test_path.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            category_results = {
                'scores': [],
                'labels': [],
                'inference_times': [],
                'n_images': 0
            }
            
            # Get all images in category
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(list(category_dir.glob(ext)))
            
            category_results['n_images'] = len(image_files)
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"Testing {category_name}"):
                # Clear cache for consistent timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                try:
                    result = model.predict(str(img_path), return_heatmap=False)
                    score = result['anomaly_score']
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
                
                inference_time = time.time() - start_time
                
                # Store results
                category_results['scores'].append(score)
                category_results['labels'].append(0 if category_name == 'good' else 1)
                category_results['inference_times'].append(inference_time)
                
                results['overall']['all_scores'].append(score)
                results['overall']['all_labels'].append(0 if category_name == 'good' else 1)
                results['overall']['inference_times'].append(inference_time)
            
            # Calculate category metrics if there are defects
            if category_name != 'good' and len(category_results['scores']) > 0:
                # Get good scores for AUROC calculation
                good_category = test_path / 'good'
                good_scores = []
                
                for img_path in good_category.glob('*.png'):
                    try:
                        result = model.predict(str(img_path), return_heatmap=False)
                        good_scores.append(result['anomaly_score'])
                    except:
                        continue
                
                if good_scores:
                    all_scores = good_scores + category_results['scores']
                    all_labels = [0] * len(good_scores) + [1] * len(category_results['scores'])
                    
                    try:
                        auroc = roc_auc_score(all_labels, all_scores)
                        category_results['auroc'] = auroc
                    except:
                        category_results['auroc'] = None
            
            category_results['avg_inference_time'] = np.mean(category_results['inference_times'])
            results['categories'][category_name] = category_results
            results['overall']['total_images'] += category_results['n_images']
        
        # Calculate overall AUROC
        if len(set(results['overall']['all_labels'])) > 1:
            results['overall']['auroc'] = roc_auc_score(
                results['overall']['all_labels'], 
                results['overall']['all_scores']
            )
            
            # Add F1-optimal threshold analysis
            f1_metrics = self.find_optimal_threshold(
                np.array(results['overall']['all_scores']),
                np.array(results['overall']['all_labels'])
            )
            results['overall']['f1_optimal'] = f1_metrics
            
            # Also evaluate at the model's 95-percentile threshold
            predictions_at_model_threshold = (
                np.array(results['overall']['all_scores']) >= model.global_threshold
            ).astype(int)
            
            results['overall']['model_threshold_metrics'] = {
                'threshold': float(model.global_threshold),
                'f1_score': float(f1_score(
                    results['overall']['all_labels'], 
                    predictions_at_model_threshold
                )),
                'precision': float(precision_score(
                    results['overall']['all_labels'], 
                    predictions_at_model_threshold
                )),
                'recall': float(recall_score(
                    results['overall']['all_labels'], 
                    predictions_at_model_threshold
                )),
                'accuracy': float(accuracy_score(
                    results['overall']['all_labels'], 
                    predictions_at_model_threshold
                ))
            }
        else:
            results['overall']['auroc'] = None
        
        results['overall']['avg_inference_time'] = np.mean(results['overall']['inference_times'])
        
        return results
    
    def run_benchmark(self, model=None):
        """Run complete benchmark"""
        objects = self.discover_objects()
        
        for obj_idx, object_name in enumerate(objects):
            print(f"\n{'#'*80}")
            print(f"Processing object {obj_idx+1}/{len(objects)}: {object_name}")
            print(f"{'#'*80}")
            
            obj_results = {
                'resnet': {},
                'dinov2': {}
            }
            
            test_dir = self.mvtec_root / object_name / "test"
            
            # Train and evaluate ResNet
            if model is None or model == 'wideresnet':
                try:
                    resnet_model, resnet_train_results = self.train_model("ResNet", object_name, sample_ratio=self.sample_ratio)
                    obj_results['resnet']['training'] = resnet_train_results
                    
                    print(f"\nEvaluating ResNet on {object_name} test set...")
                    resnet_eval_results = self.evaluate_model(resnet_model, test_dir, object_name, "ResNet")
                    obj_results['resnet']['evaluation'] = resnet_eval_results
                    
                    # Clean up
                    del resnet_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error with ResNet on {object_name}: {e}")
                    obj_results['resnet']['error'] = str(e)
            
            # Train and evaluate DINOv2
            if model is None or model == 'dinov2':
                try:
                    dino_model, dino_train_results = self.train_model("DINOv2", object_name, sample_ratio=self.sample_ratio)
                    obj_results['dinov2']['training'] = dino_train_results
                    
                    print(f"\nEvaluating DINOv2 on {object_name} test set...")
                    dino_eval_results = self.evaluate_model(dino_model, test_dir, object_name, "DINOv2")
                    obj_results['dinov2']['evaluation'] = dino_eval_results
                    
                    # Clean up
                    del dino_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error with DINOv2 on {object_name}: {e}")
                    obj_results['dinov2']['error'] = str(e)
            
            self.results['objects'][object_name] = obj_results
            
            # Save intermediate results
            self._save_results()
        
        self.results['benchmark_end'] = datetime.now().isoformat()
        self._save_results()
        self.generate_report()
        
        # Cleanup temporary files
        print("\nCleaning up temporary files...")
        shutil.rmtree(self.temp_dir)
    
    def _save_results(self):
        """Save results to JSON"""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\nGenerating benchmark report...")
        
        # Create report text
        report_lines = [
            "MVTec AD Benchmark Report",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Random seed: {self.seed}",
            "",
            "System Information:",
            "-" * 40
        ]
        
        for key, value in self.results['system_info'].items():
            report_lines.append(f"{key}: {value}")
        
        report_lines.extend(["", "=" * 80, ""])
        
        # Summary table data
        summary_data = []
        
        for object_name, obj_data in self.results['objects'].items():
            row = {'Object': object_name}
            
            # ResNet results
            if 'resnet' in obj_data and 'error' not in obj_data['resnet']:
                row['ResNet_Train_Time'] = f"{obj_data['resnet']['training']['train_time']:.1f}s"
                row['ResNet_Model_Size'] = f"{obj_data['resnet']['training']['model_size_mb']:.1f}MB"
                row['ResNet_AUROC'] = f"{obj_data['resnet']['evaluation']['overall'].get('auroc', 0):.3f}"
                row['ResNet_Inference'] = f"{obj_data['resnet']['evaluation']['overall']['avg_inference_time']*1000:.1f}ms"
            else:
                row['ResNet_Train_Time'] = "Error"
                row['ResNet_Model_Size'] = "Error"
                row['ResNet_AUROC'] = "Error"
                row['ResNet_Inference'] = "Error"
            
            # DINOv2 results
            if 'dinov2' in obj_data and 'error' not in obj_data['dinov2']:
                row['DINOv2_Train_Time'] = f"{obj_data['dinov2']['training']['train_time']:.1f}s"
                row['DINOv2_Model_Size'] = f"{obj_data['dinov2']['training']['model_size_mb']:.1f}MB"
                row['DINOv2_AUROC'] = f"{obj_data['dinov2']['evaluation']['overall'].get('auroc', 0):.3f}"
                row['DINOv2_Inference'] = f"{obj_data['dinov2']['evaluation']['overall']['avg_inference_time']*1000:.1f}ms"
            else:
                row['DINOv2_Train_Time'] = "Error"
                row['DINOv2_Model_Size'] = "Error"
                row['DINOv2_AUROC'] = "Error"
                row['DINOv2_Inference'] = "Error"
            
            summary_data.append(row)
        
        # Create DataFrame for easy formatting
        df = pd.DataFrame(summary_data)
        
        # Add summary to report
        report_lines.append("Overall Performance Summary:")
        report_lines.append("-" * 80)
        report_lines.append(df.to_string(index=False))
        
        # Calculate averages
        report_lines.extend(["", "=" * 80, "", "Average Performance Metrics:", "-" * 40])
        
        # Calculate numeric averages
        resnet_aurocs = []
        dino_aurocs = []
        resnet_times = []
        dino_times = []
        
        for obj_data in self.results['objects'].values():
            if 'resnet' in obj_data and 'error' not in obj_data['resnet']:
                if obj_data['resnet']['evaluation']['overall'].get('auroc'):
                    resnet_aurocs.append(obj_data['resnet']['evaluation']['overall']['auroc'])
                resnet_times.append(obj_data['resnet']['evaluation']['overall']['avg_inference_time'])
            
            if 'dinov2' in obj_data and 'error' not in obj_data['dinov2']:
                if obj_data['dinov2']['evaluation']['overall'].get('auroc'):
                    dino_aurocs.append(obj_data['dinov2']['evaluation']['overall']['auroc'])
                dino_times.append(obj_data['dinov2']['evaluation']['overall']['avg_inference_time'])
        
        if resnet_aurocs:
            report_lines.append(f"ResNet Average AUROC: {np.mean(resnet_aurocs):.3f}")
        if dino_aurocs:
            report_lines.append(f"DINOv2 Average AUROC: {np.mean(dino_aurocs):.3f}")
        if resnet_times:
            report_lines.append(f"ResNet Average Inference Time: {np.mean(resnet_times)*1000:.1f}ms")
        if dino_times:
            report_lines.append(f"DINOv2 Average Inference Time: {np.mean(dino_times)*1000:.1f}ms")
        
        # Add threshold analysis section
        report_lines.extend(["", "=" * 80, "", "Threshold Analysis:", "-" * 40])
        
        threshold_data = []
        
        for object_name, obj_data in self.results['objects'].items():
            for model_type in ['resnet', 'dinov2']:
                if model_type in obj_data and 'evaluation' in obj_data[model_type]:
                    eval_data = obj_data[model_type]['evaluation']['overall']
                    
                    if 'f1_optimal' in eval_data and 'model_threshold_metrics' in eval_data:
                        row = {
                            'Object': object_name,
                            'Model': model_type.upper(),
                            '95%_Threshold': f"{eval_data['model_threshold_metrics']['threshold']:.4f}",
                            '95%_F1': f"{eval_data['model_threshold_metrics']['f1_score']:.3f}",
                            'Optimal_Threshold': f"{eval_data['f1_optimal']['threshold']:.4f}",
                            'Optimal_F1': f"{eval_data['f1_optimal']['f1_score']:.3f}",
                            'F1_Improvement': f"{(eval_data['f1_optimal']['f1_score'] - eval_data['model_threshold_metrics']['f1_score']):.3f}"
                        }
                        threshold_data.append(row)
        
        if threshold_data:
            df_threshold = pd.DataFrame(threshold_data)
            report_lines.append(df_threshold.to_string(index=False))
        
        # Per-category results
        report_lines.extend(["", "=" * 80, "", "Detailed Per-Category Results:", "-" * 80])
        
        for object_name, obj_data in self.results['objects'].items():
            report_lines.extend(["", f"Object: {object_name}", "-" * 40])
            
            # ResNet categories
            if 'resnet' in obj_data and 'evaluation' in obj_data['resnet']:
                report_lines.append("ResNet:")
                for cat_name, cat_data in obj_data['resnet']['evaluation']['categories'].items():
                    if cat_name != 'good' and 'auroc' in cat_data and cat_data['auroc']:
                        report_lines.append(f"  {cat_name}: AUROC={cat_data['auroc']:.3f}, "
                                          f"Avg Time={cat_data['avg_inference_time']*1000:.1f}ms, "
                                          f"N={cat_data['n_images']}")
            
            # DINOv2 categories  
            if 'dinov2' in obj_data and 'evaluation' in obj_data['dinov2']:
                report_lines.append("DINOv2:")
                for cat_name, cat_data in obj_data['dinov2']['evaluation']['categories'].items():
                    if cat_name != 'good' and 'auroc' in cat_data and cat_data['auroc']:
                        report_lines.append(f"  {cat_name}: AUROC={cat_data['auroc']:.3f}, "
                                          f"Avg Time={cat_data['avg_inference_time']*1000:.1f}ms, "
                                          f"N={cat_data['n_images']}")
        
        # Save report
        report_file = self.output_dir / "benchmark_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nReport saved to: {report_file}")
        
        # Generate visualizations
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate comparison plots"""
        # Prepare data
        objects = []
        resnet_aurocs = []
        dino_aurocs = []
        resnet_times = []
        dino_times = []
        
        for obj_name, obj_data in self.results['objects'].items():
            if ('resnet' in obj_data and 'error' not in obj_data['resnet'] and
                'dinov2' in obj_data and 'error' not in obj_data['dinov2']):
                
                objects.append(obj_name)
                resnet_aurocs.append(obj_data['resnet']['evaluation']['overall'].get('auroc', 0))
                dino_aurocs.append(obj_data['dinov2']['evaluation']['overall'].get('auroc', 0))
                resnet_times.append(obj_data['resnet']['evaluation']['overall']['avg_inference_time'] * 1000)
                dino_times.append(obj_data['dinov2']['evaluation']['overall']['avg_inference_time'] * 1000)
        
        if not objects:
            return
        
        # AUROC comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        x = np.arange(len(objects))
        width = 0.35
        
        ax1.bar(x - width/2, resnet_aurocs, width, label='ResNet', color='blue', alpha=0.7)
        ax1.bar(x + width/2, dino_aurocs, width, label='DINOv2', color='orange', alpha=0.7)
        ax1.set_xlabel('MVTec Objects')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC Comparison: ResNet vs DINOv2')
        ax1.set_xticks(x)
        ax1.set_xticklabels(objects, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Inference time comparison
        ax2.bar(x - width/2, resnet_times, width, label='ResNet', color='blue', alpha=0.7)
        ax2.bar(x + width/2, dino_times, width, label='DINOv2', color='orange', alpha=0.7)
        ax2.set_xlabel('MVTec Objects')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Comparison: ResNet vs DINOv2')
        ax2.set_xticks(x)
        ax2.set_xticklabels(objects, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plot: AUROC vs Inference Time
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(resnet_times, resnet_aurocs, s=100, alpha=0.7, label='ResNet', color='blue')
        ax.scatter(dino_times, dino_aurocs, s=100, alpha=0.7, label='DINOv2', color='orange')
        
        # Add object labels
        for i, obj in enumerate(objects):
            ax.annotate(obj, (resnet_times[i], resnet_aurocs[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
            ax.annotate(obj, (dino_times[i], dino_aurocs[i]), 
                       xytext=(5, -5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('AUROC')
        ax.set_title('AUROC vs Inference Time Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'auroc_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate threshold comparison plot
        self._generate_threshold_comparison_plot()
        
        print(f"Plots saved to: {self.output_dir}")
    
    def _generate_threshold_comparison_plot(self):
        """Generate plot comparing 95-percentile vs F1-optimal thresholds"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, model_type in enumerate(['resnet', 'dinov2']):
            f1_95_scores = []
            f1_optimal_scores = []
            threshold_95 = []
            threshold_optimal = []
            objects = []
            
            for obj_name, obj_data in self.results['objects'].items():
                if model_type in obj_data and 'evaluation' in obj_data[model_type]:
                    eval_data = obj_data[model_type]['evaluation']['overall']
                    
                    if 'f1_optimal' in eval_data and 'model_threshold_metrics' in eval_data:
                        objects.append(obj_name)
                        f1_95_scores.append(eval_data['model_threshold_metrics']['f1_score'])
                        f1_optimal_scores.append(eval_data['f1_optimal']['f1_score'])
                        threshold_95.append(eval_data['model_threshold_metrics']['threshold'])
                        threshold_optimal.append(eval_data['f1_optimal']['threshold'])
            
            if objects:
                # F1 score comparison
                ax1 = axes[0, idx]
                x = np.arange(len(objects))
                width = 0.35
                
                ax1.bar(x - width/2, f1_95_scores, width, label='95-percentile', alpha=0.7)
                ax1.bar(x + width/2, f1_optimal_scores, width, label='F1-optimal', alpha=0.7)
                ax1.set_xlabel('Objects')
                ax1.set_ylabel('F1 Score')
                ax1.set_title(f'{model_type.upper()}: F1 Score Comparison')
                ax1.set_xticks(x)
                ax1.set_xticklabels(objects, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Threshold values comparison
                ax2 = axes[1, idx]
                ax2.scatter(threshold_95, threshold_optimal, alpha=0.7)
                
                # Add diagonal line
                min_val = min(min(threshold_95), min(threshold_optimal))
                max_val = max(max(threshold_95), max(threshold_optimal))
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                # Add labels for each point
                for i, obj in enumerate(objects):
                    ax2.annotate(obj, (threshold_95[i], threshold_optimal[i]), 
                               fontsize=8, alpha=0.7)
                
                ax2.set_xlabel('95-percentile Threshold')
                ax2.set_ylabel('F1-optimal Threshold')
                ax2.set_title(f'{model_type.upper()}: Threshold Comparison')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run the benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MVTec AD Benchmark for PatchCore')
    parser.add_argument('--mvtec-root', type=str, default='mvtec',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--category', type=str, nargs='*', default=None,
                        help='Subset of categories to benchmark (e.g. --category leather bottle)')
    parser.add_argument('--sample-ratio', type=float, default=0.01,
                        help='Sample ratio for PatchCore fit (e.g. 0.01 for 1%)')
    parser.add_argument('--model', type=str, choices=['dinov2', 'wideresnet'], default=None,
                        help='Model to benchmark: dinov2 or wideresnet (default: both)')

    args = parser.parse_args()

    # Check if MVTec directory exists
    if not Path(args.mvtec_root).exists():
        print(f"Error: MVTec directory not found at {args.mvtec_root}")
        return

    # Run benchmark
    benchmark = MVTecBenchmark(args.mvtec_root, args.output_dir, categories=args.category, sample_ratio=args.sample_ratio)
    benchmark.run_benchmark(model=args.model)

    print("\nBenchmark completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()