import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# Import both detectors
from dino_object_detector import OneShotObjectTrainer, ObjectDetector
from dino_object_detector_adv import AdvancedOneShotObjectTrainer, AdvancedObjectDetector


class DetectorComparison:
    """Compare performance between original and advanced DINO detectors"""
    
    def __init__(self, training_data: Dict[str, List[str]], test_images: List[str]):
        self.training_data = training_data
        self.test_images = test_images
        self.results = {
            'original': defaultdict(list),
            'advanced': defaultdict(list)
        }
        
    def train_both_models(self, objects_per_image: int = 1):
        """Train both models on the same data"""
        print("=" * 80)
        print("TRAINING PHASE")
        print("=" * 80)
        
        # Train original model
        print("\n[1/2] Training Original DINO Model...")
        start_time = time.time()
        self.original_trainer = OneShotObjectTrainer()
        self.original_trainer.train(self.training_data, objects_per_image=objects_per_image)
        self.original_trainer.save_model('comparison_original_model.pkl')
        original_train_time = time.time() - start_time
        print(f"Original model training time: {original_train_time:.2f}s")
        
        # Train advanced model
        print("\n[2/2] Training Advanced DINO Model...")
        start_time = time.time()
        self.advanced_trainer = AdvancedOneShotObjectTrainer()
        self.advanced_trainer.train(self.training_data, objects_per_image=objects_per_image)
        self.advanced_trainer.save_model('comparison_advanced_model.pkl')
        advanced_train_time = time.time() - start_time
        print(f"Advanced model training time: {advanced_train_time:.2f}s")
        
        self.results['training_times'] = {
            'original': original_train_time,
            'advanced': advanced_train_time
        }
        
        # Compare model sizes
        original_size = Path('comparison_original_model.pkl').stat().st_size / 1024  # KB
        advanced_size = Path('comparison_advanced_model.pkl').stat().st_size / 1024  # KB
        
        self.results['model_sizes'] = {
            'original': original_size,
            'advanced': advanced_size
        }
        
        print(f"\nModel sizes - Original: {original_size:.1f}KB, Advanced: {advanced_size:.1f}KB")
        
    def analyze_detections(self, detections1: Dict, detections2: Dict, image_name: str):
        """Analyze differences between two detection results"""
        analysis = {
            'total_objects': {
                'original': sum(len(v) for v in detections1.values()),
                'advanced': sum(len(v) for v in detections2.values())
            },
            'per_class': {},
            'spatial_differences': []
        }
        
        all_types = set(detections1.keys()) | set(detections2.keys())
        
        for obj_type in all_types:
            centers1 = detections1.get(obj_type, [])
            centers2 = detections2.get(obj_type, [])
            
            analysis['per_class'][obj_type] = {
                'original': len(centers1),
                'advanced': len(centers2)
            }
            
            # Calculate spatial differences for matched objects
            if centers1 and centers2:
                # Simple nearest neighbor matching
                for c1 in centers1:
                    if centers2:
                        distances = [np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) for c2 in centers2]
                        min_dist = min(distances)
                        analysis['spatial_differences'].append(min_dist)
        
        return analysis
    
    def run_comparison(self, visualize: bool = True, debug: bool = False):
        """Run detection comparison on test images"""
        print("\n" + "=" * 80)
        print("DETECTION PHASE")
        print("=" * 80)
        
        # Load detectors
        self.original_detector = ObjectDetector('comparison_original_model.pkl')
        self.advanced_detector = AdvancedObjectDetector('comparison_advanced_model.pkl')
        
        # Process each test image
        for img_path in self.test_images:
            print(f"\nProcessing: {img_path}")
            
            # Original detector
            start_time = time.time()
            if visualize:
                detections1, vis1 = self.original_detector.detect(img_path, visualize=True, debug=debug)
            else:
                detections1 = self.original_detector.detect(img_path, debug=debug)
            original_time = time.time() - start_time
            
            # Advanced detector
            start_time = time.time()
            if visualize:
                detections2, vis2 = self.advanced_detector.detect(img_path, visualize=True, debug=debug)
            else:
                detections2 = self.advanced_detector.detect(img_path, debug=debug)
            advanced_time = time.time() - start_time
            
            # Store timing results
            self.results['original']['inference_times'].append(original_time)
            self.results['advanced']['inference_times'].append(advanced_time)
            
            # Analyze detections
            analysis = self.analyze_detections(detections1, detections2, img_path)
            self.results['original']['detections'].append(detections1)
            self.results['advanced']['detections'].append(detections2)
            self.results['original']['analyses'].append(analysis)
            
            # Print comparison
            print(f"  Original: {analysis['total_objects']['original']} objects in {original_time:.3f}s")
            print(f"  Advanced: {analysis['total_objects']['advanced']} objects in {advanced_time:.3f}s")
            
            for obj_type in analysis['per_class']:
                orig_count = analysis['per_class'][obj_type]['original']
                adv_count = analysis['per_class'][obj_type]['advanced']
                if orig_count != adv_count:
                    print(f"    {obj_type}: Original={orig_count}, Advanced={adv_count}")
            
            if analysis['spatial_differences']:
                avg_diff = np.mean(analysis['spatial_differences'])
                print(f"  Average spatial difference: {avg_diff:.1f} pixels")
            
            # Visualize if requested
            if visualize:
                self.visualize_comparison(img_path, vis1, vis2, detections1, detections2)
    
    def visualize_comparison(self, img_path: str, vis1: np.ndarray, vis2: np.ndarray, 
                           det1: Dict, det2: Dict):
        """Create side-by-side visualization"""
        # Combine images side by side
        h1, w1 = vis1.shape[:2]
        h2, w2 = vis2.shape[:2]
        h = max(h1, h2)
        
        combined = np.zeros((h, w1 + w2 + 10, 3), dtype=np.uint8)
        combined[:h1, :w1] = vis1
        combined[:h2, w1+10:] = vis2
        
        # Add labels
        cv2.putText(combined, "Original DINO", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Advanced DINO", (w1+20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison image
        output_path = f"comparison_{Path(img_path).stem}.jpg"
        cv2.imwrite(output_path, combined)
        print(f"  Saved comparison to: {output_path}")
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("COMPARISON REPORT")
        print("=" * 80)
        
        # Training comparison
        print("\n1. TRAINING PERFORMANCE")
        print("-" * 40)
        orig_train = self.results['training_times']['original']
        adv_train = self.results['training_times']['advanced']
        print(f"Original model: {orig_train:.2f}s")
        print(f"Advanced model: {adv_train:.2f}s")
        print(f"Slowdown factor: {adv_train/orig_train:.2f}x")
        
        # Model size comparison
        print("\n2. MODEL SIZE")
        print("-" * 40)
        orig_size = self.results['model_sizes']['original']
        adv_size = self.results['model_sizes']['advanced']
        print(f"Original model: {orig_size:.1f} KB")
        print(f"Advanced model: {adv_size:.1f} KB")
        print(f"Size increase: {adv_size/orig_size:.2f}x")
        
        # Inference performance
        print("\n3. INFERENCE PERFORMANCE")
        print("-" * 40)
        orig_times = self.results['original']['inference_times']
        adv_times = self.results['advanced']['inference_times']
        
        if orig_times and adv_times:
            print(f"Original - Mean: {np.mean(orig_times):.3f}s, Std: {np.std(orig_times):.3f}s")
            print(f"Advanced - Mean: {np.mean(adv_times):.3f}s, Std: {np.std(adv_times):.3f}s")
            print(f"Slowdown factor: {np.mean(adv_times)/np.mean(orig_times):.2f}x")
        
        # Detection accuracy
        print("\n4. DETECTION RESULTS")
        print("-" * 40)
        
        total_orig = sum(a['total_objects']['original'] for a in self.results['original']['analyses'])
        total_adv = sum(a['total_objects']['advanced'] for a in self.results['original']['analyses'])
        
        print(f"Total objects detected:")
        print(f"  Original: {total_orig}")
        print(f"  Advanced: {total_adv}")
        
        # Per-class breakdown
        class_counts = defaultdict(lambda: {'original': 0, 'advanced': 0})
        for analysis in self.results['original']['analyses']:
            for obj_type, counts in analysis['per_class'].items():
                class_counts[obj_type]['original'] += counts['original']
                class_counts[obj_type]['advanced'] += counts['advanced']
        
        print("\nPer-class detection counts:")
        for obj_type, counts in class_counts.items():
            print(f"  {obj_type}: Original={counts['original']}, Advanced={counts['advanced']}")
        
        # Spatial accuracy (if objects detected by both)
        all_spatial_diffs = []
        for analysis in self.results['original']['analyses']:
            all_spatial_diffs.extend(analysis['spatial_differences'])
        
        if all_spatial_diffs:
            print(f"\n5. SPATIAL CONSISTENCY")
            print("-" * 40)
            print(f"Average position difference: {np.mean(all_spatial_diffs):.1f} pixels")
            print(f"Max position difference: {np.max(all_spatial_diffs):.1f} pixels")
        
        # Feature analysis
        print("\n6. FEATURE COMPARISON")
        print("-" * 40)
        print("Original: Uses global features only (384-dim)")
        print("Advanced: Uses global + patch features, multi-scale, multiple prototypes")
        
        # Recommendations
        print("\n7. RECOMMENDATIONS")
        print("-" * 40)
        
        speed_ratio = np.mean(adv_times)/np.mean(orig_times) if orig_times and adv_times else 1
        detection_improvement = (total_adv - total_orig) / max(total_orig, 1) * 100
        
        if speed_ratio < 2 and detection_improvement > 10:
            print("✓ Advanced model recommended: Better detection with acceptable speed")
        elif speed_ratio > 3:
            print("⚠ Original model recommended: Advanced model too slow for marginal gains")
        else:
            print("~ Both models viable: Choose based on speed vs accuracy requirements")
        
        print(f"\nSpeed-Accuracy Trade-off:")
        print(f"  {speed_ratio:.1f}x slower for {abs(detection_improvement):.0f}% "
              f"{'more' if detection_improvement > 0 else 'fewer'} detections")
    
    def plot_comparison(self):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training time comparison
        ax = axes[0, 0]
        models = ['Original', 'Advanced']
        train_times = [self.results['training_times']['original'], 
                      self.results['training_times']['advanced']]
        ax.bar(models, train_times, color=['blue', 'orange'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time Comparison')
        
        # 2. Model size comparison
        ax = axes[0, 1]
        sizes = [self.results['model_sizes']['original'], 
                self.results['model_sizes']['advanced']]
        ax.bar(models, sizes, color=['blue', 'orange'])
        ax.set_ylabel('Size (KB)')
        ax.set_title('Model Size Comparison')
        
        # 3. Inference time distribution
        ax = axes[1, 0]
        if self.results['original']['inference_times']:
            ax.hist(self.results['original']['inference_times'], alpha=0.5, 
                   label='Original', color='blue', bins=10)
            ax.hist(self.results['advanced']['inference_times'], alpha=0.5, 
                   label='Advanced', color='orange', bins=10)
            ax.set_xlabel('Inference Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Time Distribution')
            ax.legend()
        
        # 4. Detection counts per image
        ax = axes[1, 1]
        if self.results['original']['analyses']:
            orig_counts = [a['total_objects']['original'] for a in self.results['original']['analyses']]
            adv_counts = [a['total_objects']['advanced'] for a in self.results['original']['analyses']]
            x = range(len(orig_counts))
            width = 0.35
            ax.bar([i - width/2 for i in x], orig_counts, width, label='Original', color='blue')
            ax.bar([i + width/2 for i in x], adv_counts, width, label='Advanced', color='orange')
            ax.set_xlabel('Test Image Index')
            ax.set_ylabel('Objects Detected')
            ax.set_title('Detection Count Comparison')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('comparison_analysis.png', dpi=150)
        print("\nSaved analysis plots to: comparison_analysis.png")


# Example usage
if __name__ == "__main__":
    # Define training data
    training_data = {
        'red_object': ['training/r1.png', 'training/r2.png', 'training/r3.png', 
                      'training/r4.png', 'training/r5.png', 'training/r6.png', 'training/r7.png'],
        'blue_object': ['training/b1.png', 'training/b2.png', 'training/b3.png', 
                       'training/b4.png', 'training/b5.png', 'training/b6.png', 'training/b7.png']
    }
    
    # Define test images
    test_images = ['training/inference.png']  # Add more test images as needed
    
    # Run comparison
    comparison = DetectorComparison(training_data, test_images)
    
    # Train both models
    comparison.train_both_models(objects_per_image=1)
    
    # Run detection comparison
    comparison.run_comparison(visualize=True, debug=True)
    
    # Generate report
    comparison.generate_report()
    
    # Generate plots
    try:
        comparison.plot_comparison()
    except Exception as e:
        print(f"\nCould not generate plots: {e}")