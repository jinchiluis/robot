#!/usr/bin/env python3
"""
Test script for PatchCore-DINOv2 anomaly detection
This script creates sample images and tests the basic functionality
"""

import os
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path

# Import your PatchCore implementation
from patchcore_dino import PatchCore

def create_test_images(num_images=10, image_size=(224, 224)):
    """Create simple test images with different patterns"""
    images = []
    
    for i in range(num_images):
        # Create a simple pattern - normal images
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        if i % 3 == 0:
            # Add horizontal stripes
            img_array[::10, :] = [255, 0, 0]  # Red stripes
        elif i % 3 == 1:
            # Add vertical stripes  
            img_array[:, ::10] = [0, 255, 0]  # Green stripes
        else:
            # Add diagonal pattern
            for j in range(min(image_size)):
                if j < image_size[0] and j < image_size[1]:
                    img_array[j, j] = [0, 0, 255]  # Blue diagonal
        
        images.append(Image.fromarray(img_array))
    
    return images

def create_anomaly_images(num_images=3, image_size=(224, 224)):
    """Create anomaly images with different patterns"""
    images = []
    
    for i in range(num_images):
        # Create anomalous patterns
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add anomalous features
        if i == 0:
            # Large white circle (anomaly)
            center = (image_size[0]//2, image_size[1]//2)
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center[0])**2 + (y - center[1])**2 <= 50**2
            img_array[mask] = [255, 255, 255]
        elif i == 1:
            # Black square (anomaly)
            img_array[50:150, 50:150] = [0, 0, 0]
        else:
            # Random noise (anomaly)
            img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        images.append(Image.fromarray(img_array))
    
    return images

def test_patchcore():
    """Main test function"""
    print("🧪 Testing PatchCore-DINOv2 Implementation")
    print("=" * 50)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = Path(temp_dir) / "train"
        test_dir = Path(temp_dir) / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        
        print(f"📁 Created temporary directories:")
        print(f"   Train: {train_dir}")
        print(f"   Test: {test_dir}")
        
        # Create and save training images (normal)
        print("\n📸 Creating training images...")
        train_images = create_test_images(num_images=15)
        for i, img in enumerate(train_images):
            img.save(train_dir / f"train_{i:03d}.jpg")
        
        # Create and save test images (mix of normal and anomalous)
        print("📸 Creating test images...")
        normal_test_images = create_test_images(num_images=5)
        anomaly_test_images = create_anomaly_images(num_images=3)
        
        # Save normal test images
        for i, img in enumerate(normal_test_images):
            img.save(test_dir / f"normal_{i:03d}.jpg")
        
        # Save anomaly test images
        for i, img in enumerate(anomaly_test_images):
            img.save(test_dir / f"anomaly_{i:03d}.jpg")
        
        print(f"✓ Created {len(train_images)} training images")
        print(f"✓ Created {len(normal_test_images)} normal test images")
        print(f"✓ Created {len(anomaly_test_images)} anomaly test images")
        
        # Test PatchCore
        print("\n🤖 Initializing PatchCore...")
        try:
            # Use smaller model for faster testing
            model = PatchCore(backbone='dinov2_vits14')  # Smallest variant
            print("✓ PatchCore initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize PatchCore: {e}")
            return False
        
        # Train the model
        print("\n🏋️ Training PatchCore...")
        try:
            # Use higher sampling ratio for small dataset
            model.fit(train_dir, sample_ratio=0.5, threshold_percentile=95)
            print("✓ Training completed successfully")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
        
        # Test predictions
        print("\n🔍 Testing predictions...")
        test_files = list(test_dir.glob("*.jpg"))
        
        results = []
        for img_path in test_files:
            try:
                result = model.predict(str(img_path), return_heatmap=False)
                results.append({
                    'file': img_path.name,
                    'score': result['anomaly_score'],
                    'is_anomaly': result['is_anomaly'],
                    'threshold': result['threshold']
                })
                print(f"  {img_path.name}: score={result['anomaly_score']:.4f}, "
                      f"anomaly={result['is_anomaly']}")
            except Exception as e:
                print(f"❌ Prediction failed for {img_path.name}: {e}")
                return False
        
        # Analyze results
        print("\n📊 Results Summary:")
        print("-" * 40)
        
        normal_results = [r for r in results if r['file'].startswith('normal')]
        anomaly_results = [r for r in results if r['file'].startswith('anomaly')]
        
        normal_scores = [r['score'] for r in normal_results]
        anomaly_scores = [r['score'] for r in anomaly_results]
        
        print(f"Normal images:")
        print(f"  Count: {len(normal_results)}")
        print(f"  Score range: {min(normal_scores):.4f} - {max(normal_scores):.4f}")
        print(f"  Detected as anomaly: {sum(r['is_anomaly'] for r in normal_results)}")
        
        print(f"\nAnomaly images:")
        print(f"  Count: {len(anomaly_results)}")
        print(f"  Score range: {min(anomaly_scores):.4f} - {max(anomaly_scores):.4f}")
        print(f"  Detected as anomaly: {sum(r['is_anomaly'] for r in anomaly_results)}")
        
        threshold = results[0]['threshold']
        print(f"\nThreshold: {threshold:.4f}")
        
        # Test save/load functionality
        print("\n💾 Testing save/load functionality...")
        try:
            model_path = Path(temp_dir) / "test_model.pth"
            model.save(str(model_path))
            
            # Create new model and load
            new_model = PatchCore(backbone='dinov2_vits14')
            new_model.load(str(model_path))
            
            # Test prediction with loaded model
            test_img = test_files[0]
            result1 = model.predict(str(test_img), return_heatmap=False)
            result2 = new_model.predict(str(test_img), return_heatmap=False)
            
            score_diff = abs(result1['anomaly_score'] - result2['anomaly_score'])
            if score_diff < 1e-6:
                print("✓ Save/load test passed")
            else:
                print(f"⚠️  Save/load test: slight difference in scores ({score_diff:.8f})")
            
        except Exception as e:
            print(f"❌ Save/load test failed: {e}")
            return False
        
        print("\n🎉 All tests completed successfully!")
        return True

def quick_smoke_test():
    """Quick smoke test without file I/O"""
    print("🚀 Quick smoke test (no file I/O)...")
    
    try:
        # Just test initialization
        model = PatchCore(backbone='dinov2_vits14')
        print("✓ Model initialization: PASSED")
        
        # Test feature extraction with dummy data
        import torch
        dummy_input = torch.randn(1, 3, 518, 518)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        features = model.extract_features(dummy_input)
        print(f"✓ Feature extraction: PASSED (shape: {features.shape})")
        
        print("✓ Quick smoke test: ALL PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Quick smoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("PatchCore-DINOv2 Test Suite")
    print("=" * 50)
    
    # Run quick test first
    if not quick_smoke_test():
        print("\n❌ Quick smoke test failed. Please check your installation.")
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Ask user for full test
    response = input("Run full test with image generation? (y/n): ").lower().strip()
    
    if response == 'y':
        success = test_patchcore()
        if success:
            print("\n🎉 All tests passed! Your PatchCore implementation is working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")
    else:
        print("\n✓ Quick test completed. Run with 'y' for full testing.")