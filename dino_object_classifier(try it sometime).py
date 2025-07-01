import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import pickle
from pathlib import Path
from typing import Dict, List, Union
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# You'll need to install: pip install torch torchvision opencv-python pillow tqdm


class DINOFeatureExtractor:
    """
    Simple feature extractor using DINOv2 for image classification.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load DINOv2 model
        print("Loading DINOv2 model...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.dino_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Extract DINOv2 features from a single image"""
        # Handle different input types
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Transform and extract features
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.dino_model(tensor).cpu().numpy()[0]
        
        # L2 normalize
        features = features / np.linalg.norm(features)
        return features
    
    def extract_features_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> np.ndarray:
        """Extract DINOv2 features from a batch of images"""
        if not images:
            return np.array([])
        
        features = []
        for img in tqdm(images, desc="Extracting features"):
            features.append(self.extract_features(img))
        
        return np.array(features)


class ObjectClassifierTrainer:
    """
    Trains a simple object classifier using DINOv2 features.
    Each object type is represented by a prototype (mean feature vector).
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.extractor = DINOFeatureExtractor(device)
        self.prototypes = {}
        self.config = {
            'similarity_threshold': 0.75
        }
    
    def train(self, training_images_dict: Dict[str, List[str]]) -> Dict:
        """
        Train the classifier on provided cropped images.
        
        Args:
            training_images_dict: Dictionary mapping object types to lists of image paths
                                Example: {'object1': ['obj1_1.jpg', 'obj1_2.jpg'], 
                                         'object2': ['obj2_1.jpg', 'obj2_2.jpg']}
        
        Returns:
            Dictionary with training results and statistics
        """
        print("Starting training process...")
        all_features = {}
        
        for obj_type, image_paths in training_images_dict.items():
            print(f"\nProcessing {obj_type} images...")
            
            # Extract features from all images
            features = self.extractor.extract_features_batch(image_paths)
            
            if len(features) == 0:
                print(f"Warning: No valid images found for {obj_type}")
                continue
            
            all_features[obj_type] = features
            
            # Create prototype (mean of all features)
            prototype = np.mean(features, axis=0)
            self.prototypes[obj_type] = prototype / np.linalg.norm(prototype)
            
            print(f"{obj_type}: {len(features)} training images processed")
        
        # Check prototype similarities
        print("\nPrototype similarities:")
        obj_types = list(self.prototypes.keys())
        for i, type1 in enumerate(obj_types):
            for j, type2 in enumerate(obj_types):
                if i < j:
                    sim = np.dot(self.prototypes[type1], self.prototypes[type2])
                    print(f"  {type1} <-> {type2}: {sim:.3f}")
                    if sim > 0.9:
                        print(f"  WARNING: Very high similarity between {type1} and {type2}!")
        
        # Calculate intra-class variance
        print("\nIntra-class variance (lower is better):")
        for obj_type, features in all_features.items():
            if len(features) > 1:
                prototype = self.prototypes[obj_type]
                similarities = [np.dot(feat, prototype) for feat in features]
                variance = np.std(similarities)
                print(f"  {obj_type}: {variance:.4f} (mean similarity: {np.mean(similarities):.3f})")
        
        return {
            'prototypes': self.prototypes,
            'config': self.config,
            'feature_counts': {k: len(v) for k, v in all_features.items()},
            'training_complete': True
        }
    
    def save_model(self, save_path: str):
        """Save trained model to disk"""
        model_data = {
            'prototypes': self.prototypes,
            'config': self.config,
            'model_info': {
                'feature_dim': 384,
                'dinov2_model': 'dinov2_vits14',
                'type': 'object_classifier'
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")


class ObjectClassifier:
    """
    Simple object classifier using trained prototypes.
    Classifies cropped images into object types.
    """
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.extractor = DINOFeatureExtractor(device)
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.prototypes = model_data['prototypes']
        self.config = model_data['config']
        self.object_types = list(self.prototypes.keys())
        
        print(f"Loaded model with object types: {self.object_types}")
    
    def classify(self, image: Union[str, np.ndarray, Image.Image], 
                 return_all_scores: bool = False) -> Union[str, Dict]:
        """
        Classify a single cropped image.
        
        Args:
            image: Path to image, numpy array, or PIL Image
            return_all_scores: If True, return scores for all object types
        
        Returns:
            Object type (string) or dict with all scores if return_all_scores=True
        """
        # Extract features
        features = self.extractor.extract_features(image)
        
        # Compare with all prototypes
        similarities = {}
        for obj_type, prototype in self.prototypes.items():
            sim = np.dot(features, prototype)
            similarities[obj_type] = float(sim)
        
        # Find best match
        best_type = max(similarities, key=similarities.get)
        best_score = similarities[best_type]
        
        if return_all_scores:
            return {
                'predicted_type': best_type if best_score > self.config['similarity_threshold'] else 'unknown',
                'confidence': best_score,
                'all_scores': similarities
            }
        else:
            if best_score > self.config['similarity_threshold']:
                return best_type
            else:
                return 'unknown'
    
    def classify_batch(self, images: List[Union[str, np.ndarray, Image.Image]], 
                      verbose: bool = True) -> List[Dict]:
        """
        Classify multiple cropped images.
        
        Args:
            images: List of image paths, numpy arrays, or PIL Images
            verbose: Print progress
        
        Returns:
            List of classification results
        """
        results = []
        
        for i, image in enumerate(images):
            if verbose:
                print(f"Classifying image {i+1}/{len(images)}...", end='\r')
            
            result = self.classify(image, return_all_scores=True)
            results.append(result)
        
        if verbose:
            print(f"\nClassification complete. Processed {len(images)} images.")
            
            # Summary statistics
            predictions = [r['predicted_type'] for r in results]
            for obj_type in set(predictions):
                count = predictions.count(obj_type)
                print(f"  {obj_type}: {count} images")
        
        return results


# Example usage
if __name__ == "__main__":
    # Training phase
    trainer = ObjectClassifierTrainer()
    
    # Prepare training images - cropped images of each object type
    training_data = {
        'object1': ['crops/obj1_sample1.png', 'crops/obj1_sample2.png', 'crops/obj1_sample3.png'],
        'object2': ['crops/obj2_sample1.png', 'crops/obj2_sample2.png', 'crops/obj2_sample3.png'],
        'object3': ['crops/obj3_sample1.png', 'crops/obj3_sample2.png', 'crops/obj3_sample3.png']
    }
    
    # Train model
    results = trainer.train(training_data)
    
    # Save model
    trainer.save_model('object_classifier_model.pkl')
    
    # Inference phase
    classifier = ObjectClassifier('object_classifier_model.pkl')
    
    # Classify single image
    object_type = classifier.classify('crops/test_object.png')
    print(f"\nSingle image classification: {object_type}")
    
    # Classify with detailed scores
    detailed = classifier.classify('crops/test_object.png', return_all_scores=True)
    print(f"\nDetailed classification:")
    print(f"  Predicted: {detailed['predicted_type']}")
    print(f"  Confidence: {detailed['confidence']:.3f}")
    print(f"  All scores: {detailed['all_scores']}")
    
    # Batch classification
    test_images = ['crops/test1.png', 'crops/test2.png', 'crops/test3.png']
    batch_results = classifier.classify_batch(test_images)