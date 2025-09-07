import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
from efficientnet_pytorch import EfficientNet
from timm import create_model
from ..utils.config import config

class ImageClassifier:
    """Image classification using EfficientNet-B7 and Vision Transformers"""
    
    def __init__(self, model_name: str = None, num_classes: int = None):
        self.model_name = model_name or config.get('models.classification.model', 'efficientnet-b7')
        self.num_classes = num_classes or config.get('models.classification.num_classes', 1000)
        
        self.device = config.get('models.device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
        # ImageNet class names (for demonstration)
        self.class_names = self._load_class_names()
    
    def _load_model(self):
        """Load classification model"""
        if 'efficientnet' in self.model_name:
            model = EfficientNet.from_pretrained(self.model_name)
            # Modify classifier for custom number of classes if needed
            if self.num_classes != 1000:
                model._fc = nn.Linear(model._fc.in_features, self.num_classes)
        elif 'vit' in self.model_name:
            model = create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        if 'efficientnet' in self.model_name:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif 'vit' in self.model_name:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _load_class_names(self):
        """Load ImageNet class names"""
        # This is a simplified version - in practice, you'd load from a file
        return [f"class_{i}" for i in range(self.num_classes)]
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for classification
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Preprocessed image tensor
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def classify(self, image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Classify image
        
        Args:
            image: Input image as numpy array
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries with keys:
            - class_id: Class ID
            - class_name: Class name
            - confidence: Confidence score
        """
        import cv2
        
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            class_id = top_k_indices[i].item()
            confidence = top_k_probs[i].item()
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            predictions.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence
            })
        
        return predictions
    
    def classify_batch(self, images: List[np.ndarray], top_k: int = 5) -> List[List[Dict]]:
        """
        Classify batch of images
        
        Args:
            images: List of input images
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction lists for each image
        """
        input_tensors = torch.cat([self.preprocess_image(img) for img in images])
        
        with torch.no_grad():
            outputs = self.model(input_tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        batch_predictions = []
        for i in range(len(images)):
            top_k_probs, top_k_indices = torch.topk(probabilities[i], top_k)
            
            predictions = []
            for j in range(top_k):
                class_id = top_k_indices[j].item()
                confidence = top_k_probs[j].item()
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                predictions.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
            
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Feature vector as numpy array
        """
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            # Get features before the final classification layer
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(input_tensor)
            else:
                # For models without explicit feature extraction
                features = input_tensor
                for name, module in self.model.named_children():
                    if name not in ['_fc', 'classifier', 'head']:
                        features = module(features)
                    else:
                        break
            
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(start_dim=1)
        
        return features.cpu().numpy()[0]
    
    def get_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate similarity between two images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Similarity score (0-1)
        """
        features1 = self.get_feature_vector(image1)
        features2 = self.get_feature_vector(image2)
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return float(similarity)