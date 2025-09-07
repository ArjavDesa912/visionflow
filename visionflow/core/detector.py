import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from ..utils.config import config

class ObjectDetector:
    """Real-time object detection using YOLO-v8"""
    
    def __init__(self, model_name: str = None, confidence_threshold: float = None):
        self.model_name = model_name or config.get('models.object_detection.model', 'yolov8x')
        self.confidence_threshold = confidence_threshold or config.get('models.object_detection.confidence_threshold', 0.5)
        self.nms_threshold = config.get('models.object_detection.nms_threshold', 0.4)
        
        self.device = config.get('models.device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = self._load_model()
        self.class_names = self.model.names
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            model = YOLO(self.model_name)
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Class ID
            - class_name: Class name
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        results = self.model(image, conf=self.confidence_threshold, iou=self.nms_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists for each image
        """
        return [self.detect(img) for img in images]
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def get_detections_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """
        Filter detections by class name
        
        Args:
            detections: List of detection dictionaries
            class_name: Class name to filter by
            
        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d['class_name'] == class_name]
    
    def count_objects(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count objects by class
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with class names as keys and counts as values
        """
        counts = {}
        for detection in detections:
            class_name = detection['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts