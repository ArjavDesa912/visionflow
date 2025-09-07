"""
VisionFlow Pro - Advanced Multi-Modal Computer Vision Platform

A comprehensive computer vision platform integrating PyTorch and TensorFlow
for multi-task visual processing, video generation, and visual search.
"""

__version__ = "1.0.0"
__author__ = "VisionFlow Pro Team"

from .core.detector import ObjectDetector
from .core.classifier import ImageClassifier
from .core.deepfake_detector import DeepfakeDetector
from .video.generator import VideoGenerator
from .video.processor import VideoProcessor
from .search.engine import VisualSearch
from .utils.config import Config

__all__ = [
    'ObjectDetector',
    'ImageClassifier', 
    'DeepfakeDetector',
    'VideoGenerator',
    'VideoProcessor',
    'VisualSearch',
    'Config'
]