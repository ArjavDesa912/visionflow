#!/usr/bin/env python3
"""
VisionFlow Pro Examples

This script demonstrates the key features of VisionFlow Pro:
- Object detection with YOLO-v8
- Image classification with EfficientNet-B7
- Deepfake detection with Vision Transformers
- Video generation with Stable Video Diffusion
- Visual search with CLIP embeddings and Faiss
"""

import cv2
import numpy as np
import os
import sys
from PIL import Image
import torch

# Add the visionflow module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visionflow import (
    ObjectDetector, 
    ImageClassifier, 
    DeepfakeDetector,
    VideoGenerator,
    VideoProcessor,
    VisualSearch,
    Config
)

def create_sample_image():
    """Create a sample image for demonstration"""
    # Create a simple image with some shapes
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles and circles
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(image, (400, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(image, (150, 300), (350, 400), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(image, 'VisionFlow Pro Sample', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image

def example_object_detection():
    """Example: Object detection with YOLO-v8"""
    print("=" * 50)
    print("Object Detection Example")
    print("=" * 50)
    
    # Create sample image
    image = create_sample_image()
    
    # Initialize detector
    detector = ObjectDetector(model_name='yolov8n', confidence_threshold=0.5)
    
    # Detect objects
    detections = detector.detect(image)
    
    print(f"Detected {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
    
    # Draw detections
    result_image = detector.draw_detections(image, detections)
    
    # Save result
    cv2.imwrite('object_detection_result.jpg', result_image)
    print("Result saved to object_detection_result.jpg")

def example_image_classification():
    """Example: Image classification with EfficientNet-B7"""
    print("\n" + "=" * 50)
    print("Image Classification Example")
    print("=" * 50)
    
    # Create sample image
    image = create_sample_image()
    
    # Initialize classifier
    classifier = ImageClassifier(model_name='efficientnet-b0')
    
    # Classify image
    predictions = classifier.classify(image, top_k=5)
    
    print("Top 5 predictions:")
    for i, prediction in enumerate(predictions):
        print(f"  {i+1}. {prediction['class_name']}: {prediction['confidence']:.4f}")
    
    # Extract features
    features = classifier.get_feature_vector(image)
    print(f"Feature vector shape: {features.shape}")

def example_deepfake_detection():
    """Example: Deepfake detection with Vision Transformers"""
    print("\n" + "=" * 50)
    print("Deepfake Detection Example")
    print("=" * 50)
    
    # Create sample image
    image = create_sample_image()
    
    # Initialize detector
    detector = DeepfakeDetector(confidence_threshold=0.8)
    
    # Detect deepfake
    result = detector.detect(image)
    
    print("Deepfake detection result:")
    print(f"  Is deepfake: {result['is_deepfake']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Real probability: {result['real_probability']:.4f}")
    print(f"  Fake probability: {result['fake_probability']:.4f}")
    
    # Analyze artifacts
    artifacts = detector.analyze_artifacts(image)
    print(f"\nArtifact analysis:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")

def example_video_generation():
    """Example: Video generation with Stable Video Diffusion"""
    print("\n" + "=" * 50)
    print("Video Generation Example")
    print("=" * 50)
    
    # Create sample image
    image = create_sample_image()
    cv2.imwrite('sample_image.jpg', image)
    
    # Initialize video generator
    generator = VideoGenerator()
    
    # Generate short video (this would require actual model downloads)
    print("Note: Video generation requires downloading large models.")
    print("Skipping actual generation in this example.")
    
    # Demonstrate the API
    print("Video generation API demonstration:")
    print("  generator.generate_from_image('sample_image.jpg', duration=2.0)")
    print("  generator.generate_from_text('A beautiful landscape', duration=3.0)")

def example_video_processing():
    """Example: Video processing with frame interpolation and stabilization"""
    print("\n" + "=" * 50)
    print("Video Processing Example")
    print("=" * 50)
    
    # Initialize video processor
    processor = VideoProcessor()
    
    # Create sample frames
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add moving circle
        x = 100 + i * 40
        cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
        frames.append(frame)
    
    # Save as video
    processor.save_video(frames, 'sample_video.mp4', fps=10)
    print("Sample video created: sample_video.mp4")
    
    # Process video
    stats = processor.process_video(
        'sample_video.mp4',
        'processed_video.mp4',
        enhance=True,
        interpolate=False
    )
    
    print(f"Video processing stats:")
    print(f"  Input frames: {stats['input_frames']}")
    print(f"  Output frames: {stats['output_frames']}")
    print(f"  Processing steps: {stats['processing_steps']}")

def example_visual_search():
    """Example: Visual search with CLIP embeddings and Faiss"""
    print("\n" + "=" * 50)
    print("Visual Search Example")
    print("=" * 50)
    
    # Initialize search engine
    search_engine = VisualSearch()
    
    # Create sample images
    sample_images = []
    for i in range(5):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add different colored rectangles
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i]
        cv2.rectangle(image, (100 + i*20, 100 + i*20), (300 + i*20, 300 + i*20), color, -1)
        cv2.imwrite(f'sample_image_{i}.jpg', image)
        sample_images.append(f'sample_image_{i}.jpg')
    
    # Add images to search index
    search_engine.add_images(sample_images)
    
    # Search by image
    print("Search by image:")
    results = search_engine.search_by_image('sample_image_0.jpg', top_k=3)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['image_path']} (score: {result.get('similarity_score', 0):.4f})")
    
    # Search by text
    print("\nSearch by text:")
    results = search_engine.search_by_text('blue rectangle', top_k=3)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['image_path']} (score: {result.get('similarity_score', 0):.4f})")
    
    # Performance stats
    stats = search_engine.get_performance_stats()
    print(f"\nSearch performance:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Average latency: {stats['average_latency']:.4f}s")

def example_configuration():
    """Example: Configuration management"""
    print("\n" + "=" * 50)
    print("Configuration Example")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Show current configuration
    print("Current configuration:")
    print(f"  Device: {config.get('models.device')}")
    print(f"  Object detection model: {config.get('models.object_detection.model')}")
    print(f"  Video frame rate: {config.get('video.frame_rate')}")
    print(f"  Search embedding dimension: {config.get('search.embedding_dim')}")
    
    # Update configuration
    config.update('video.frame_rate', 60)
    print(f"Updated video frame rate: {config.get('video.frame_rate')}")

def example_batch_processing():
    """Example: Batch processing for multiple images"""
    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("=" * 50)
    
    # Create sample images
    images = []
    for i in range(3):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(image, (200 + i*50, 240), 50, (255, 255, 255), -1)
        images.append(image)
    
    # Initialize models
    detector = ObjectDetector()
    classifier = ImageClassifier()
    deepfake_detector = DeepfakeDetector()
    
    # Batch object detection
    print("Batch object detection:")
    detections_list = detector.detect_batch(images)
    for i, detections in enumerate(detections_list):
        print(f"  Image {i+1}: {len(detections)} detections")
    
    # Batch classification
    print("\nBatch image classification:")
    predictions_list = classifier.classify_batch(images, top_k=3)
    for i, predictions in enumerate(predictions_list):
        print(f"  Image {i+1}: {len(predictions)} predictions")
    
    # Batch deepfake detection
    print("\nBatch deepfake detection:")
    results_list = deepfake_detector.detect_batch(images)
    for i, result in enumerate(results_list):
        print(f"  Image {i+1}: deepfake={result['is_deepfake']}, confidence={result['confidence']:.3f}")

def main():
    """Run all examples"""
    print("VisionFlow Pro Examples")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 50)
    
    try:
        example_object_detection()
        example_image_classification()
        example_deepfake_detection()
        example_video_generation()
        example_video_processing()
        example_visual_search()
        example_configuration()
        example_batch_processing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()