#!/usr/bin/env python3
"""
VisionFlow Pro CLI Interface
"""

import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image

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

def main():
    parser = argparse.ArgumentParser(description='VisionFlow Pro - Advanced Multi-Modal Computer Vision Platform')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Object Detection
    detect_parser = subparsers.add_parser('detect', help='Object detection')
    detect_parser.add_argument('image', help='Path to image file')
    detect_parser.add_argument('--model', default='yolov8x', help='YOLO model to use')
    detect_parser.add_argument('--output', help='Output image path')
    detect_parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    # Image Classification
    classify_parser = subparsers.add_parser('classify', help='Image classification')
    classify_parser.add_argument('image', help='Path to image file')
    classify_parser.add_argument('--model', default='efficientnet-b7', help='Classification model')
    classify_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')
    
    # Deepfake Detection
    deepfake_parser = subparsers.add_parser('deepfake', help='Deepfake detection')
    deepfake_parser.add_argument('image', help='Path to image file')
    deepfake_parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
    deepfake_parser.add_argument('--heatmap', help='Output heatmap path')
    
    # Video Generation
    video_gen_parser = subparsers.add_parser('generate-video', help='Generate video from image')
    video_gen_parser.add_argument('input', help='Input image path or text prompt')
    video_gen_parser.add_argument('--duration', type=float, default=2.0, help='Video duration in seconds')
    video_gen_parser.add_argument('--output', help='Output video path')
    video_gen_parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    video_gen_parser.add_argument('--text', action='store_true', help='Generate from text prompt')
    
    # Video Processing
    video_proc_parser = subparsers.add_parser('process-video', help='Process video')
    video_proc_parser.add_argument('input', help='Input video path')
    video_proc_parser.add_argument('output', help='Output video path')
    video_proc_parser.add_argument('--stabilize', action='store_true', help='Apply stabilization')
    video_proc_parser.add_argument('--enhance', action='store_true', help='Apply enhancement')
    video_proc_parser.add_argument('--interpolate', action='store_true', help='Apply frame interpolation')
    video_proc_parser.add_argument('--target-fps', type=int, help='Target FPS for interpolation')
    
    # Visual Search
    search_parser = subparsers.add_parser('search', help='Visual search')
    search_parser.add_argument('query', help='Query image path or text')
    search_parser.add_argument('--index', help='Path to search index')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--text', action='store_true', help='Text query')
    search_parser.add_argument('--build-index', help='Build index from directory')
    
    # Configuration
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
    config_parser.add_argument('--save', action='store_true', help='Save configuration')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        run_object_detection(args)
    elif args.command == 'classify':
        run_image_classification(args)
    elif args.command == 'deepfake':
        run_deepfake_detection(args)
    elif args.command == 'generate-video':
        run_video_generation(args)
    elif args.command == 'process-video':
        run_video_processing(args)
    elif args.command == 'search':
        run_visual_search(args)
    elif args.command == 'config':
        run_config_management(args)
    else:
        parser.print_help()

def run_object_detection(args):
    print(f"Running object detection on {args.image}")
    
    detector = ObjectDetector(model_name=args.model, confidence_threshold=args.confidence)
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    detections = detector.detect(image)
    
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
    
    if args.output:
        result_image = detector.draw_detections(image, detections)
        cv2.imwrite(args.output, result_image)
        print(f"Result saved to {args.output}")

def run_image_classification(args):
    print(f"Classifying image {args.image}")
    
    classifier = ImageClassifier(model_name=args.model)
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    predictions = classifier.classify(image, top_k=args.top_k)
    
    print(f"Top {args.top_k} predictions:")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}. {pred['class_name']}: {pred['confidence']:.4f}")

def run_deepfake_detection(args):
    print(f"Analyzing {args.image} for deepfake detection")
    
    detector = DeepfakeDetector(confidence_threshold=args.threshold)
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    result = detector.detect(image)
    
    print(f"Deepfake detection result:")
    print(f"  Is deepfake: {result['is_deepfake']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Real probability: {result['real_probability']:.4f}")
    print(f"  Fake probability: {result['fake_probability']:.4f}")
    
    if args.heatmap:
        heatmap = detector.generate_heatmap(image)
        cv2.imwrite(args.heatmap, heatmap)
        print(f"Heatmap saved to {args.heatmap}")

def run_video_generation(args):
    print(f"Generating video from {args.input}")
    
    generator = VideoGenerator()
    
    if args.text:
        output_path = generator.generate_from_text(
            args.input,
            duration=args.duration,
            fps=args.fps,
            output_path=args.output
        )
    else:
        output_path = generator.generate_from_image(
            args.input,
            duration=args.duration,
            fps=args.fps,
            output_path=args.output
        )
    
    print(f"Video generated: {output_path}")

def run_video_processing(args):
    print(f"Processing video {args.input}")
    
    processor = VideoProcessor()
    
    stats = processor.process_video(
        args.input,
        args.output,
        stabilize=args.stabilize,
        enhance=args.enhance,
        interpolate=args.interpolate,
        target_fps=args.target_fps
    )
    
    print(f"Video processing completed:")
    print(f"  Input: {stats['input_frames']} frames at {stats['input_fps']} FPS")
    print(f"  Output: {stats['output_frames']} frames at {stats['output_fps']} FPS")
    print(f"  Processing steps: {', '.join(stats['processing_steps'])}")

def run_visual_search(args):
    if args.build_index:
        print(f"Building search index from {args.build_index}")
        search_engine = VisualSearch()
        
        # Collect images from directory
        image_paths = []
        for root, dirs, files in os.walk(args.build_index):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images")
        search_engine.add_images(image_paths)
        
        if args.index:
            search_engine.save_index(args.index)
            print(f"Index saved to {args.index}")
        
        return
    
    if not args.index:
        print("Error: --index parameter is required for search")
        return
    
    print(f"Loading search index from {args.index}")
    search_engine = VisualSearch()
    search_engine.load_index(args.index)
    
    if args.text:
        results = search_engine.search_by_text(args.query, top_k=args.top_k)
        print(f"Text search results for '{args.query}':")
    else:
        results = search_engine.search_by_image(args.query, top_k=args.top_k)
        print(f"Image search results for '{args.query}':")
    
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['image_path']}")
        if 'similarity_score' in result:
            print(f"      Similarity: {result['similarity_score']:.4f}")

def run_config_management(args):
    config = Config()
    
    if args.show:
        print("Current configuration:")
        for key, value in config.config.items():
            print(f"  {key}: {value}")
    
    if args.set:
        key, value = args.set
        # Try to convert to appropriate type
        try:
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
        
        config.update(key, value)
        print(f"Set {key} = {value}")
    
    if args.save:
        config.save()
        print("Configuration saved")

if __name__ == '__main__':
    main()