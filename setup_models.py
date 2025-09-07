#!/usr/bin/env python3
"""
VisionFlow Pro Model Setup Script

This script downloads and initializes the required models for VisionFlow Pro.
"""

import os
import sys
import torch
from pathlib import Path

def setup_models():
    """Download and setup required models"""
    print("VisionFlow Pro Model Setup")
    print("=" * 50)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # List of models to download
    models_to_download = [
        ('YOLO-v8', 'yolov8n', 'Object detection'),
        ('EfficientNet', 'efficientnet-b0', 'Image classification'),
        ('Vision Transformer', 'vit_base_patch16_224', 'Deepfake detection'),
        ('CLIP', 'openai/clip-vit-base-patch32', 'Visual search'),
        ('Stable Diffusion', 'runwayml/stable-diffusion-v1-5', 'Image generation')
    ]
    
    print("Models to download:")
    for name, model_id, purpose in models_to_download:
        print(f"  - {name} ({model_id}): {purpose}")
    
    print("\nNote: Models will be downloaded automatically when first used")
    print("This setup script creates the necessary directory structure")
    
    # Create model directories
    for name, model_id, purpose in models_to_download:
        model_dir = models_dir / name.lower().replace(' ', '_')
        model_dir.mkdir(exist_ok=True)
        print(f"Created model directory: {model_dir}")
    
    print("\n" + "=" * 50)
    print("Model setup completed!")
    print("=" * 50)
    print("\nModels will be downloaded automatically when you:")
    print("1. Run object detection")
    print("2. Perform image classification")
    print("3. Use deepfake detection")
    print("4. Conduct visual search")
    print("5. Generate videos")

if __name__ == '__main__':
    setup_models()