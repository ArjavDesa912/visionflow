#!/usr/bin/env python3
"""
VisionFlow Pro Setup Script

This script helps set up the VisionFlow Pro environment,
download required models, and initialize the system.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"Python version: {sys.version}")
    return True

def check_pytorch():
    """Check if PyTorch is properly installed"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        return True
    except ImportError:
        print("Error: PyTorch is not installed")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'data',
        'cache',
        'output',
        'logs',
        'examples'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def download_models():
    """Download required models"""
    print("Downloading models...")
    
    try:
        # This would typically download models from HuggingFace or other sources
        # For now, we'll just demonstrate the API
        print("Note: Model downloading would happen here in production")
        print("Models to download:")
        print("  - YOLO-v8 for object detection")
        print("  - EfficientNet-B7 for image classification")
        print("  - Vision Transformers for deepfake detection")
        print("  - Stable Video Diffusion for video generation")
        print("  - CLIP for visual search")
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

def setup_configuration():
    """Setup configuration files"""
    print("Setting up configuration...")
    
    # Create default configuration
    config_content = """
# VisionFlow Pro Configuration
models:
  device: auto
  object_detection:
    model: yolov8n
    confidence_threshold: 0.5
  classification:
    model: efficientnet-b0
  deepfake_detection:
    model: vit_base_patch16_224

video:
  frame_rate: 30
  resolution: [640, 480]

search:
  embedding_dim: 512
  index_type: hnsw
"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_content)
    
    print("Configuration file created: config.yaml")

def test_installation():
    """Test if the installation works"""
    print("Testing installation...")
    
    try:
        # Test imports
        from visionflow import ObjectDetector, ImageClassifier, DeepfakeDetector
        from visionflow import VideoGenerator, VideoProcessor, VisualSearch
        print("✓ All imports successful")
        
        # Test basic functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test object detection
        detector = ObjectDetector()
        print("✓ Object detector initialized")
        
        # Test image classification
        classifier = ImageClassifier()
        print("✓ Image classifier initialized")
        
        # Test deepfake detection
        deepfake_detector = DeepfakeDetector()
        print("✓ Deepfake detector initialized")
        
        return True
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("VisionFlow Pro Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        return False
    
    if not check_pytorch():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Setup configuration
    setup_configuration()
    
    # Download models
    if not download_models():
        print("Warning: Model download failed, but setup can continue")
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print("✓ VisionFlow Pro setup completed successfully!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run 'python examples/demo.py' to see examples")
        print("2. Check the documentation for usage instructions")
        print("3. Use 'python -m visionflow.cli --help' for CLI commands")
        return True
    else:
        print("\n" + "=" * 50)
        print("✗ Setup completed with errors")
        print("=" * 50)
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)