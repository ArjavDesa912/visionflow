import os
import yaml
import torch
from typing import Dict, Any, Optional

class Config:
    """Configuration management for VisionFlow Pro"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.yaml')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'object_detection': {
                    'model': 'yolov8x',
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4
                },
                'classification': {
                    'model': 'efficientnet-b7',
                    'num_classes': 1000
                },
                'deepfake_detection': {
                    'model': 'vit-base',
                    'confidence_threshold': 0.8
                }
            },
            'video': {
                'frame_rate': 30,
                'resolution': [1920, 1080],
                'batch_size': 8,
                'temporal_consistency': True
            },
            'search': {
                'embedding_dim': 512,
                'index_type': 'hnsw',
                'ef_construction': 100,
                'ef_search': 50,
                'max_connections': 32
            },
            'paths': {
                'models_dir': './models',
                'data_dir': './data',
                'cache_dir': './cache',
                'output_dir': './output'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Global configuration instance
config = Config()