import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import os
from tqdm import tqdm
from ..utils.config import config

class VideoGenerator:
    """Video generation using Stable Video Diffusion with temporal consistency"""
    
    def __init__(self, model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt"):
        self.model_name = model_name
        self.device = config.get('models.device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.frame_rate = config.get('video.frame_rate', 30)
        self.resolution = config.get('video.resolution', [1920, 1080])
        self.batch_size = config.get('video.batch_size', 8)
        self.temporal_consistency = config.get('video.temporal_consistency', True)
        
        self.pipeline = self._load_pipeline()
        
    def _load_pipeline(self):
        """Load Stable Video Diffusion pipeline"""
        try:
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                variant="fp16" if self.device == 'cuda' else None
            )
            pipeline = pipeline.to(self.device)
            
            if self.device == 'cuda':
                pipeline.enable_model_cpu_offload()
                pipeline.enable_vae_slicing()
            
            return pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to load video generation pipeline: {e}")
    
    def generate_from_image(self, 
                          image: Union[str, np.ndarray, Image.Image],
                          duration: float = 2.0,
                          fps: int = None,
                          num_inference_steps: int = 25,
                          guidance_scale: float = 7.0,
                          output_path: str = None) -> str:
        """
        Generate video from a single image
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            duration: Video duration in seconds
            fps: Frames per second
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            output_path: Output video file path
            
        Returns:
            Path to generated video file
        """
        fps = fps or self.frame_rate
        num_frames = int(duration * fps)
        
        # Load and preprocess image
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize image to target resolution
        image = image.resize(self.resolution)
        
        # Generate video frames
        with torch.no_grad():
            frames = self.pipeline(
                image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).frames
        
        # Apply temporal consistency if enabled
        if self.temporal_consistency:
            frames = self._apply_temporal_consistency(frames)
        
        # Save video
        if output_path is None:
            output_path = f"generated_video_{int(time.time())}.mp4"
        
        self._save_video(frames, output_path, fps)
        
        return output_path
    
    def generate_from_text(self,
                          prompt: str,
                          duration: float = 2.0,
                          fps: int = None,
                          num_inference_steps: int = 25,
                          guidance_scale: float = 7.0,
                          output_path: str = None) -> str:
        """
        Generate video from text prompt (requires text-to-image first)
        
        Args:
            prompt: Text description of video content
            duration: Video duration in seconds
            fps: Frames per second
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            output_path: Output video file path
            
        Returns:
            Path to generated video file
        """
        # This would require a text-to-image model first
        # For now, we'll create a simple implementation
        from diffusers import StableDiffusionPipeline
        
        # Generate initial image from text
        text2img = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            initial_image = text2img(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=self.resolution[1],
                width=self.resolution[0]
            ).images[0]
        
        # Generate video from the initial image
        return self.generate_from_image(
            initial_image, duration, fps, num_inference_steps, guidance_scale, output_path
        )
    
    def generate_style_transfer(self,
                               source_video: str,
                               style_image: Union[str, np.ndarray, Image.Image],
                               output_path: str = None) -> str:
        """
        Generate style-transferred video
        
        Args:
            source_video: Path to source video
            style_image: Style reference image
            output_path: Output video file path
            
        Returns:
            Path to generated video file
        """
        # Load source video
        cap = cv2.VideoCapture(source_video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Load style image
        if isinstance(style_image, str):
            style_image = load_image(style_image)
        elif isinstance(style_image, np.ndarray):
            style_image = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
        
        # Apply style transfer to each frame
        style_frames = []
        for frame in tqdm(frames, desc="Applying style transfer"):
            styled_frame = self._apply_style_transfer(frame, style_image)
            style_frames.append(styled_frame)
        
        # Save video
        if output_path is None:
            output_path = f"style_transfer_{int(time.time())}.mp4"
        
        self._save_video(style_frames, output_path, fps)
        
        return output_path
    
    def _apply_temporal_consistency(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Apply temporal consistency to video frames"""
        if len(frames) < 3:
            return frames
        
        consistent_frames = [frames[0]]
        
        for i in range(1, len(frames) - 1):
            prev_frame = np.array(consistent_frames[-1])
            curr_frame = np.array(frames[i])
            next_frame = np.array(frames[i + 1])
            
            # Apply optical flow for temporal smoothing
            consistent_frame = self._optical_flow_smoothing(prev_frame, curr_frame, next_frame)
            consistent_frames.append(Image.fromarray(consistent_frame))
        
        consistent_frames.append(frames[-1])
        return consistent_frames
    
    def _optical_flow_smoothing(self, prev_frame: np.ndarray, curr_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """Apply optical flow-based temporal smoothing"""
        # Convert to grayscale for optical flow
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow
        flow_forward = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_backward = cv2.calcOpticalFlowFarneback(next_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Smooth frame based on optical flow
        h, w = curr_frame.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Forward warp
        coords_forward = np.stack([x_coords + flow_forward[..., 0], y_coords + flow_forward[..., 1]], axis=-1)
        warped_forward = cv2.remap(prev_frame, coords_forward.astype(np.float32), None, cv2.INTER_LINEAR)
        
        # Backward warp
        coords_backward = np.stack([x_coords + flow_backward[..., 0], y_coords + flow_backward[..., 1]], axis=-1)
        warped_backward = cv2.remap(next_frame, coords_backward.astype(np.float32), None, cv2.INTER_LINEAR)
        
        # Blend frames
        alpha = 0.3
        smoothed_frame = (1 - alpha) * curr_frame + alpha * (warped_forward + warped_backward) / 2
        
        return smoothed_frame.astype(np.uint8)
    
    def _apply_style_transfer(self, content_frame: np.ndarray, style_image: Image.Image) -> np.ndarray:
        """Apply neural style transfer to frame"""
        # Simple style transfer implementation
        # In practice, you'd use a more sophisticated approach
        style_frame = cv2.cvtColor(content_frame, cv2.COLOR_BGR2LAB)
        style_frame[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(style_frame[:,:,0])
        style_frame = cv2.cvtColor(style_frame, cv2.COLOR_LAB2BGR)
        
        return style_frame
    
    def _save_video(self, frames: List[Union[Image.Image, np.ndarray]], output_path: str, fps: int):
        """Save frames as video file"""
        if isinstance(frames[0], Image.Image):
            frames = [np.array(frame) for frame in frames]
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def interpolate_frames(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        """
        Interpolate frames to achieve target frame rate
        
        Args:
            frames: Input frames
            target_fps: Target frames per second
            
        Returns:
            Interpolated frames
        """
        if len(frames) < 2:
            return frames
        
        interpolated_frames = [frames[0]]
        
        for i in range(len(frames) - 1):
            interpolated_frames.append(frames[i])
            
            # Calculate number of frames to interpolate
            interpolation_factor = target_fps // self.frame_rate - 1
            
            for j in range(interpolation_factor):
                alpha = (j + 1) / (interpolation_factor + 1)
                interpolated_frame = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
                interpolated_frames.append(interpolated_frame)
        
        interpolated_frames.append(frames[-1])
        return interpolated_frames