import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import os
from tqdm import tqdm
from ..utils.config import config

class VideoProcessor:
    """Advanced video processing with frame interpolation, stabilization, and enhancement"""
    
    def __init__(self):
        self.device = config.get('models.device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.frame_rate = config.get('video.frame_rate', 30)
        self.resolution = config.get('video.resolution', [1920, 1080])
        self.batch_size = config.get('video.batch_size', 8)
        
        # Initialize processing parameters
        self.stabilization_enabled = True
        self.enhancement_enabled = True
        self.noise_reduction_enabled = True
    
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Load video file and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, video_info)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video info
        video_info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        return frames, video_info
    
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = None):
        """
        Save frames as video file
        
        Args:
            frames: List of frames
            output_path: Output video file path
            fps: Frames per second
        """
        fps = fps or self.frame_rate
        
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def stabilize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Stabilize video frames
        
        Args:
            frames: Input frames
            
        Returns:
            Stabilized frames
        """
        if len(frames) < 2:
            return frames
        
        # Convert frames to grayscale for stabilization
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Initialize stabilizer
        stabilized_frames = []
        prev_gray = gray_frames[0]
        stabilized_frames.append(frames[0].copy())
        
        # Calculate transformations
        transforms = []
        for i in range(1, len(gray_frames)):
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Estimate affine transformation
            prev_pts = np.array([[x, y] for y in range(0, gray_frames[i].shape[0], 10) 
                               for x in range(0, gray_frames[i].shape[1], 10)], dtype=np.float32)
            
            curr_pts = prev_pts + flow[prev_pts[:, 1], prev_pts[:, 0]]
            
            transform_matrix = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            transforms.append(transform_matrix)
            
            prev_gray = gray_frames[i]
        
        # Apply smoothing to transformations
        smooth_transforms = self._smooth_transforms(transforms)
        
        # Apply transformations
        for i, (frame, transform) in enumerate(zip(frames[1:], smooth_transforms)):
            if transform is not None:
                stabilized_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))
                stabilized_frames.append(stabilized_frame)
            else:
                stabilized_frames.append(frame)
        
        return stabilized_frames
    
    def _smooth_transforms(self, transforms: List[np.ndarray]) -> List[np.ndarray]:
        """Apply smoothing to transformation matrices"""
        if len(transforms) < 3:
            return transforms
        
        smooth_transforms = []
        for i in range(len(transforms)):
            if i == 0 or i == len(transforms) - 1:
                smooth_transforms.append(transforms[i])
            else:
                # Moving average smoothing
                smooth_transform = (transforms[i-1] + transforms[i] + transforms[i+1]) / 3
                smooth_transforms.append(smooth_transform)
        
        return smooth_transforms
    
    def enhance_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance video frames with super-resolution and noise reduction
        
        Args:
            frames: Input frames
            
        Returns:
            Enhanced frames
        """
        enhanced_frames = []
        
        for frame in tqdm(frames, desc="Enhancing frames"):
            enhanced_frame = frame.copy()
            
            # Apply noise reduction
            if self.noise_reduction_enabled:
                enhanced_frame = cv2.fastNlMeansDenoisingColored(enhanced_frame, None, 10, 10, 7, 21)
            
            # Apply contrast enhancement
            lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0).apply(l)
            enhanced_frame = cv2.merge([l, a, b])
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
            
            # Apply sharpening
            enhanced_frame = self._sharpen_frame(enhanced_frame)
            
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def _sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening to frame"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    def interpolate_frames(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        """
        Interpolate frames to achieve higher frame rate
        
        Args:
            frames: Input frames
            target_fps: Target frames per second
            
        Returns:
            Interpolated frames
        """
        if len(frames) < 2:
            return frames
        
        original_fps = self.frame_rate
        if target_fps <= original_fps:
            return frames
        
        interpolation_factor = target_fps // original_fps
        
        interpolated_frames = [frames[0]]
        
        for i in range(len(frames) - 1):
            interpolated_frames.append(frames[i])
            
            for j in range(interpolation_factor - 1):
                alpha = (j + 1) / interpolation_factor
                interpolated_frame = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
                
                # Apply motion-aware interpolation
                interpolated_frame = self._motion_aware_interpolation(frames[i], frames[i + 1], alpha)
                interpolated_frames.append(interpolated_frame)
        
        interpolated_frames.append(frames[-1])
        return interpolated_frames
    
    def _motion_aware_interpolation(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Apply motion-aware frame interpolation"""
        # Calculate optical flow
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Create meshgrid for warping
        h, w = frame1.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate intermediate positions
        coords = np.stack([x_coords + alpha * flow[..., 0], y_coords + alpha * flow[..., 1]], axis=-1)
        
        # Warp frame
        interpolated_frame = cv2.remap(frame1, coords.astype(np.float32), None, cv2.INTER_LINEAR)
        
        return interpolated_frame
    
    def detect_scenes(self, frames: List[np.ndarray]) -> List[int]:
        """
        Detect scene changes in video
        
        Args:
            frames: Input frames
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) < 2:
            return []
        
        scene_changes = []
        prev_hist = self._calculate_histogram(frames[0])
        
        for i in range(1, len(frames)):
            curr_hist = self._calculate_histogram(frames[i])
            
            # Calculate histogram difference
            diff = np.sum(np.abs(prev_hist - curr_hist))
            
            # Threshold for scene change
            if diff > 0.3:  # Adjust threshold as needed
                scene_changes.append(i)
            
            prev_hist = curr_hist
        
        return scene_changes
    
    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame"""
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist / hist.sum()
        return hist.flatten()
    
    def extract_keyframes(self, frames: List[np.ndarray], method: str = 'uniform') -> List[int]:
        """
        Extract keyframes from video
        
        Args:
            frames: Input frames
            method: Method for keyframe extraction ('uniform', 'motion', 'scene')
            
        Returns:
            List of keyframe indices
        """
        if method == 'uniform':
            # Uniform sampling
            num_keyframes = max(1, len(frames) // 30)  # 1 keyframe per 30 frames
            step = len(frames) // num_keyframes
            return list(range(0, len(frames), step))
        
        elif method == 'motion':
            # Motion-based keyframe extraction
            return self._motion_based_keyframes(frames)
        
        elif method == 'scene':
            # Scene-based keyframe extraction
            scene_changes = self.detect_scenes(frames)
            keyframes = [0] + scene_changes
            if keyframes[-1] != len(frames) - 1:
                keyframes.append(len(frames) - 1)
            return keyframes
        
        else:
            raise ValueError(f"Unknown keyframe extraction method: {method}")
    
    def _motion_based_keyframes(self, frames: List[np.ndarray]) -> List[int]:
        """Extract keyframes based on motion content"""
        if len(frames) < 3:
            return list(range(len(frames)))
        
        motion_scores = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_score = np.mean(np.abs(flow))
            motion_scores.append(motion_score)
        
        # Select frames with high motion
        threshold = np.mean(motion_scores) + np.std(motion_scores)
        keyframes = [0]
        
        for i, score in enumerate(motion_scores):
            if score > threshold:
                keyframes.append(i + 1)
        
        if keyframes[-1] != len(frames) - 1:
            keyframes.append(len(frames) - 1)
        
        return keyframes
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     stabilize: bool = True,
                     enhance: bool = True,
                     interpolate: bool = False,
                     target_fps: int = None) -> Dict:
        """
        Process video with multiple enhancement options
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            stabilize: Apply video stabilization
            enhance: Apply frame enhancement
            interpolate: Apply frame interpolation
            target_fps: Target frame rate for interpolation
            
        Returns:
            Processing statistics
        """
        # Load video
        frames, video_info = self.load_video(input_path)
        
        stats = {
            'input_frames': len(frames),
            'input_fps': video_info['fps'],
            'input_resolution': (video_info['width'], video_info['height']),
            'processing_steps': []
        }
        
        # Apply stabilization
        if stabilize and self.stabilization_enabled:
            frames = self.stabilize_video(frames)
            stats['processing_steps'].append('stabilization')
        
        # Apply enhancement
        if enhance and self.enhancement_enabled:
            frames = self.enhance_frames(frames)
            stats['processing_steps'].append('enhancement')
        
        # Apply interpolation
        if interpolate and target_fps and target_fps > video_info['fps']:
            frames = self.interpolate_frames(frames, target_fps)
            stats['processing_steps'].append('interpolation')
            stats['output_fps'] = target_fps
        else:
            stats['output_fps'] = video_info['fps']
        
        # Save processed video
        self.save_video(frames, output_path, stats['output_fps'])
        
        stats['output_frames'] = len(frames)
        stats['output_resolution'] = (frames[0].shape[1], frames[0].shape[0])
        
        return stats