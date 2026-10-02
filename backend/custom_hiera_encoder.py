"""
Custom Hiera-B Encoder with Domain-Specific Preprocessing and Augmentation
Implements:
- Resize to 1024×1024 with aspect-preserving padding
- ImageNet normalization
- Frame extraction at 10 FPS
- Optical flow interpolation for corrupted frames
- Spatial augmentations (flip, rotation, scale)
- Temporal augmentations (frame dropout, temporal jitter)
- Appearance/motion augmentations (brightness, contrast, HSV, motion blur)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import SAM2 Hiera encoder base (optional - will work without SAM2 installed for preprocessing only)
try:
    from sam2.modeling.backbones.hieradet import HieraBlock, PatchEmbed
    from sam2.modeling.backbones.utils import window_partition, window_unpartition
    HIERA_AVAILABLE = True
except ImportError:
    logger.warning("SAM2 Hiera not available, using fallback")
    HIERA_AVAILABLE = False
    # Define dummy classes to avoid errors
    HieraBlock = None
    PatchEmbed = None


class OpticalFlowInterpolator:
    """Interpolate corrupted frames using optical flow"""
    
    def __init__(self, method: str = "farneback"):
        """
        Args:
            method: Optical flow method ('farneback' or 'lucas_kanade')
        """
        self.method = method
    
    def interpolate_frame(
        self, 
        prev_frame: np.ndarray, 
        next_frame: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate corrupted frame using optical flow
        
        Args:
            prev_frame: Previous frame (H, W, C)
            next_frame: Next frame (H, W, C)
        
        Returns:
            Interpolated frame
        """
        if self.method == "farneback":
            return self._farneback_interpolation(prev_frame, next_frame)
        else:
            return self._lucas_kanade_interpolation(prev_frame, next_frame)
    
    def _farneback_interpolation(
        self, 
        prev_frame: np.ndarray, 
        next_frame: np.ndarray
    ) -> np.ndarray:
        """Use Farneback dense optical flow"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY) if len(next_frame.shape) == 3 else next_frame
        
        # Compute forward and backward flow
        flow_forward = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Interpolate at midpoint
        h, w = prev_gray.shape
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Warp both frames towards center
        flow_half = flow_forward * 0.5
        x1 = x + flow_half[:, :, 0]
        y1 = y + flow_half[:, :, 1]
        
        # Remap and blend
        map_x = x1.astype(np.float32)
        map_y = y1.astype(np.float32)
        
        warped_prev = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
        warped_next = cv2.remap(next_frame, -flow_half[:, :, 0] + x, -flow_half[:, :, 1] + y, cv2.INTER_LINEAR)
        
        # Blend warped frames
        interpolated = 0.5 * warped_prev + 0.5 * warped_next
        
        return interpolated.astype(np.uint8)
    
    def _lucas_kanade_interpolation(
        self, 
        prev_frame: np.ndarray, 
        next_frame: np.ndarray
    ) -> np.ndarray:
        """Use Lucas-Kanade sparse optical flow"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY) if len(next_frame.shape) == 3 else next_frame
        
        # Feature detection
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        
        if corners is None or len(corners) < 4:
            # Fallback to simple averaging
            return (0.5 * prev_frame + 0.5 * next_frame).astype(np.uint8)
        
        # Calculate optical flow
        flow, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, corners, None)
        
        # Select good points
        good = status.ravel() == 1
        if good.sum() < 4:
            return (0.5 * prev_frame + 0.5 * next_frame).astype(np.uint8)
        
        prev_pts = corners[good]
        next_pts = flow[good]
        
        # Estimate global transformation
        transform, _ = cv2.estimateAffinePartial2D(prev_pts, next_pts)
        
        if transform is None:
            return (0.5 * prev_frame + 0.5 * next_frame).astype(np.uint8)
        
        # Apply half transformation
        transform_half = transform * 0.5
        transform_half[0, 2] = transform[0, 2] * 0.5
        transform_half[1, 2] = transform[1, 2] * 0.5
        
        h, w = prev_frame.shape[:2]
        warped = cv2.warpAffine(prev_frame, transform_half, (w, h))
        
        # Blend with next frame
        interpolated = 0.5 * warped + 0.5 * next_frame
        
        return interpolated.astype(np.uint8)


class DomainSpecificPreprocessor:
    """
    Domain-specific preprocessing for video segmentation
    - Resize to 1024×1024 with aspect-preserving padding
    - ImageNet normalization
    - Frame extraction at 10 FPS
    - Optical flow interpolation for corrupted frames
    """
    
    def __init__(
        self,
        target_size: int = 1024,
        fps: int = 10,
        normalize: bool = True,
        interpolate_corrupted: bool = True
    ):
        """
        Args:
            target_size: Target size for resizing (default 1024)
            fps: Frames per second for extraction
            normalize: Whether to normalize with ImageNet stats
            interpolate_corrupted: Whether to interpolate corrupted frames
        """
        self.target_size = target_size
        self.fps = fps
        self.normalize = normalize
        self.interpolate_corrupted = interpolate_corrupted
        
        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if interpolate_corrupted:
            self.flow_interpolator = OpticalFlowInterpolator()
    
    def resize_with_padding(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Resize image to target_size with aspect-preserving padding
        
        Args:
            image: Input image (H, W, C) in RGB format
        
        Returns:
            Resized image and metadata (scale, padding)
        """
        h, w = image.shape[:2]
        
        # Calculate scale
        scale = min(self.target_size / h, self.target_size / w)
        
        # Resize
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=image.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        metadata = {
            'scale': scale,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }
        
        return padded, metadata
    
    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image with ImageNet statistics
        
        Args:
            image: Tensor of shape (B, C, H, W) or (C, H, W) in [0, 1]
        
        Returns:
            Normalized tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        
        normalized = (image - mean) / std
        
        if image.dim() == 3:
            normalized = normalized.squeeze(0)
        
        return normalized
    
    def preprocess_frame(
        self, 
        frame: np.ndarray,
        is_corrupted: bool = False,
        prev_frame: Optional[np.ndarray] = None,
        next_frame: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess single frame
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            is_corrupted: Whether frame is corrupted
            prev_frame: Previous frame for interpolation
            next_frame: Next frame for interpolation
        
        Returns:
            Preprocessed tensor (C, H, W) and metadata
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Interpolate corrupted frames
        if is_corrupted and self.interpolate_corrupted:
            if prev_frame is not None and next_frame is not None:
                prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB) if len(prev_frame.shape) == 3 else prev_frame
                next_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB) if len(next_frame.shape) == 3 else next_frame
                frame_rgb = self.flow_interpolator.interpolate_frame(prev_rgb, next_rgb)
        
        # Resize with padding
        resized, metadata = self.resize_with_padding(frame_rgb)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        
        if self.normalize:
            tensor = self.normalize_image(tensor)
        
        return tensor, metadata
    
    def extract_frames(
        self, 
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames


class DomainSpecificAugmentation:
    """
    Domain-specific augmentations for video segmentation
    - Spatial: Random flip, rotation ±15°, scale 0.8–1.2
    - Temporal: Frame dropout p=0.1, temporal jitter ±2 frames
    - Appearance/motion: Brightness/contrast ±0.2, HSV jitter, Gaussian motion blur
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        rotate_range: Tuple[float, float] = (-15, 15),
        scale_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        frame_dropout_prob: float = 0.1,
        temporal_jitter_range: Tuple[int, int] = (-2, 2),
        brightness_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_prob: float = 0.5,
        contrast_range: Tuple[float, float] = (-0.2, 0.2),
        hsv_prob: float = 0.5,
        motion_blur_prob: float = 0.3,
        motion_blur_kernel_range: Tuple[int, int] = (3, 7)
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.frame_dropout_prob = frame_dropout_prob
        self.temporal_jitter_range = temporal_jitter_range
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.hsv_prob = hsv_prob
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_range = motion_blur_kernel_range
    
    def spatial_augment(
        self, 
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply spatial augmentations"""
        h, w = frame.shape[:2]
        
        # Random flip
        if np.random.rand() < self.flip_prob:
            frame = cv2.flip(frame, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        
        # Random rotation
        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(*self.rotate_range)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        
        # Random scale (with crop/pad)
        if np.random.rand() < self.scale_prob:
            scale = np.random.uniform(*self.scale_range)
            new_h, new_w = int(h * scale), int(w * scale)
            frame_scaled = cv2.resize(frame, (new_w, new_h))
            if mask is not None:
                mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop center
                y1 = (new_h - h) // 2
                x1 = (new_w - w) // 2
                frame = frame_scaled[y1:y1+h, x1:x1+w]
                if mask is not None:
                    mask = mask_scaled[y1:y1+h, x1:x1+w]
            else:
                # Pad
                y_pad = (h - new_h) // 2
                x_pad = (w - new_w) // 2
                frame = cv2.copyMakeBorder(frame_scaled, y_pad, h-new_h-y_pad, 
                                         x_pad, w-new_w-x_pad, cv2.BORDER_REFLECT)
                if mask is not None:
                    mask = cv2.copyMakeBorder(mask_scaled, y_pad, h-new_h-y_pad,
                                            x_pad, w-new_w-x_pad, cv2.BORDER_REFLECT)
        
        return frame, mask
    
    def appearance_augment(self, frame: np.ndarray) -> np.ndarray:
        """Apply appearance augmentations"""
        # Brightness
        if np.random.rand() < self.brightness_prob:
            delta = np.random.uniform(*self.brightness_range)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=delta*255)
        
        # Contrast
        if np.random.rand() < self.contrast_prob:
            alpha = 1.0 + np.random.uniform(*self.contrast_range)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        
        # HSV jitter
        if np.random.rand() < self.hsv_prob:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Motion blur
        if np.random.rand() < self.motion_blur_prob:
            kernel_size = np.random.choice([k for k in range(self.motion_blur_kernel_range[0], 
                                                           self.motion_blur_kernel_range[1]+1, 2)])
            angle = np.random.uniform(0, 360)
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # Rotate kernel
            M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            
            frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    def temporal_augment(
        self, 
        frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply temporal augmentations"""
        augmented = []
        
        for i, frame in enumerate(frames):
            # Frame dropout
            if np.random.rand() < self.frame_dropout_prob:
                continue  # Skip this frame
            
            # Temporal jitter
            jitter = np.random.randint(*self.temporal_jitter_range)
            jittered_idx = np.clip(i + jitter, 0, len(frames) - 1)
            
            if jittered_idx != i:
                frame = frames[jittered_idx]
            
            augmented.append(frame)
        
        return augmented if augmented else frames
    
    def augment(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        apply_spatial: bool = True,
        apply_appearance: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply all augmentations
        
        Args:
            frame: Input frame (H, W, C)
            mask: Input mask (H, W) optional
            apply_spatial: Whether to apply spatial augmentations
            apply_appearance: Whether to apply appearance augmentations
        
        Returns:
            Augmented frame and mask
        """
        if apply_spatial:
            frame, mask = self.spatial_augment(frame, mask)
        
        if apply_appearance:
            frame = self.appearance_augment(frame)
        
        return frame, mask


# Custom Hiera-B Encoder wrapper
class CustomHieraBEncoder(nn.Module):
    """
    Custom Hiera-B encoder with domain-specific preprocessing
    Wraps the base Hiera encoder with preprocessing pipeline
    """
    
    def __init__(
        self,
        base_encoder: Optional[nn.Module] = None,
        preprocessor: Optional[DomainSpecificPreprocessor] = None,
        augmenter: Optional[DomainSpecificAugmentation] = None
    ):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.preprocessor = preprocessor or DomainSpecificPreprocessor()
        self.augmenter = augmenter
        
        if base_encoder is None and HIERA_AVAILABLE:
            # Initialize base Hiera encoder here if needed
            # This is a placeholder - actual initialization would depend on SAM2 config
            logger.warning("Base encoder not provided, preprocessing only mode")
    
    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor (B, C, H, W) or list of frames
            is_training: Whether in training mode
        
        Returns:
            Dictionary with features and metadata
        """
        # Preprocessing is typically done before passing to model
        # This is mainly for integration
        
        if self.base_encoder is not None:
            features = self.base_encoder(x)
            return features
        else:
            # Fallback: return input
            return {"features": x}

