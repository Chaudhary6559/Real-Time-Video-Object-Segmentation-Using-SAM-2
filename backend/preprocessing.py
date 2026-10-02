"""
Video Preprocessing Module
Handles frame extraction, resizing, normalization, and augmentation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import custom preprocessing and augmentation
try:
    from backend.custom_hiera_encoder import (
        DomainSpecificPreprocessor,
        DomainSpecificAugmentation,
        OpticalFlowInterpolator
    )
    CUSTOM_PREPROCESSOR_AVAILABLE = True
except ImportError as e:
    CUSTOM_PREPROCESSOR_AVAILABLE = False
    DomainSpecificPreprocessor = None
    DomainSpecificAugmentation = None
    OpticalFlowInterpolator = None
    logger.warning(f"Custom preprocessing not available, using basic preprocessing: {e}")

class VideoProcessor:
    """
    Process video files for segmentation
    """
    
    def __init__(
        self,
        fps: int = 10,
        target_size: int = 1024,
        normalize: bool = True
    ):
        """
        Initialize video processor
        
        Args:
            fps: Target frames per second for extraction
            target_size: Target size for resizing (square)
            normalize: Whether to normalize frames
        """
        self.fps = fps
        self.target_size = target_size
        self.normalize = normalize
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of frames
        """
        try:
            logger.info(f"Extracting frames from {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / self.fps))
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every nth frame
                if frame_count % frame_interval == 0:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    frames.append(processed_frame)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {extracted_count} frames from {total_frames} total frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess single frame
        
        Args:
            frame: Input frame
        
        Returns:
            Preprocessed frame
        """
        # Resize with aspect-preserving padding
        frame = self._resize_with_padding(frame, self.target_size)
        
        # Normalize
        if self.normalize:
            frame = self._normalize_frame(frame)
        
        return frame
    
    def _resize_with_padding(
        self,
        frame: np.ndarray,
        target_size: int
    ) -> np.ndarray:
        """
        Resize frame with aspect-preserving padding
        
        Args:
            frame: Input frame
            target_size: Target size
        
        Returns:
            Resized frame with padding
        """
        h, w = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / h, target_size / w)
        
        # Resize
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to target size
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = np.zeros((target_size, target_size, 3), dtype=frame.dtype)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame with ImageNet statistics
        
        Args:
            frame: Input frame
        
        Returns:
            Normalized frame
        """
        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        frame = (frame - self.mean) / self.std
        
        return frame
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video info
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}


class DataAugmentation:
    """
    Data augmentation for training
    Uses custom domain-specific augmentation if available
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        brightness_prob: float = 0.2,
        contrast_prob: float = 0.2,
        hsv_prob: float = 0.2,
        use_domain_specific: bool = True
    ):
        """
        Initialize augmentation
        
        Args:
            flip_prob: Probability of horizontal flip
            rotate_prob: Probability of rotation
            brightness_prob: Probability of brightness adjustment
            contrast_prob: Probability of contrast adjustment
            hsv_prob: Probability of HSV adjustment
            use_domain_specific: Use domain-specific augmentation if available
        """
        self.use_domain_specific = use_domain_specific and CUSTOM_PREPROCESSOR_AVAILABLE
        
        if self.use_domain_specific:
            self.augmenter = DomainSpecificAugmentation(
                flip_prob=flip_prob,
                rotate_prob=rotate_prob,
                brightness_prob=brightness_prob,
                contrast_prob=contrast_prob,
                hsv_prob=hsv_prob
            )
        else:
            self.flip_prob = flip_prob
            self.rotate_prob = rotate_prob
            self.brightness_prob = brightness_prob
            self.contrast_prob = contrast_prob
            self.hsv_prob = hsv_prob
    
    def augment(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple:
        """
        Apply augmentation to frame and mask
        
        Args:
            frame: Input frame
            mask: Input mask (optional)
        
        Returns:
            Augmented frame and mask
        """
        if self.use_domain_specific:
            return self.augmenter.augment(frame, mask, apply_spatial=True, apply_appearance=True)
        
        # Fallback to basic augmentation
        # Spatial augmentations
        if np.random.rand() < self.flip_prob:
            frame = cv2.flip(frame, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        
        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(-15, 15)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h))
        
        # Appearance augmentations
        if np.random.rand() < self.brightness_prob:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
        
        if np.random.rand() < self.contrast_prob:
            contrast_factor = np.random.uniform(0.8, 1.2)
            frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        
        if np.random.rand() < self.hsv_prob:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return frame, mask
    
    def augment_batch(
        self,
        frames: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None
    ) -> Tuple[List, Optional[List]]:
        """
        Apply augmentation to batch of frames
        
        Args:
            frames: List of frames
            masks: List of masks (optional)
        
        Returns:
            Augmented frames and masks
        """
        augmented_frames = []
        augmented_masks = [] if masks else None
        
        for i, frame in enumerate(frames):
            mask = masks[i] if masks else None
            aug_frame, aug_mask = self.augment(frame, mask)
            augmented_frames.append(aug_frame)
            
            if masks:
                augmented_masks.append(aug_mask)
        
        return augmented_frames, augmented_masks
