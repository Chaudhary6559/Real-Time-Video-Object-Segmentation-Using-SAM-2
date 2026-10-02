"""
Video Loader with OpenCV Fallback
Handles video loading when decord is not available (e.g., on Windows)
"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def load_video_frames_opencv_fallback(
    video_path: str,
    image_size: int = 1024,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    compute_device: torch.device = torch.device("cpu"),
    offload_video_to_cpu: bool = True,
    max_frames: int = 300,  # Limit frames to prevent memory issues
) -> Tuple[torch.Tensor, int, int]:
    """
    Load video frames using OpenCV (fallback when decord is not available)
    
    Args:
        video_path: Path to video file
        image_size: Target image size for resizing
        img_mean: ImageNet mean values
        img_std: ImageNet std values
        compute_device: Device to load frames to
        offload_video_to_cpu: Whether to keep on CPU
    
    Returns:
        Tuple of (images tensor, video_height, video_width)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Loading video: {video_path}")
    logger.info(f"  Resolution: {video_width}x{video_height}, FPS: {fps:.2f}, Frames: {total_frames}")
    
    # Prepare normalization tensors
    img_mean_tensor = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std_tensor = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    
    # Limit maximum frames to prevent memory issues
    MAX_FRAMES = max_frames  # Use parameter or default to 300
    if total_frames > MAX_FRAMES:
        logger.warning(f"Video has {total_frames} frames, limiting to {MAX_FRAMES} for memory efficiency")
        frame_skip = max(1, total_frames // MAX_FRAMES)
    else:
        frame_skip = 1
    
    # Load and process frames
    images = []
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if needed
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize with aspect-preserving padding
        h, w = frame_rgb.shape[:2]
        scale = min(image_size / h, image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = (image_size - new_h) // 2
        pad_w = (image_size - new_w) // 2
        padded = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        # Use float32 directly to avoid memory issues
        frame_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        images.append(frame_tensor)
        processed_count += 1
        
        # Memory safety: if we've processed enough frames, break
        if processed_count >= MAX_FRAMES:
            logger.info(f"Reached frame limit ({MAX_FRAMES}), stopping frame loading")
            break
        
        frame_count += 1
    
    cap.release()
    
    if len(images) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")
    
    # Stack frames: (N, C, H, W) - do this in chunks if needed
    try:
        images_tensor = torch.stack(images, dim=0)
    except RuntimeError as e:
        if "not enough memory" in str(e).lower():
            logger.error(f"Out of memory loading {len(images)} frames. Try a shorter video or reduce image_size.")
            # Try processing in smaller batches
            logger.info("Attempting to process in smaller batches...")
            # Clear memory
            del images
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            raise RuntimeError(
                f"Video too large for available memory. "
                f"Try: 1) Using a shorter video, 2) Reducing resolution, or 3) Installing decord for better memory efficiency."
            )
        raise
    
    # Normalize
    if not offload_video_to_cpu:
        images_tensor = images_tensor.to(compute_device)
        img_mean_tensor = img_mean_tensor.to(compute_device)
        img_std_tensor = img_std_tensor.to(compute_device)
    
    images_tensor = (images_tensor - img_mean_tensor) / img_std_tensor
    
    logger.info(f"Loaded {processed_count} frames (from {frame_count} total), shape: {images_tensor.shape}")
    
    return images_tensor, video_height, video_width


def patch_sam2_video_loader():
    """
    Monkey-patch SAM2's video loader to use OpenCV fallback when decord is not available
    """
    try:
        from sam2.utils import misc
        
        # Store original function (avoid re-patching)
        if not hasattr(misc, '_original_load_video_frames_from_video_file'):
            misc._original_load_video_frames_from_video_file = misc.load_video_frames_from_video_file
        
        original_load_video_frames_from_video_file = misc._original_load_video_frames_from_video_file
        
        def patched_load_video_frames_from_video_file(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean=(0.485, 0.456, 0.406),
            img_std=(0.229, 0.224, 0.225),
            compute_device=torch.device("cuda"),
        ):
            """Patched version that falls back to OpenCV"""
            # Check if decord is available first (before calling original function)
            decord_available = False
            try:
                import decord
                decord_available = True
            except (ImportError, ModuleNotFoundError):
                # decord not available, use OpenCV directly
                logger.info("decord not available, using OpenCV fallback for video loading")
                return load_video_frames_opencv_fallback(
                    video_path, image_size, img_mean, img_std,
                    compute_device, offload_video_to_cpu, max_frames=300
                )
            
            # decord is available, try original loader
            if decord_available:
                try:
                    return original_load_video_frames_from_video_file(
                        video_path, image_size, offload_video_to_cpu,
                        img_mean, img_std, compute_device
                    )
                except Exception as e:
                    # Original loader failed for other reasons, try fallback
                    logger.warning(f"decord loader failed ({str(e)[:200]}), trying OpenCV fallback")
                    return load_video_frames_opencv_fallback(
                        video_path, image_size, img_mean, img_std,
                        compute_device, offload_video_to_cpu, max_frames=300
                    )
        
        # Apply patch
        misc.load_video_frames_from_video_file = patched_load_video_frames_from_video_file
        logger.info("Patched SAM2 video loader with OpenCV fallback")
        return True
        
    except Exception as e:
        logger.warning(f"Could not patch SAM2 video loader: {e}")
        return False

