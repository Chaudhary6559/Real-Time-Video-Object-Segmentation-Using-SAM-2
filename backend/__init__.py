"""
Backend modules for SAM2 Video Segmentation
"""

# Patch video loader first (before importing other modules)
try:
    from backend.video_loader import patch_sam2_video_loader
    patch_sam2_video_loader()
except Exception:
    pass  # Fail silently if patch doesn't work

from backend.sam2_model import SAM2Model
from backend.preprocessing import VideoProcessor, DataAugmentation
from backend.metrics import MetricsCalculator
from backend.training import ModelTrainer

__all__ = [
    'SAM2Model',
    'VideoProcessor',
    'DataAugmentation',
    'MetricsCalculator',
    'ModelTrainer'
]
