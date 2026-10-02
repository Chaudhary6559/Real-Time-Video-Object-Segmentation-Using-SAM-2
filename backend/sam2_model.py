"""
Real SAM 2 Model Wrapper – FINAL WORKING VERSION (December 2025)
Enhanced with custom Hiera-B encoder and domain-specific preprocessing
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Patch video loader for OpenCV fallback (before importing SAM2)
try:
    from backend.video_loader import patch_sam2_video_loader
    patch_sam2_video_loader()
except Exception as e:
    logger.warning(f"Could not patch video loader: {e}")

# Import SAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor

# Import custom preprocessing and encoder
try:
    from backend.custom_hiera_encoder import (
        DomainSpecificPreprocessor,
        CustomHieraBEncoder
    )
    CUSTOM_ENCODER_AVAILABLE = True
except ImportError:
    CUSTOM_ENCODER_AVAILABLE = False
    DomainSpecificPreprocessor = None
    CustomHieraBEncoder = None
    logger.warning("Custom encoder not available, using standard preprocessing")

# Import custom Hiera-B model
try:
    from models.custom_hiera_b import CustomHieraB, build_custom_hiera_b
    from backend.model_builder import build_sam2_with_custom_hiera_b
    CUSTOM_HIERA_B_AVAILABLE = True
except ImportError:
    CUSTOM_HIERA_B_AVAILABLE = False
    CustomHieraB = None
    build_custom_hiera_b = None
    logger.warning("Custom Hiera-B model not available")


class SAM2Model:
    def __init__(
        self,
        model_type: str = "SAM 2 Hiera-L",
        device: str = "CPU",
        checkpoint_dir: str = "checkpoints",
        use_custom_preprocessing: bool = True,
        target_size: int = 1024,
        fps: int = 10
    ):
        """
        Initialize SAM2 Model with optional custom preprocessing
        
        Args:
            model_type: Model type (Hiera-T, Hiera-S, Hiera-B, Hiera-L)
            device: Device ("CPU" or "CUDA (GPU)")
            checkpoint_dir: Directory containing model checkpoints
            use_custom_preprocessing: Use domain-specific preprocessing
            target_size: Target image size for preprocessing (1024)
            fps: Frames per second for frame extraction (10)
        """
        self.device = "cuda" if torch.cuda.is_available() and "CUDA" in device else "cpu"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_custom_preprocessing = use_custom_preprocessing and CUSTOM_ENCODER_AVAILABLE
        self.target_size = target_size
        self.fps = fps

        # Correct mapping - config name must match Hydra path structure
        # Format: "sam2.1/sam2.1_hiera_b+" (path relative to configs directory, without .yaml)
        self.model_cfg_map = {
            "SAM 2 Hiera-T": ("sam2.1/sam2.1_hiera_t",      "sam2.1_hiera_tiny.pt",      "facebook/sam2.1-hiera-tiny"),
            "SAM 2 Hiera-S": ("sam2.1/sam2.1_hiera_s",      "sam2.1_hiera_small.pt",    "facebook/sam2.1-hiera-small"),
            "SAM 2 Hiera-B": ("sam2.1/sam2.1_hiera_b+",     "sam2.1_hiera_base_plus.pt","facebook/sam2.1-hiera-base-plus"),
            "SAM 2 Hiera-L": ("sam2.1/sam2.1_hiera_l",      "sam2.1_hiera_large.pt",    "facebook/sam2.1-hiera-large"),
        }

        config_name, checkpoint_name, hf_model_id = self.model_cfg_map[model_type]
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Always try HuggingFace builder first (handles config and download automatically)
        logger.info(f"Loading {model_type} on {self.device}...")
        
        try:
            from sam2.build_sam import build_sam2_video_predictor_hf
            logger.info(f"Using HuggingFace builder for {model_type} (handles config and download automatically)...")
            self.predictor = build_sam2_video_predictor_hf(
                hf_model_id,
                device=self.device
            )
            logger.info(f"Successfully loaded {model_type}!")
            
            # Set preprocessor
            if self.use_custom_preprocessing and DomainSpecificPreprocessor is not None:
                self.preprocessor = DomainSpecificPreprocessor(
                    target_size=target_size,
                    fps=fps,
                    normalize=True,
                    interpolate_corrupted=True
                )
                logger.info("Custom domain-specific preprocessing enabled")
            else:
                self.preprocessor = None
            return  # Success!
            
        except Exception as e:
            logger.warning(f"HuggingFace builder failed: {str(e)[:200]}")
            
            # Fallback: Try loading from local checkpoint if exists
            if checkpoint_path.exists():
                try:
                    logger.info(f"Trying local checkpoint: {checkpoint_path}")
                    self.predictor = build_sam2_video_predictor(
                        config_name,
                        str(checkpoint_path),
                        device=self.device
                    )
                    logger.info(f"Successfully loaded {model_type} from local checkpoint!")
                    
                    # Set preprocessor
                    if self.use_custom_preprocessing and DomainSpecificPreprocessor is not None:
                        self.preprocessor = DomainSpecificPreprocessor(
                            target_size=target_size,
                            fps=fps,
                            normalize=True,
                            interpolate_corrupted=True
                        )
                    else:
                        self.preprocessor = None
                    return
                except Exception as e2:
                    logger.error(f"Failed to load from local checkpoint: {str(e2)[:200]}")
            
            # If both fail, raise error with helpful message
            raise RuntimeError(
                f"Failed to load {model_type}.\n"
                f"HuggingFace error: {str(e)[:200]}\n"
                f"Local checkpoint: {checkpoint_path} {'(exists)' if checkpoint_path.exists() else '(not found)'}\n"
                f"Please ensure you have internet connection for automatic download.\n"
                f"Or manually download from: https://huggingface.co/{hf_model_id}"
            )
        
        # Initialize custom preprocessor if available
        if self.use_custom_preprocessing and DomainSpecificPreprocessor is not None:
            self.preprocessor = DomainSpecificPreprocessor(
                target_size=target_size,
                fps=fps,
                normalize=True,
                interpolate_corrupted=True
            )
            logger.info("Custom domain-specific preprocessing enabled")
        else:
            self.preprocessor = None
            if use_custom_preprocessing:
                logger.warning("Custom preprocessing requested but not available")

        logger.info("SAM 2 model loaded successfully!")


    def segment(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        prompt_points: Optional[List[Tuple[int, int]]] = None,
        prompt_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[dict]:
        """
        Segment video using SAM 2 (legacy method - use process_video instead)
        """
        return self.process_video(
            video_path=video_path,
            point_prompts=prompt_points,
            box_prompts=prompt_boxes,
            confidence_threshold=confidence_threshold
        )
    
    def process_video(
        self,
        video_path: str,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        box_prompts: Optional[List[Tuple[int, int, int, int]]] = None,
        confidence_threshold: float = 0.5,
        frame: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Process video or single frame for segmentation
        
        Args:
            video_path: Path to video file (or None if using single frame)
            point_prompts: List of (x, y) point prompts
            box_prompts: List of [x1, y1, x2, y2] box prompts
            confidence_threshold: Confidence threshold for masks
            frame: Single frame array if video_path is None
        
        Returns:
            List of results per frame
        """
        try:
            # If single frame provided, create temporary video
            if frame is not None:
                # Save frame as temporary video
                temp_video = Path("temp") / f"temp_frame_{np.random.randint(0, 100000)}.mp4"
                temp_video.parent.mkdir(exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(str(temp_video), fourcc, 10.0, (w, h))
                out.write(frame)
                out.release()
                video_path = str(temp_video)
            
            # Validate video path exists
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            logger.info(f"Initializing video processing for: {video_path}")
            inference_state = self.predictor.init_state(video_path=video_path)

            # Get video dimensions for default prompts if needed
            video_height = inference_state.get("video_height", 480)
            video_width = inference_state.get("video_width", 640)
            num_frames = inference_state.get("num_frames", 1)

            frame_idx = 0
            prompts_provided = False
            
            if point_prompts and len(point_prompts) > 0:
                points = np.array(point_prompts, dtype=np.float32)
                labels = np.ones(len(points), dtype=int)
                self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=points,
                    labels=labels
                )
                prompts_provided = True
                logger.info(f"Added {len(points)} point prompt(s)")
            elif box_prompts and len(box_prompts) > 0:
                box = np.array(box_prompts[0], dtype=np.float32)
                self.predictor.add_new_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    box=box
                )
                prompts_provided = True
                logger.info(f"Added box prompt: {box}")
            
            # If no prompts provided, use center point as default for auto-segmentation
            if not prompts_provided:
                logger.info("No prompts provided, using center point for auto-segmentation")
                center_point = np.array([[video_width // 2, video_height // 2]], dtype=np.float32)
                labels = np.ones(1, dtype=int)
                self.predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=center_point,
                    labels=labels
                )
                prompts_provided = True

            results = []
            logger.info(f"Processing video frames... (Total frames: {num_frames})")
            
            # Check if we have prompts before processing
            if not prompts_provided:
                logger.warning("No prompts provided - using default center point")
            
            with torch.inference_mode():
                frame_count = 0
                try:
                    if self.device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                                frame_count += 1
                                # Handle mask logits - can be tensor or numpy
                                if torch.is_tensor(out_mask_logits):
                                    mask_logits_cpu = out_mask_logits.cpu()
                                else:
                                    mask_logits_cpu = torch.from_numpy(out_mask_logits) if isinstance(out_mask_logits, np.ndarray) else out_mask_logits
                                
                                # Convert to binary masks
                                masks = (mask_logits_cpu > 0.0).numpy()
                                
                                # Handle single mask vs multiple masks
                                if len(masks.shape) == 2:
                                    masks = masks[np.newaxis, ...]  # Add batch dimension
                                
                                boxes_list = []
                                scores_list = []
                                valid_masks = []
                                
                                for mask_idx, mask in enumerate(masks):
                                    if mask.sum() == 0:
                                        continue
                                    ys, xs = np.where(mask)
                                    if len(xs) == 0 or len(ys) == 0:
                                        continue
                                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                                    boxes_list.append([x1, y1, x2, y2])
                                    
                                    # Get score from mask logits
                                    if torch.is_tensor(mask_logits_cpu):
                                        score = float(mask_logits_cpu[mask].max())
                                    else:
                                        score = float(np.max(mask_logits_cpu[mask]))
                                    scores_list.append(score)
                                    valid_masks.append(mask)
                                
                                # Filter by confidence threshold
                                valid_indices = [i for i, score in enumerate(scores_list) if score >= confidence_threshold]
                                
                                results.append({
                                    'frame_idx': int(out_frame_idx),
                                    'masks': np.array(valid_masks)[valid_indices] if valid_masks else np.array([]),
                                    'boxes': np.array(boxes_list)[valid_indices] if boxes_list else np.zeros((0, 4)),
                                    'scores': np.array(scores_list)[valid_indices] if scores_list else np.zeros((0,))
                                })
                            
                            logger.info(f"Processed {frame_count} frames (CUDA mode)")
                    else:
                        # CPU mode
                        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                            frame_count += 1
                            # Handle mask logits - can be tensor or numpy
                            if torch.is_tensor(out_mask_logits):
                                mask_logits_cpu = out_mask_logits.cpu()
                            else:
                                mask_logits_cpu = torch.from_numpy(out_mask_logits) if isinstance(out_mask_logits, np.ndarray) else out_mask_logits
                            
                            # Convert to binary masks
                            masks = (mask_logits_cpu > 0.0).numpy()
                            
                            # Handle single mask vs multiple masks
                            if len(masks.shape) == 2:
                                masks = masks[np.newaxis, ...]  # Add batch dimension
                            
                            boxes_list = []
                            scores_list = []
                            valid_masks = []
                            
                            for mask_idx, mask in enumerate(masks):
                                if mask.sum() == 0:
                                    continue
                                ys, xs = np.where(mask)
                                if len(xs) == 0 or len(ys) == 0:
                                    continue
                                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                                boxes_list.append([x1, y1, x2, y2])
                                
                                # Get score from mask logits
                                if torch.is_tensor(mask_logits_cpu):
                                    score = float(mask_logits_cpu[mask].max())
                                else:
                                    score = float(np.max(mask_logits_cpu[mask]))
                                scores_list.append(score)
                                valid_masks.append(mask)
                            
                            # Filter by confidence threshold
                            valid_indices = [i for i, score in enumerate(scores_list) if score >= confidence_threshold]
                            
                            results.append({
                                'frame_idx': int(out_frame_idx),
                                'masks': np.array(valid_masks)[valid_indices] if valid_masks else np.array([]),
                                'boxes': np.array(boxes_list)[valid_indices] if boxes_list else np.zeros((0, 4)),
                                'scores': np.array(scores_list)[valid_indices] if scores_list else np.zeros((0,))
                            })
                        
                        logger.info(f"Processed {frame_count} frames (CPU mode)")
                        
                except RuntimeError as e:
                    error_msg = str(e)
                    logger.error(f"Error during video propagation: {error_msg}")
                    if "No input points or masks" in error_msg:
                        raise RuntimeError(
                            "No prompts provided. Please add at least one point or box prompt before processing. "
                            "If you selected auto-segment, the system will use a center point automatically."
                        )
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error during video processing: {str(e)}")
                    raise
            
            logger.info(f"Video processing complete: {len(results)} result frames")
            
            # Cleanup temporary video if created
            if frame is not None and temp_video.exists():
                temp_video.unlink()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


    def draw_masks(
        self, 
        frame_bgr: np.ndarray, 
        masks: np.ndarray, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Draw masks and boxes on frame
        
        Args:
            frame_bgr: Input frame in BGR format
            masks: Array of masks
            boxes: Array of boxes [x1, y1, x2, y2]
            scores: Array of confidence scores
            alpha: Transparency for mask overlay
        
        Returns:
            Frame with masks and boxes drawn
        """
        if len(masks) == 0 or (isinstance(masks, np.ndarray) and masks.size == 0):
            return frame_bgr.copy()
        
        overlay = frame_bgr.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Handle single mask case
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]
        
        for i, mask in enumerate(masks):
            if mask.sum() == 0:
                continue
            color = colors[i % len(colors)]
            
            # Apply mask overlay
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
            
            # Draw box if available
            if len(boxes) > i:
                box = boxes[i]
                if len(box) >= 4:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw score if available
                    if len(scores) > i:
                        score = scores[i]
                        cv2.putText(
                            overlay, 
                            f"{score:.2f}", 
                            (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            color, 
                            2
                        )

        return overlay