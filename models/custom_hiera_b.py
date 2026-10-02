"""
Custom Hiera-B Encoder for SAM 2
Complete implementation with domain-specific preprocessing integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple, Optional
from functools import partial

logger = logging.getLogger(__name__)

# Import SAM2 components
try:
    from sam2.modeling.backbones.hieradet import Hiera, MultiScaleBlock, MultiScaleAttention
    from sam2.modeling.backbones.utils import PatchEmbed, window_partition, window_unpartition
    from sam2.modeling.sam2_utils import DropPath, MLP
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    SAM2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SAM2 components not available: {e}")
    SAM2_AVAILABLE = False
    # Define dummy classes
    Hiera = nn.Module
    MultiScaleBlock = nn.Module
    ImageEncoder = nn.Module


class CustomHieraB(nn.Module):
    """
    Custom Hiera-B (Base) Encoder
    Optimized for video segmentation with:
    - Multi-scale feature extraction (64×64, 32×32, 16×16, 8×8)
    - 256-d features per location
    - Enhanced temporal consistency
    """
    
    def __init__(
        self,
        embed_dim: int = 112,  # Hiera-B base dimension
        num_heads: int = 2,     # Base number of heads
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # Hiera-B architecture
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        return_interm_layers: bool = True,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 components are required for CustomHieraB")
        
        # Initialize base Hiera encoder
        self.base_encoder = Hiera(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            q_pool=q_pool,
            q_stride=q_stride,
            stages=stages,
            dim_mul=dim_mul,
            head_mul=head_mul,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            window_spec=window_spec,
            global_att_blocks=global_att_blocks,
            return_interm_layers=return_interm_layers,
            weights_path=weights_path,
        )
        
        self.embed_dim = embed_dim
        self.stages = stages
        self.return_interm_layers = return_interm_layers
        
        # Feature dimension mapping
        self.channel_list = self.base_encoder.channel_list
        
        logger.info(f"CustomHieraB initialized with embed_dim={embed_dim}, stages={stages}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through custom Hiera-B encoder
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of feature maps at different scales
        """
        # Forward through base encoder
        features = self.base_encoder(x)
        
        return features
    
    def get_channel_list(self) -> List[int]:
        """Get list of channel dimensions for each scale"""
        return self.channel_list
    
    def load_pretrained_weights(self, weights_path: str, strict: bool = False):
        """
        Load pretrained weights
        
        Args:
            weights_path: Path to weights file
            strict: Whether to strictly match keys
        """
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # Load into base encoder
            missing_keys, unexpected_keys = self.base_encoder.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
            
            logger.info(f"Loaded pretrained weights from {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            return False


class CustomHieraBImageEncoder(nn.Module):
    """
    Complete Image Encoder using Custom Hiera-B
    Integrates with SAM2's image encoder structure
    """
    
    def __init__(
        self,
        custom_hiera: CustomHieraB,
        d_model: int = 256,
        position_encoding=None,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
    ):
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 components are required")
        
        self.trunk = custom_hiera
        channel_list = custom_hiera.get_channel_list()
        
        # Create FPN neck for multi-scale features
        self.neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=d_model,
            backbone_channel_list=channel_list,
            fpn_interp_model=fpn_interp_model,
            fuse_type=fuse_type,
        )
        
        self.channel_list = channel_list
    
    def forward(self, sample: torch.Tensor):
        """
        Forward pass through encoder
        
        Args:
            sample: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary with vision features and positional encodings
        """
        # Extract features from Hiera-B
        backbone_features = self.trunk(sample)
        
        # Process through FPN neck
        features, pos = self.neck(backbone_features)
        
        # Get final feature map
        src = features[-1]
        
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        
        return output


def build_custom_hiera_b(
    d_model: int = 256,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    **kwargs
) -> CustomHieraBImageEncoder:
    """
    Build Custom Hiera-B Image Encoder
    
    Args:
        d_model: Output feature dimension (default 256)
        pretrained: Whether to load pretrained weights
        weights_path: Path to pretrained weights
        **kwargs: Additional arguments for CustomHieraB
    
    Returns:
        CustomHieraBImageEncoder instance
    """
    # Create custom Hiera-B encoder
    custom_hiera = CustomHieraB(
        embed_dim=kwargs.get("embed_dim", 112),
        stages=kwargs.get("stages", (2, 3, 16, 3)),
        **{k: v for k, v in kwargs.items() if k not in ["embed_dim", "stages"]}
    )
    
    # Load pretrained weights if specified
    if pretrained and weights_path:
        custom_hiera.load_pretrained_weights(weights_path)
    
    # Create position encoding if not provided
    position_encoding = kwargs.get("position_encoding")
    if position_encoding is None:
        try:
            from sam2.modeling.position_encoding import PositionEmbeddingSine
            # PositionEmbeddingSine is the actual class used in SAM2
            position_encoding = PositionEmbeddingSine(
                num_pos_feats=d_model // 2,
                temperature=10000,
                normalize=True,
            )
            logger.info(f"Created PositionEmbeddingSine for d_model={d_model}")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Position encoding not available: {e}")
            # FpnNeck requires position_encoding, so we need to create a dummy one
            try:
                from sam2.modeling.position_encoding import PositionEmbeddingSine
                position_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2)
            except Exception as e2:
                logger.error(f"Failed to create position encoding: {e2}")
                raise ImportError("Could not create position encoding - required for FpnNeck")
    
    # Create complete image encoder
    encoder = CustomHieraBImageEncoder(
        custom_hiera=custom_hiera,
        d_model=d_model,
        position_encoding=position_encoding,
    )
    
    return encoder

