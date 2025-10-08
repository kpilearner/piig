"""
Thermal Query Decoder Utilities

Inspired by DDColor's DETR-based architecture for image colorization,
adapted for infrared thermal image generation.
"""

from .position_encoding import PositionEmbeddingSine
from .attention_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP

__all__ = [
    'PositionEmbeddingSine',
    'SelfAttentionLayer',
    'CrossAttentionLayer',
    'FFNLayer',
    'MLP',
]
