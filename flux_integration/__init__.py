"""
FLUX Integration Module

Integrates physics-inspired decomposition with FLUX diffusion model.
"""

from .model import DecompositionGuidedFluxModel
from .cross_attention import DecompositionCrossAttention

__all__ = [
    'DecompositionGuidedFluxModel',
    'DecompositionCrossAttention'
]
