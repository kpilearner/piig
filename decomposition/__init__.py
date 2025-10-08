"""
Physics-Inspired Multi-Task Decomposition Module
"""

from .model import MultiTaskDecompositionNet, PhysicsInspiredFusion
from .losses import MultiTaskPretrainLoss

__all__ = [
    'MultiTaskDecompositionNet',
    'PhysicsInspiredFusion',
    'MultiTaskPretrainLoss'
]
