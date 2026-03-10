"""EE Trajectory Controller — backed by manifold types."""

from .controller import computeSingleDeltaTwist, computeDeltaTwists
from .metrics import computeMetrics, plotMetrics

__all__ = [
    "computeSingleDeltaTwist",
    "computeDeltaTwists",
    "computeMetrics",
    "plotMetrics",
]
