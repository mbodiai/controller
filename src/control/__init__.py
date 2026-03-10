"""EE Trajectory Controller — backed by manifold types."""

from .controller import computeSingleDeltaTwist, computeDeltaTwists
from .metrics import computeMetrics, plotMetrics
from .utils import return_data

__all__ = [
    "computeSingleDeltaTwist",
    "computeDeltaTwists",
    "computeMetrics",
    "plotMetrics",
    "return_data",
]
