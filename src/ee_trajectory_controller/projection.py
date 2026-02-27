"""Re-export manifold projection utility used by the controller.

The canonical implementation lives in manifold.utils.geometry.project_pose.
This module provides the backward-compatible alias.
"""

from manifold.utils.geometry import project_pose

# Backward-compatible alias
projectObjectState = project_pose

__all__ = [
    "project_pose",
    "projectObjectState",
]
