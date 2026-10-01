"""
Negative Obstacle & Trench Drop-off Detector (Feature F5.1).
Exposes blind negative terrain hazards (ditches, drop-offs, trenches, erosion gullies)
using geometric ray shadow and line-of-sight dropout analysis.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.mapping.quadtree import QuadtreeNode
from backend.mapping.trench_detector import TrenchDetector

__all__ = ["TrenchDetector"]
