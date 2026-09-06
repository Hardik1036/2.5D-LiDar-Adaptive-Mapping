"""
Compact statistical height representations for 2.5D elevation grid cells.
Stores [z_max, z_min, delta_z, variance, slope] without copying raw points.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(slots=True)
class CellStats:
    """
    Compact 2.5D statistical descriptor of a spatial grid cell.
    Zero raw points stored to achieve edge memory efficiency.
    """
    z_min: float
    z_max: float
    delta_z: float
    variance: float
    slope: float        # Inclination angle in degrees relative to horizontal
    point_count: int
    mean_z: float

    def to_tuple(self):
        """Returns compact numerical tuple [z_max, z_min, delta_z, variance, slope]."""
        return (self.z_max, self.z_min, self.delta_z, self.variance, self.slope)

    def to_dict(self):
        return {
            "z_min": round(self.z_min, 3),
            "z_max": round(self.z_max, 3),
            "delta_z": round(self.delta_z, 3),
            "variance": round(self.variance, 4),
            "slope": round(self.slope, 2),
            "point_count": self.point_count,
            "mean_z": round(self.mean_z, 3),
        }


def compute_cell_statistics(points: np.ndarray, cell_size: float = 0.50) -> Optional[CellStats]:
    """
    Computes statistical attributes for points falling inside a 2D cell.

    Args:
        points: (N, 3) or (N, 4) point array containing [x, y, z, ...]
        cell_size: Spatial span of the cell in meters.

    Returns:
        CellStats instance or None if empty.
    """
    n = len(points)
    if n == 0:
        return None

    z = points[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    delta_z = z_max - z_min
    mean_z = float(np.mean(z))
    variance = float(np.var(z)) if n > 1 else 0.0

    # Fast 2.5D geometric slope estimation: theta = arctan(delta_z / cell_size)
    if delta_z < 0.01 or cell_size <= 0.0:
        slope_deg = 0.0
    else:
        slope_deg = float(np.degrees(np.arctan2(delta_z, cell_size)))

    return CellStats(
        z_min=z_min,
        z_max=z_max,
        delta_z=delta_z,
        variance=variance,
        slope=slope_deg,
        point_count=n,
        mean_z=mean_z,
    )
