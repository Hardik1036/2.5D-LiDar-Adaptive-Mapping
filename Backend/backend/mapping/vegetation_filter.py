
"""
Vegetation Penetration and Traversability Cost Filter.
Differentiates passable tall grass and shrubbery from rigid lethal obstacles
for DRDO SIH 2026 Problem Statement 53.
"""

from typing import List, Optional
import numpy as np

from backend.config import BOUNDS, QUADTREE, COSTMAP
from backend.mapping.quadtree import QuadtreeNode


class VegetationFilter:
    """
    Overrules geometric height variance false-positives caused by passable vegetation.
    In outdoor off-road scenarios, tall grass produces significant delta_z and elevation variance,
    falsely triggering lethal traversability cost.
    
    VegetationFilter uses ML semantic classification:
    - Cells classified as Passable Vegetation (Label 1) have their traversability cost capped at 20 (Safe).
    - Cells containing Rigid Obstacles (Label 2) are forced to 255 (Lethal).
    """

    DEFAULT_MAX_VEGETATION_COST: int = 20
    LETHAL_COST: int = 255

    def __init__(
        self,
        max_vegetation_cost: int = DEFAULT_MAX_VEGETATION_COST,
        lethal_cost: int = LETHAL_COST,
        coarse_resolution: float = QUADTREE.COARSE_RESOLUTION,
    ):
        self.max_veg_cost = max_vegetation_cost
        self.lethal_cost = lethal_cost
        self.coarse_res = coarse_resolution

    def apply_vegetation_scaling(
        self,
        leaves: List[QuadtreeNode],
        vegetation_points: np.ndarray,
        rigid_points: Optional[np.ndarray] = None,
    ) -> None:
        """
        Scales cell traversability costs based on semantic point distributions.
        Operates in < 1.0 ms via 2D spatial binning.

        Args:
            leaves: List of QuadtreeNode leaves in the active 2.5D map.
            vegetation_points: (K, 3) point array labeled as vegetation (Label 1).
            rigid_points: (J, 3) optional point array labeled as rigid obstacles (Label 2).
        """
        if not leaves:
            return

        has_veg = vegetation_points is not None and len(vegetation_points) > 0
        has_rigid = rigid_points is not None and len(rigid_points) > 0

        if not has_veg and not has_rigid:
            return

        x_min = BOUNDS.X_MIN
        y_min = BOUNDS.Y_MIN
        inv_res = 1.0 / self.coarse_res
        num_x = int((BOUNDS.X_MAX - x_min) * inv_res) + 1
        num_y = int((BOUNDS.Y_MAX - y_min) * inv_res) + 1

        # Spatial hash of rigid obstacle points
        rigid_set = set()
        if has_rigid:
            rx = rigid_points[:, 0]
            ry = rigid_points[:, 1]
            r_ix = np.clip(((rx - x_min) * inv_res).astype(np.int32), 0, num_x - 1)
            r_iy = np.clip(((ry - y_min) * inv_res).astype(np.int32), 0, num_y - 1)
            rigid_set = set((r_iy * num_x + r_ix).tolist())

        # Spatial hash of vegetation points
        veg_set = set()
        if has_veg:
            vx = vegetation_points[:, 0]
            vy = vegetation_points[:, 1]
            v_ix = np.clip(((vx - x_min) * inv_res).astype(np.int32), 0, num_x - 1)
            v_iy = np.clip(((vy - y_min) * inv_res).astype(np.int32), 0, num_y - 1)
            veg_set = set((v_iy * num_x + v_ix).tolist())

        # In-place node cost adjustment
        for leaf in leaves:
            l_ix = int((leaf.x - x_min) * inv_res)
            l_iy = int((leaf.y - y_min) * inv_res)
            if 0 <= l_ix < num_x and 0 <= l_iy < num_y:
                cell_idx = l_iy * num_x + l_ix

                if cell_idx in rigid_set:
                    # Rigid obstacles always take precedence and are lethal
                    leaf.cost = self.lethal_cost
                elif cell_idx in veg_set:
                    # Passable vegetation: suppress geometric roughness ONLY for low ground vegetation (tall grass/brush).
                    # Elevated vegetation (tree canopies, overhanging branches with delta_z > 0.12m and Z > -1.2m)
                    # sits well above ground clearance and must not be suppressed to passable.
                    st = leaf.stats
                    is_elevated_tree = (
                        st is not None
                        and st.delta_z > 0.12
                        and (st.mean_z > COSTMAP.ELEVATED_OBSTACLE_Z or st.z_max > COSTMAP.ELEVATED_OBSTACLE_Z)
                    )
                    if not is_elevated_tree and leaf.cost > self.max_veg_cost:
                        leaf.cost = self.max_veg_cost
