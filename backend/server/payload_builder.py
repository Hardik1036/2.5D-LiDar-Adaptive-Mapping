"""
Serializes quadtree cells, dynamic tracks, hazard cones, and telemetry into optimized JSON payloads.
Edge-tailored for high-speed transmission at >= 25 Hz.
"""

import json
from typing import Dict, List, Optional
import numpy as np

from backend.config import SERVER
from backend.mapping.quadtree import QuadtreeNode
from backend.tracking.kalman_tracker import TrackedObject
from backend.tracking.trajectory_rollout import HazardCone


class PayloadBuilder:
    """
    Constructs high-speed JSON payloads for frontend visualization.
    Prioritizes fine-resolution, hazardous, and dynamic areas for network efficiency.
    """

    def __init__(self, max_cells: int = SERVER.MAX_PAYLOAD_CELLS):
        self.max_cells = max_cells

    def build_payload(
        self,
        frame_id: int,
        timestamp: float,
        system_stats: dict,
        leaves: List[QuadtreeNode],
        tracks: List[TrackedObject],
        hazard_cones: Dict[int, List[HazardCone]],
    ) -> str:
        """
        Builds a ready-to-broadcast JSON string in < 3ms.
        """
        # 1. Process cells: prioritize obstacles, caution zones, and downsample flat terrain
        if len(leaves) <= self.max_cells:
            serialized_cells = [leaf.to_dict() for leaf in leaves]
        else:
            high_priority = []
            standard = []
            for leaf in leaves:
                if leaf.cost > 50:
                    high_priority.append(leaf.to_dict())
                else:
                    standard.append(leaf)

            remaining_slots = max(0, self.max_cells - len(high_priority))
            step = max(1, len(standard) // max(remaining_slots, 1))
            sampled_standard = [leaf.to_dict() for leaf in standard[::step][:remaining_slots]]
            serialized_cells = high_priority[:self.max_cells] + sampled_standard

        # 2. Process dynamic tracks and associated hazard cones
        serialized_objects = []
        for track in tracks:
            obj_dict = track.to_dict()
            cones = hazard_cones.get(track.track_id, [])
            obj_dict["hazard_cones"] = [cone.to_dict() for cone in cones]
            serialized_objects.append(obj_dict)

        payload_dict = {
            "timestamp": round(float(timestamp), 3),
            "frame_id": int(frame_id),
            "system_status": system_stats.get("system_status", "ALL_SYSTEMS_NOMINAL"),
            "system_stats": system_stats,
            "cells": serialized_cells,
            "dynamic_objects": serialized_objects,
        }

        def _failsafe(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            return str(obj)

        return json.dumps(payload_dict, separators=(",", ":"), default=_failsafe)
