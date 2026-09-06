"""
Unit tests for PorosityClassifier (Feature F1.5).
"""

import numpy as np
import pytest

from backend.mapping.porosity_filter import PorosityClassifier


def test_porosity_classifier():
    classifier = PorosityClassifier(
        min_veg_height=0.15,
        max_veg_height=1.30,
        porous_cost=20,
        rigid_cost=255,
    )

    rng = np.random.default_rng(42)

    # 1. Simulate soft porous vegetation (tall grass)
    # Height between 0.2m and 0.8m, horizontal spread, low intensity
    n_pts = 60
    grass_x = rng.uniform(-0.5, 0.5, n_pts)
    grass_y = rng.uniform(-0.5, 0.5, n_pts)
    grass_z = rng.uniform(0.1, 0.7, n_pts)
    grass_intensity = rng.uniform(0.2, 0.5, n_pts)  # Diffuse return
    grass_cluster = np.column_stack([grass_x, grass_y, grass_z, grass_intensity])

    is_porous, cost = classifier.evaluate_cluster(grass_cluster)
    assert is_porous is True
    assert cost == 20

    # 2. Simulate solid tree trunk / rigid boulder
    # Tall dense column, solid termination, high intensity
    trunk_x = rng.normal(0.0, 0.05, n_pts)
    trunk_y = rng.normal(0.0, 0.05, n_pts)
    trunk_z = np.linspace(0.0, 2.5, n_pts)  # 2.5m tall rigid trunk
    trunk_intensity = rng.uniform(0.85, 1.0, n_pts)  # High solid reflection
    trunk_cluster = np.column_stack([trunk_x, trunk_y, trunk_z, trunk_intensity])

    is_porous_trunk, cost_trunk = classifier.evaluate_cluster(trunk_cluster)
    assert is_porous_trunk is False
    assert cost_trunk == 255


def test_porosity_edge_cases():
    classifier = PorosityClassifier()
    # Empty and tiny clusters
    assert classifier.evaluate_cluster(np.empty((0, 3))) == (False, 255)
    assert classifier.evaluate_cluster(np.array([[0, 0, 0]])) == (False, 255)
