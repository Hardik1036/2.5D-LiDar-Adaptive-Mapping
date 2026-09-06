"""
ML Perception Adapter re-export in backend.ingestion for modular import convenience.
"""

from backend.adapters.ml_adapter import (
    MLPerceptionAdapter,
    evaluate_thin_hazards,
    _init_threatnet_session,
)

__all__ = [
    "MLPerceptionAdapter",
    "evaluate_thin_hazards",
]
