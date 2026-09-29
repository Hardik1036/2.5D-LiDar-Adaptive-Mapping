import sys
from pathlib import Path

# Ensure Backend directory is in sys.path so 'backend' package imports work cleanly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.tests.conftest import generate_procedural_sweep, sample_sweep_points  # noqa: F401
