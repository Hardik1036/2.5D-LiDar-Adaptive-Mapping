"""
Thin compatibility wrapper around canonical audit script at scripts/audit_assets.py.
Do NOT duplicate audit logic here.
"""

import sys
from pathlib import Path

# Resolve path to canonical audit script
CANONICAL_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(CANONICAL_SCRIPT_DIR))

try:
    from audit_assets import audit_repository_assets  # type: ignore
except ImportError:
    # Fallback to direct file execution
    import runpy
    audit_assets_path = CANONICAL_SCRIPT_DIR / "audit_assets.py"
    if audit_assets_path.exists():
        runpy.run_path(str(audit_assets_path), run_name="__main__")
        sys.exit(0)
    else:
        raise FileNotFoundError(f"Canonical audit script not found at {audit_assets_path}")

if __name__ == "__main__":
    audit_repository_assets()
