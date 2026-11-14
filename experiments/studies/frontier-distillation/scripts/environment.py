"""Machine-specific locations, read from environment variables so no server path lives in the code.

DRONE_SOURCE: checkout of the drone package, placed on sys.path.
DRONE_PREP: directory holding manifest.json and the cached features.
DRONE_GIANTSTEPS: GiantSteps dataset root.
DRONE_WORK: the experiment working directory.
DRONE_MYNA: directory of the released Myna encoder (myna.py, model.safetensors, config.json).
DRONE_MYNA_FRONTEND: directory of the qualified Myna mel frontend and playback-rate shift.
DRONE_MYNA_CACHE: directory of the earlier Myna embedding cache.
"""

from __future__ import annotations

import os
from pathlib import Path


def required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"environment variable {variable} must be set to a directory path")
    return Path(value)
