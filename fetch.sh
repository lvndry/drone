#!/usr/bin/env bash
# Download and verify training audio/labels and held-out GiantSteps data.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m drone.download "$@"
