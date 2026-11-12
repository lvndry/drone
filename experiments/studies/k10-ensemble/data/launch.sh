#!/bin/bash
set -u
ROOT=/home/lysk-admin/drone-experiments/k10-ensemble-20261112-v1
cd "$ROOT/source" || exit 1
export PYTHONPATH="$ROOT/source:/home/lysk-admin/drone-experiments/octave-deps-py312"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
REF=/home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1
SAM=/home/lysk-admin/drone-experiments/k10-sam-20261112-v1
printf '%s\n' c6b881ea365396ac53393de0dc945092714d626f > "$ROOT/source-commit.txt"
sha256sum /home/lysk-admin/drone-experiments/drone-k10-ensemble-c6b881e.tar.gz > "$ROOT/archive.sha256"
timeout 1250 /home/lysk-admin/drone/.venv/bin/python -u -m experiments.ensemble \
 --data "$REF/data" --output "$ROOT/result" --reference-source "$REF/source" \
 --joint-runs "$REF/joint-seed13/run.json" "$REF/joint-seed37/run.json" "$REF/joint-seed73/run.json" \
 --bp-runs "$REF/control-seed13/run.json" "$REF/control-seed37/run.json" "$REF/control-seed73/run.json" \
 --sam-result "$SAM/candidate-seed13/sam-result.json" --sam-exit "$SAM/candidate.exit"
RESULT=$?
printf '%s\n' "$RESULT" > "$ROOT/stage.exit"
exit "$RESULT"
