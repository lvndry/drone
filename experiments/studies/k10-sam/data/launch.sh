#!/bin/bash
set -u
ROOT=/home/lysk-admin/drone-experiments/k10-sam-20261112-v1
cd "$ROOT/source" || exit 1
export PYTHONPATH="$ROOT/source:/home/lysk-admin/drone-experiments/octave-deps-py312"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf '%s\n' 'f09febd2d658f82fe31ee18593a988ac59733e65' > "$ROOT/source-commit.txt"
sha256sum /home/lysk-admin/drone-experiments/drone-k10-sam-f09febd.tar.gz > "$ROOT/archive.sha256"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/gpu-before.csv"
REF=/home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1
timeout 1250 /home/lysk-admin/drone/.venv/bin/python -u -m experiments.samstudy \
 --data "$REF/data" --output "$ROOT/control-seed13" \
 --reference-run "$REF/joint-seed13/run.json" --reference-bp "$REF/control-seed13/run.json"
RESULT=$?
printf '%s\n' "$RESULT" > "$ROOT/control.exit"
if [ "$RESULT" -ne 0 ]; then exit "$RESULT"; fi
/home/lysk-admin/drone/.venv/bin/python - "$ROOT/control-seed13/sam-result.json" <<'PY'
import sys,json
r=json.load(open(sys.argv[1]))
assert r['status']=='complete' and r['kind']=='control' and r['control_reproduced'] is True
PY
RESULT=$?
if [ "$RESULT" -ne 0 ]; then printf '%s\n' "$RESULT" > "$ROOT/control-gate.exit"; exit "$RESULT"; fi
timeout 2450 /home/lysk-admin/drone/.venv/bin/python -u -m experiments.samstudy --candidate \
 --data "$REF/data" --output "$ROOT/candidate-seed13" \
 --reference-run "$REF/joint-seed13/run.json" --reference-bp "$REF/control-seed13/run.json" \
 --paired-control "$ROOT/control-seed13/run.json"
RESULT=$?
printf '%s\n' "$RESULT" > "$ROOT/candidate.exit"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/gpu-after.csv"
exit "$RESULT"
