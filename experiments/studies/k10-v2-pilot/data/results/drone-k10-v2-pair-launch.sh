#!/bin/bash
set -uo pipefail
pair_root=/home/lysk-admin/drone-experiments/k10-v2-pair-20261112-v1
full_root=/home/lysk-admin/drone-experiments/k10-v2-full-20261112-v1
mkdir "$pair_root/launch-once" || exit 98
cd "$full_root/source" || exit 97
export PYTHONPATH="$full_root/source"
export CUDA_VISIBLE_DEVICES=GPU-387d0c41-22e5-4c3b-efe8-1d8f1dd0494c
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
pair_remaining_seconds=$(python3 - <<'PY'
import datetime
now=datetime.datetime.now(datetime.timezone.utc)
start=datetime.datetime(2026,11,12,10,47,tzinfo=datetime.timezone.utc)
cutoff=datetime.datetime(2026,11,12,11,46,58,tzinfo=datetime.timezone.utc)
assert now>=start
remaining=int((cutoff-now).total_seconds())
assert remaining>=2515
print(remaining)
PY
) || exit 96
printf '%s\n' "$pair_remaining_seconds" > "$pair_root/outer-timeout-seconds.txt"
/usr/bin/time -v timeout -s INT -k 2 "$pair_remaining_seconds" /home/lysk-admin/drone/.venv/bin/python "$full_root/eligibility-v2/run_k10_v2_pair.py" --config "$full_root/eligibility-v2/pair-launch-config.json" --permit "$pair_root/pair-permit.json" --plan "$full_root/eligibility-v2/plan-v2.json.gz" --features "$full_root/prepared-full/manifest.eligibility-v2.json" --training-source "$full_root/source" --output-root "$pair_root/runs"
pair_exit=$?
printf '%s\n' "$pair_exit" > "$pair_root/launch.exit"
exit "$pair_exit"
