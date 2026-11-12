#!/usr/bin/env bash
set -uo pipefail
root=/home/lysk-admin/drone-experiments/bp-gs-cached-audit-20261112-faa3a6f-v1
cd "$root/source"
export PYTHONPATH="$root/source"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -t 7200
printf '%s\n' "$$" > "$root/wrapper.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/start.utc"
set +e
/usr/bin/time -v -o "$root/resources.txt" timeout -s INT -k 3 3650 taskset -c 4,5 nice -n 15 /home/lysk-admin/drone/.venv/bin/python -m tools.audit_bp_gs run --plan "$root/plan.json" --plan-sha256 8e127968f40651276efdb7ec10f2229aa8204daaf53676a0d3a37677a5532356 --output "$root/evidence" > "$root/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/run.exit.pending"
mv "$root/run.exit.pending" "$root/run.exit"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/end.utc"
exit "$status"
