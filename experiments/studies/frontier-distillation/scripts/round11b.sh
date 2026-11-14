#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
run() { DRONE_FEATURE_VIEW=harmonic CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --pool pool-r9h --soft-labels teacher-r9h.npz --max-shift 12 --dj-fraction 0.0 > logs/$2.log 2>&1; }
run 0 harmonic-s13 13 &
run 0 harmonic-s15 15 &
wait -n
run 0 harmonic-s42 42 &
wait
echo "round11 done" > logs/round11.done
