#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --pool pool-r9 --soft-labels teacher-r9.npz --max-shift 12 --dj-fraction 0.0 "${@:4}" > logs/$2.log 2>&1; }
run 1 gen3w2-s13 13 --width 2 &
run 1 gen3w2-s15 15 --width 2 &
run 0 gen3w2-s42 42 --width 2 &
wait
echo "round10 done" > logs/round10.done
