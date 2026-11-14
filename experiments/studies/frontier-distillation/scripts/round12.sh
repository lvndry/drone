#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --pool pool-r9 --soft-labels teacher-r9.npz --max-shift 12 --dj-fraction 0.0 --suspect-labels confident-learning.npz "${@:4}" > logs/$2.log 2>&1; }
run 0 clteacher-s13 13 --suspect-mode teacher &
run 0 clteacher-s15 15 --suspect-mode teacher &
run 0 clteacher-s42 42 --suspect-mode teacher &
run 0 cldrop-s13 13 --suspect-mode drop &
run 0 cldrop-s15 15 --suspect-mode drop &
run 1 cldrop-s42 42 --suspect-mode drop &
wait
echo "round12 done" > logs/round12.done
