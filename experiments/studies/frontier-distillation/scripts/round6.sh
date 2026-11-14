#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
until [ -f logs/round5.done ]; do sleep 30; done
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --unlabeled fma-medium/cqt --soft-labels teacher-control.npz "${@:4}" > logs/$2.log 2>&1; }
for seed in 13 15 42; do
  run 1 shift7-s$seed $seed --max-shift 7 &
  run 0 shift12-s$seed $seed --max-shift 12 &
done
wait
echo "round6 done" > logs/round6.done
