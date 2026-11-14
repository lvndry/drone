#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0
for seed in 13 15 42; do
  "$python_bin" scripts/frontier_train.py train --out runs/consistency-s$seed --seed $seed --epochs 120 --unlabeled fma-medium/cqt > logs/consistency-s$seed.log 2>&1 &
done
wait
echo "round2 done" > logs/round2.done
