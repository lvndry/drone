#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=1
for seed in 13 15 42; do
  "$python_bin" scripts/frontier_train.py train --out runs/distill-s$seed --seed $seed --epochs 120 --unlabeled fma-medium/cqt --soft-labels teacher-control.npz > logs/distill-s$seed.log 2>&1 &
done
"$python_bin" scripts/frontier_train.py train --out runs/consistency-s13 --seed 13 --epochs 120 --unlabeled fma-medium/cqt > logs/consistency-s13.log 2>&1 &
wait
echo "round3 done" > logs/round3.done
