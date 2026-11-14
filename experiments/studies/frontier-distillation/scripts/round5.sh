#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --unlabeled fma-medium/cqt "${@:4}" > logs/$2.log 2>&1; }
for seed in 13 15 42; do
  run 1 softonly-s$seed $seed --soft-labels teacher-control.npz --hard-weight 0.0 &
  run 1 gen2-s$seed $seed --soft-labels teacher-gen2.npz --hard-weight 0.5 &
  run 0 softheavy-s$seed $seed --soft-labels teacher-control.npz --hard-weight 0.25 --unlabeled-weight 2.0 &
done
wait
echo "round5 done" > logs/round5.done
