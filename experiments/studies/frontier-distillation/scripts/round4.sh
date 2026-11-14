#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
until [ -f logs/round1b.done ] && [ -f logs/round2.done ]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
for seed in 13 15 42; do
  "$python_bin" scripts/frontier_train.py train --out runs/wide2-s$seed --seed $seed --epochs 120 --width 2 > logs/wide2-s$seed.log 2>&1 &
  "$python_bin" scripts/frontier_train.py train --out runs/longcrop-s$seed --seed $seed --epochs 120 --crop 640 > logs/longcrop-s$seed.log 2>&1 &
done
wait
echo "round4 done" > logs/round4.done
