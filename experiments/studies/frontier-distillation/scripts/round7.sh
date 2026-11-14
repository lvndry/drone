#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
until [ -f logs/round6.done ]; do sleep 30; done
for fold in 0 1 2 3 4; do
  gpu=$(( fold % 2 ))
  CUDA_VISIBLE_DEVICES=$gpu "$python_bin" scripts/frontier_train.py train --out runs/fold$fold-s13 --seed 13 --epochs 120 --folds 5 --fold $fold > logs/fold$fold-s13.log 2>&1 &
done
wait
echo "round7 done" > logs/round7.done
