#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=1
for seed in 13 15 42; do
  "$python_bin" scripts/frontier_train.py train --out runs/control-s$seed --seed $seed --epochs 120 > logs/control-s$seed.log 2>&1 &
done
"$python_bin" scripts/frontier_train.py pretrain --out runs/pretrain-pool --seed 1 --epochs 100 > logs/pretrain-pool.log 2>&1
pretrain_status=$?
if [ $pretrain_status -eq 0 ]; then
  for seed in 13 15 42; do
    "$python_bin" scripts/frontier_train.py train --out runs/sslinit-s$seed --seed $seed --epochs 120 --init runs/pretrain-pool/pretrained-aligned.pt > logs/sslinit-s$seed.log 2>&1 &
  done
fi
wait
echo "round1 done pretrain_status=$pretrain_status" > logs/round1.done
