#!/bin/bash
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
for seed in 13 15 42; do
  mv logs/pooldj-s$seed.log logs/pooldj-s$seed-oom.log 2>/dev/null
  rm -rf runs/pooldj-s$seed
  CUDA_VISIBLE_DEVICES=0 "$python_bin" scripts/frontier_train.py train --out runs/pooldj-s$seed --seed $seed --epochs 120 --pool pool-r8 --soft-labels teacher-r8.npz --max-shift 12 --dj-fraction 0.8 > logs/pooldj-s$seed.log 2>&1 &
done
wait
touch logs/round8b.done
