#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
until [ -f logs/large.done ] && [ -f logs/round8b.done ] && [ -f round9-dj-fraction.txt ]; do sleep 60; done
dj_fraction=$(cat round9-dj-fraction.txt)
"$python_bin" scripts/pack_pool.py --source fma_full=fma-full/cqt --source fma_clip=fma-medium/cqt --source fma_large=fma-large/cqt-all --out pool-r9 > logs/pack-r9.log 2>&1 || exit 1
members=""
for arm in shift12 shift7 poolall; do
  for seed in 13 15 42; do members="$members,runs/$arm-s$seed/best.pt"; done
done
CUDA_VISIBLE_DEVICES=0 "$python_bin" scripts/teacher_pool.py --checkpoints "${members#,}" --pool pool-r9 --out teacher-r9.npz > logs/teacher-r9.log 2>&1 || exit 1
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --pool pool-r9 --soft-labels teacher-r9.npz --max-shift 12 --dj-fraction "$dj_fraction" "${@:4}" > logs/$2.log 2>&1; }
for seed in 13 15 42; do
  run 0 gen3-s$seed $seed &
done
wait
echo "round9 done" > logs/round9.done
