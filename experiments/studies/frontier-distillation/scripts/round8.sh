#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK"
python_bin="${DRONE_PYTHON:-python}"
until [ -f fma-full/fma-full-summary.json ] && [ -f logs/round7.done ]; do sleep 60; done
"$python_bin" scripts/pack_pool.py --source fma_full=fma-full/cqt --source fma_clip=fma-medium/cqt --out pool-r8 > logs/pack-r8.log 2>&1 || exit 1
runs_directory=runs
CUDA_VISIBLE_DEVICES=1 "$python_bin" scripts/teacher_pool.py --checkpoints $runs_directory/control-s13/best.pt,$runs_directory/control-s15/best.pt,$runs_directory/control-s42/best.pt --pool pool-r8 --out teacher-r8.npz > logs/teacher-r8.log 2>&1 || exit 1
run() { CUDA_VISIBLE_DEVICES=$1 "$python_bin" scripts/frontier_train.py train --out runs/$2 --seed $3 --epochs 120 --pool pool-r8 --soft-labels teacher-r8.npz --max-shift 12 "${@:4}" > logs/$2.log 2>&1; }
for seed in 13 15 42; do
  run 0 poolall-s$seed $seed --dj-fraction 0.0 &
  run 1 pooldj-s$seed $seed --dj-fraction 0.8 &
done
wait
echo "round8 done" > logs/round8.done
