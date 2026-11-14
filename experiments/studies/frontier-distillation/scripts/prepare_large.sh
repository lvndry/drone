#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK/fma-large"
python_bin="${DRONE_PYTHON:-python}"
while pgrep -f "curl -sSL -C - -o fma_large.zip" > /dev/null; do sleep 15; done
if [ "$(stat -c %s fma_large.zip)" != "100306112191" ]; then echo "download incomplete" > ../logs/large.done; exit 1; fi
for folder in $(seq -f %03g 0 155); do
  rm -rf part && mkdir part
  unzip -qo fma_large.zip "fma_large/$folder/*" -d part 2>/dev/null
  "$python_bin" ../scripts/make_jobs.py unlabeled part part-jobs.json > /dev/null
  nice -n 10 "$python_bin" ../scripts/extract_views.py --jobs part-jobs.json --workers 24 --summary "summaries/$folder.json" >> ../logs/extract-large.log 2>&1
  printf "%s " "$folder" >> ../logs/extract-large-progress.log
done
rm -rf part part-jobs.json fma_large.zip
echo "large done" > ../logs/large.done
