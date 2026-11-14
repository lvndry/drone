#!/usr/bin/env bash
set -uo pipefail
cd "$DRONE_WORK/fma-medium"
python_bin="${DRONE_PYTHON:-python}"
while pgrep -f "curl -sSL -C - -o fma_medium.zip" > /dev/null; do sleep 15; done
size=$(stat -c %s fma_medium.zip)
if [ "$size" != "23825005356" ]; then echo "download incomplete: $size" > ../logs/unlabeled.done; exit 1; fi
sha1sum fma_medium.zip > fma_medium.sha1
mkdir -p extracted && unzip -qo fma_medium.zip -d extracted
nice -n 10 "$python_bin" ../scripts/extract_unlabeled.py --audio-root extracted --out cqt > ../logs/extract-unlabeled.log 2>&1
echo "unlabeled done status=$?" > ../logs/unlabeled.done
