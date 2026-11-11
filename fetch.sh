#!/usr/bin/env bash
# Downloads the training set (1,486 tracks, 2.1 GB) and labels to data/.
# The audio is dev-time only and never ships; see DATA.md.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
[ -f data/zenodo_1101082_audio.zip ] || \
  curl -fL --retry 3 -o data/zenodo_1101082_audio.zip \
  "https://zenodo.org/api/records/1101082/files/audio.zip/content"
echo "sha256 $(sha256sum data/zenodo_1101082_audio.zip | cut -d' ' -f1)"
if [ ! -d data/audio ]; then
  mkdir -p data/audio
  unzip -q data/zenodo_1101082_audio.zip -d data/audio
fi
echo "audio files: $(find data/audio -name '*.mp3' | wc -l)"
