# KeyFinder reference wrapper qualification — 2026-11-13

This isolated repair qualifies enum conversion, not audio accuracy. Compile against
existing libkeyfinder 2.2.8 headers/library, run `--self-test` only, and preserve
all prior files. No source recordings, decoder, classifier, GPU, or shared environment
changes are authorized by this qualification. New source uses the repository MIT
license; libkeyfinder retains its own license. Do not distribute a linked binary
under an assertion that its dependencies are MIT.

## Frozen behavior

`tools/native/keyfinder_reference.cpp` maps each of the 24 named enum members
explicitly to Drone's C-first interleaved major/minor classes. An independent
literal 25-entry raw-enum golden table checks all keys and silence at compile
time. Runtime `--self-test` emits every mapping as JSON; it also checks rejection
of raw -1 and 25 without classifier calls. Unknown enums abort. Silence is explicit
with null tonic/mode/class, not a numeric key.

Output schema `drone.keyfinder-reference.v1` includes `raw_enum`, status, tonic,
mode, and `drone_class`. Normal input must be a complete RIFF/WAVE mono PCM16 file
with consistent format, rate and chunk sizes. Reject malformed/truncated/empty or
ambiguous chunks. Default `KeyFinder::keyOfAudio` settings remain unchanged. No
stereo handling, normalization, crop, key-profile changes or silence imputation.
A later evaluator must explicitly specify mono conversion and failure denominators.

## Artifact pins

- Existing faulty wrapper source SHA256: `7ca105984a1112c621ff71c2cc98dbecc1a0ad294c6d565531fbb5bac5be122e`.
- Existing faulty binary SHA256: `a8af5fbd4a5b4d8973687d0002c9036e3aea36c39c7340558a350e14ba79c6d3`.
- libkeyfinder archive SHA256: `a54fc6c5ff435bb4b447f175bc97f9081fb5abf0edd5d125e6f5215c8fff4d11`.
- libkeyfinder.so.2.2.8 SHA256: `f475cde06faa3e2ace1d6d8e544076b4d75d168417c303c9808b0c5397948128`.
- FFTW libfftw3.so.3 SHA256: `b5cc02c4d360b5b20111cb0a251f8368af8c9ce3145357f5ba54c51ac4ca9b35`.

Base library directory: `/home/lysk-admin/drone-experiments/native-baseline-20261111`.
Source is `libkeyfinder-2.2.8/src`, library `keyfinder-build/libkeyfinder.so.2.2.8`,
FFTW `prefix/usr/lib/x86_64-linux-gnu/libfftw3.so.3`.
New task directory: `/home/lysk-admin/drone-experiments/keyfinder-reference-fix-20261113-v1`.
Seal source/protocol/test/guardian hashes before compilation. Record actual compiler,
header, linked-library and resulting binary hashes in the execution receipt.

Build: `g++ -std=c++17 -O2 -Wall -Wextra -Werror -I BASE/libkeyfinder-2.2.8/src keyfinder_reference.cpp -L BASE/keyfinder-build -Wl,-rpath,BASE/keyfinder-build -Wl,-rpath,BASE/prefix/usr/lib/x86_64-linux-gnu -lkeyfinder -o keyfinder_reference`.
BASE is replaced with the absolute directory above. Run only
`./keyfinder_reference --self-test`; inspect the 25 JSON outputs independently.

## Bounds and stop rule

Compile and self-test sequentially in one task-owned process group on CPUs10–11,
nice19, no more than120 seconds wall/aggregate CPU, 1GiB owned RSS and16MiB task
outputs. Maintain at least40GiB available host memory and32GiB disk. Preserve
vLLM PID/start ticks:1478814/35698640,1481756/35700162,3448680/111966827,
3451013/111967770. Abort only task-owned processes on a breached bound or changed
protected identity. No retries or inference follow automatically. Address-space
and file-size resource limits supplement sampled RSS/output checks; sub-poll
spikes are not guaranteed observable. The source and tests are frozen before build.

## Interpretation of previous benchmark

The old wrapper incorrectly classifies `key >= C_MINOR` as minor although the
actual library enums alternate major/minor. It also subtracts one from every enum
at/above that boundary before tonic conversion. `A_MINOR` becomes A major and
`D_FLAT_MAJOR` becomes C minor. Its 93-track KeyFinder results therefore do not
measure correctly mapped KeyFinder accuracy. Preserve those historical files,
flag the affected comparison as invalid, and do not infer corrected scores from
mapped output: mapping is many-to-one. This finding alone does not invalidate
separate native-reference experiments or the other detectors' predictions.
