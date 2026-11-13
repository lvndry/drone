# KeyFinder enum conversion repaired and qualified

The new standalone reference CLI correctly maps every libkeyfinder 2.2.8 key enum
and silence, verified both at compile time and by an actual-library runtime enum
self-test. No recording was decoded or classified. This is an interface
qualification, not a new accuracy result or an end-to-end audio qualification.

The [prospective protocol](PROTOCOL.md) and [frozen input hashes](frozen.json)
were copied into a new task-owned directory before compilation. The executed
[guardian source](qualify.py.txt) is preserved byte-for-byte as a text artifact;
its original filename was `qualify.py`. Source/test/protocol hashes in the
[receipt](receipt.json) bind the committed files. The existing wrapper, shared
libraries, environments, release weights and user-owned benchmark report were
not modified. No branch was pushed by this qualification.

## Executed result

- Server binary: `/home/lysk-admin/drone-experiments/keyfinder-reference-fix-20261113-v1/keyfinder_reference`.
- Binary SHA256: `3be7ac37598aeef09af1b9f2f002e139d5b920888a6fef197f40e70c66a1764a`.
- C++ source SHA256: `a407aebf8b877d392f526b9df9885b4fd5094153735bd5a50fa8cfa29d54a781`.
- Runtime enum output SHA256: `0a1b10013b743fa70f32da1a397f753d64b14d820366f524cb350f6b695d58b3`.
- Compile/self-test wall:0.623288 seconds; child CPU:0.313313 seconds.
- Sampled aggregate peak RSS:94,969,856 bytes; this is sampled, not an exact peak.
- CPUs10–11, nice19; all four protected vLLM start identities passed throughout
  the short run and at completion. Host/disk reserve and output checks passed.
- Actual header hashes, compiler executable hash and exact build argv are in the
  receipt. [Loader inspection](linked-libraries.txt) resolves libkeyfinder and
  FFTW to the pinned paths; no library substitution occurred.

The 25 [JSON lines](self-test.jsonl) cover all24 keys and silence. Local tests
compile a synthetic header ABI, independently check all output labels/classes,
reject malformed WAV before any stub classifier invocation, and require compilation
failure after deliberately swapping A-major/A-minor enum values. Both tests pass;
scoped Ruff and ty checks pass. These tests do not invoke actual audio inference.

## Consumer contract

Run the binary with one mono PCM16 RIFF/WAVE pathname. Parse exactly one JSON
object with schema `drone.keyfinder-reference.v1`, integer raw enum and status
`key` or `silence`. Key results carry C-first interleaved `drone_class`0…23,
sharp tonic and major/minor. Silence has null tonic/mode/class. Nonzero exit is
an error and never a missing prediction silently dropped from denominators.
Do not pass stereo WAV: the later evaluator must explicitly freeze its common
mono decoding/resampling policy and the treatment of silence/errors.

The original wrapper source uses a range comparison against C_MINOR rather than
alternating-mode semantics, and shifts several major tonics too. Its existing
93-track KF accuracy is invalid as a correctly mapped KeyFinder comparison.
Preserve the original report as historical evidence; do not manufacture corrected
scores by remapping old outputs, because multiple raw enums collapse to the same
wrong output. Separate native-reference experiments and the other detectors are
not invalidated merely by this wrapper finding.

The library's default `keyOfAudio` algorithm is unchanged. The new wrapper accepts
only explicit, validated mono PCM16, preserves the signed sample amplitude
conversion and includes raw enum in output for future audits. Future GTZAN scoring
must separately freeze the audio path, artifact versions, cohort and evaluator.
