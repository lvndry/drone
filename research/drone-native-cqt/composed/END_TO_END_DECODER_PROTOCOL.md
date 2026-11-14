# Prospective original-decoder lane

This amendment adds an explicitly different input-admission lane to the fixed
[five-case measurement](END_TO_END_PROTOCOL.md). The canonical lane, its first
failed Mac attempt and every historical reference remain unchanged. Source audio,
case order, one fresh plus three warm passes, CPU/thread policy, model, captured
plans, comparisons and resource ceilings are unchanged.

The new configuration must use `drone-original-file-measurement-v2` and
`input_lane: decoder_fidelity`. A v1 configuration cannot select this lane.
It records the actual local decoder's PCM length and hash before inference.
PCM sample count and CQT frame count must still match the captured canonical
geometry exactly in both lanes. Only PCM sample values/hash may differ in v2;
any length or frame mismatch stops the run.
Every subsequent repetition must reproduce that first local identity exactly.
The first observation is a repeatability anchor, not a replacement historical
reference or a claim of decoder equivalence. No PCM is trimmed, padded, shifted,
renormalized or altered to force a match.

Each pass records its observed PCM and feature geometry, whether canonical PCM
and frame counts match, feature hashes, logits and both existing reference
comparisons. The historical `atol=1e-5, rtol=1e-5` checks and key decisions stay
unchanged. Failed identity or geometry retains its WAV and stops the invocation.
This worker retires admitted WAVs only after frame admission; the separately
recorded retirement cost is diagnostic overhead outside CQT and ONNX timers.

Completion and qualification are separate:

- `timing_complete` means all fixed operations finished within the existing
  resource limits, so their execution costs can be reported.
- `numerical_pass`, `unchanged_key` and `historical_reference_qualified` retain
  the original comparisons against both canonical Mac and Linux reference logits.
- `canonical_pcm_matches` explicitly reports whether preprocessing reproduced
  the historical input. A numerical match does not turn this flag into true.
- The retained `benchmark_qualified` field is an alias for historical numerical
  qualification, not a condition for publishing completed, unqualified costs.

All failed comparisons remain visible. No tolerances are fitted to these five
cases, and no reference or model changes. Neither the preceding one-track
decoder diagnostic nor five duration timings establishes portable production
accuracy. A future shipping decision needs a predeclared original-file comparison
across the full public panel and independent full-track evidence.

The fixed panel's source metadata must accompany results. Its two-minute
Beatport item is a preview; the FMAK recordings' full-song status is unverified.
Their raw genre tags do not establish coverage of house, techno, hip-hop or all
genres. BPM supplied by a publisher is not a measured reference BPM.
