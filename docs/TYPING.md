# Type checking

Run `uv sync --locked --all-extras`, then `uv run ty check .`. CI runs the same
whole-repository check, including research tools and tests. The `research` extra
pins nnAudio 0.3.4, which the octave research frontend imports; ordinary production
installation does not require that extra.

Every Python function has explicit parameter and return annotations, including
the training CLI, historical research modules, tools, nested helpers and tests.
Implicit `self`/`cls` parameters need no annotation; lambdas do not support
annotation syntax and are checked in the context of their callers.

CI enforces two separate properties: `uv run ruff check .` includes the full
`ANN` rule set, which rejects missing parameter/return annotations and direct
`Any` parameter annotations; `uv run ty check .` checks type compatibility.
Removing an annotation therefore fails CI even when ty could infer a type.

Heterogeneous JSON/checkpoint records use explicit `dict[str, Any]` annotations
where their fields are loaded dynamically. Scientific tests narrow concrete
PyTorch layers and gradients before accessing them. NumPy integer arrays remain
valid scoring inputs; callers do not need to convert them to Python lists.

No research directories or annotation rules are excluded to make the checks pass.
Annotations do not replace runtime validation, numerical parity or the full test
suite. The existing feature contracts, model weights and experiment protocols
are unchanged; source snapshots acquire new hashes when rerunning annotated code.
