# Type checking

Run `uv sync --locked --all-extras`, then `uv run ty check .`. CI runs the same
whole-repository check, including research tools and tests. The `research` extra
pins nnAudio 0.3.4, which the octave research frontend imports; ordinary production
installation does not require that extra.

The production package, training CLI and K10 tests have explicit function
annotations. Historical research modules are checked too, but are not fully
annotated: ty can analyze unannotated functions and does not require every
function to have annotations. A passing check is not a claim of complete
annotation coverage or runtime validation.

Heterogeneous JSON/checkpoint records use explicit `dict[str, Any]` annotations
where their fields are loaded dynamically. Scientific tests narrow concrete
PyTorch layers and gradients before accessing them. NumPy integer arrays remain
valid scoring inputs; callers do not need to convert them to Python lists.

No repository directories or diagnostics are excluded to make the check pass.
The full runtime tests, Ruff and build checks remain separate CI steps.
