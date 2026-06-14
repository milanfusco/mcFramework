# AGENTS.md

## Cursor Cloud specific instructions

`mcframework` is a pure-Python Monte Carlo simulation library (package in `src/mcframework`,
installed editable). Demos live in `demos/` and the test suite in `tests/`. There is no
long-running server; the "application" is the library plus the demo scripts.

### Environment notes
- Dependencies are installed into the system Python with `pip --break-system-packages`
  (Ubuntu's Python is PEP 668 externally-managed). There is no virtualenv.
- Console entry points (`pytest`, `ruff`, `pylint`) install to `~/.local/bin`, which is **not**
  on `PATH`. Invoke tools via the module form: `python3 -m pytest`, `python3 -m ruff`,
  `python3 -m pylint`.
- The update script pins `matplotlib<3.11`. The demo plotting scripts call
  `boxplot(..., labels=...)`, which matplotlib 3.11 removed (renamed to `tick_labels`).
  With `matplotlib<3.11` (still satisfies the project's `>=3.7`) the demos run. Do not bump
  matplotlib to 3.11+ unless the demo code is updated.

### Lint / test / run (standard commands documented in README.md / CONTRIBUTING.md)
- Lint (the gating check in CI): `python3 -m ruff check src/ tests/`. Pylint
  (`python3 -m pylint src/mcframework tests/`) is informational only — CI runs it with
  `continue-on-error`, so its non-zero exit on test-file style warnings is expected.
- Tests: `python3 -m pytest`. GPU backend tests (`test_torch_cuda.py`, `test_torch_mps.py`)
  auto-skip because there is no CUDA/MPS hardware (~37 skips is normal).
- Demos (headless): `MPLBACKEND=Agg python3 demos/demo.py` (it calls `plt.show()` and
  `input()`; with `Agg` the show is a no-op, and piping input avoids the save prompt).

### Known environment limitation (no swap)
- This VM has 16GB RAM and **no swap**. `tests/test_torch_cpu.py::TestTorchNumPyParity::test_torch_stats_computation_works`
  runs the default stats engine's bootstrap CI on 100k samples, which momentarily allocates a
  `(10_000, 100_000)` index array plus its gather (~16GB peak) and gets OOM-killed here. It
  passes on GitHub runners because they have swap. Deselect it when running the full suite:
  `python3 -m pytest --deselect "tests/test_torch_cpu.py::TestTorchNumPyParity::test_torch_stats_computation_works"`.
  This is an environment constraint, not a code defect — do not "fix" the library for it.
