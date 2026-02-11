<!-- Auto-generated guidance for AI coding agents. Edit with care. -->
# Copilot instructions — medGemma

Purpose: give an AI coding agent the minimal, actionable context to be immediately productive in this repo.

Big picture
- Single-process ETL-style pipeline implemented under `src/`.
- Key stages are split into modules: `src/extract.py`, `src/validate.py`, `src/detect.py`, `src/tag.py`, `src/aggregate.py`, `src/io.py`.
- Entrypoint: `src/run_pipeline.py` (currently a stub that prints a status message).
- Data lives in `data_raw/` (inputs) and `data_synth/` (synthetic/test fixtures). Notebooks live in `notebooks/`.

Dataflow & patterns (discoverable)
- Expected flow: extract -> validate -> detect -> tag -> aggregate -> io (output). Keep functions pure where possible and return pandas DataFrames rather than writing files inside lower-level functions.
- Common return type: `pandas.DataFrame` for tabular transforms. Use `pydantic`/`jsonschema` defined under `schemas/` for schema validation when available.
- Function signatures: prefer explicit inputs and outputs. Example patterns to follow:
  - `def extract(path: str) -> pd.DataFrame`
  - `def validate(df: pd.DataFrame) -> pd.DataFrame`
  - `def aggregate(dfs: List[pd.DataFrame]) -> pd.DataFrame`

Developer workflows (how to run & iterate)
- Install deps: `pip install -r requirements.txt` (see `requirements.txt` — pandas, numpy, pydantic, jsonschema).
- Run the pipeline stub locally: `python src/run_pipeline.py` or `python -m src.run_pipeline`.
- Open and run the notebook: `notebooks/01_end_to_end_stub.ipynb` for exploratory runs and quick checks.

Project-specific conventions
- Repo is a scaffold: many `src/*.py` are stubs. Add unitable, importable functions — avoid top-level side-effects so tests and notebooks can import modules.
- Use `schemas/` for canonical data shapes if present. Prefer `pydantic` models for structured records and `jsonschema` for file-level validation.
- Keep I/O centralized in `src/io.py`; transforms should not perform file writes directly unless explicitly the I/O layer.

Integration points & dependencies
- External libs: `pandas`, `numpy`, `pydantic`, `jsonschema`. No network services or cloud integrations detected in repo.
- Data files: `data_raw/` (source), `data_synth/` (fixtures). Use `data_synth/` for reproducible tests/notebooks.

Editing & PR guidance for agents
- Make minimal, focused edits; keep changes module-local and update `README.md` or this file when adding new developer-facing behaviors.
- If you implement behavior in a module (e.g. `src/extract.py`), add a small notebook cell or a short script in `notebooks/` demonstrating the API with sample data from `data_synth/`.
- Add tests under a `tests/` directory (not present yet) when adding non-trivial logic.

Examples observed in this repo
- `src/run_pipeline.py` is the canonical entrypoint (currently prints "Pipeline stub: repo scaffold is ready.").
- `requirements.txt` lists runtime libs: pandas, numpy, pydantic, jsonschema.

If anything here is unclear or you want more detail (test commands, CI hooks, or data examples), ask and I'll expand specific sections.
