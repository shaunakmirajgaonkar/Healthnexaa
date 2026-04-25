# Changelog

All notable changes to this project will be documented here.

---

## [0.2.0] — 2026-04-25

### Added
- `check_ollama()` — raises a clear `RuntimeError` if Ollama is not running or the model is not pulled
- `tests/` — full pytest suite with 31 tests covering all public functions (mocked Ollama)
- `examples/` folder — old standalone scripts moved here with a README
- `[dev]` optional dependency group: `pytest>=8.0`, `pytest-mock>=3.12`

### Changed
- Confidence field now always clamped to 1–10 regardless of model output
- Dependency versions pinned in `pyproject.toml` (`ollama>=0.6.1`, `pillow>=10.2.0`, `pandas>=2.1.1`)
- `extract_folder()` now calls `check_ollama()` before processing — fails fast with a helpful message
- Old scripts moved from repo root to `examples/` to reduce clutter

---

## [0.1.0] — 2026-04-25

### Added
- `medextract` Python library with full pip-installable package structure
- `extract_folder()` — batch process all images in a folder with parallel workers
- `analyze_image()` — extract readings from a single image
- `classify_bp()` — AHA blood pressure classification (Normal → Hypertensive Crisis)
- `validate_bp()` — flag out-of-range or physiologically impossible readings
- CLI entry point: `python3 -m medextract.cli`
- `pyproject.toml` packaging config
- 14-field extraction schema: systolic, diastolic, pulse, brand, date, time, memory_slot, ihb, afib, error_code, user, battery_low, has_glare, confidence
- `plans/` documentation folder with 4 guides

### Changed
- Switched backend from OpenAI GPT-4o / Gemini API to local Ollama + MedGemma 1.5 4b
- Removed all hardcoded personal paths — all paths now read from environment variables
- Replaced global script constants with function parameters
- Replaced module-level `logging.basicConfig()` with `NullHandler` (library best practice)

### Fixed
- `display(df)` crash outside Jupyter — replaced with `df.to_string()`
- Output CSV and failed log no longer hardcoded to Desktop path

---

## [0.0.1] — 2026-04-20

### Added
- Initial single-file script (`bp-monitor-extraction`) using Gemini 1.5 Flash
- 5-field extraction: systolic, diastolic, pulse, brand, has_glare
- Exponential backoff retry (up to 5 attempts)
- CSV export to working directory
