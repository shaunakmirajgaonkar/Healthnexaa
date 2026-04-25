# Contributing to medextract

Thank you for your interest in contributing.

---

## Getting Started

```bash
git clone https://github.com/shaunakmirajgaonkar/Healthnexaa.git
cd Healthnexaa
pip install -e ".[dev]"
```

You also need [Ollama](https://ollama.com) running with the model pulled:

```bash
ollama serve
ollama pull medgemma1.5:4b
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 37 tests use mocked Ollama responses — no model or internet required to run the test suite.

Tests run automatically on every push and PR via GitHub Actions (Python 3.10, 3.11, 3.12).

---

## How to Contribute

### Reporting a bug
Open an issue with:
- Python version and OS
- The exact command or code you ran
- The full error message or unexpected output
- A sample image if the issue is extraction-related (remove any personal health data first)

### Suggesting a feature
Open an issue describing the use case. Areas we'd welcome contributions:
- Glucometer support (`analyze_glucose_image()`)
- Additional output fields
- Better handling of glare or blurry images
- Alternative local model support (LLaVA, Moondream, etc.)
- FastAPI web dashboard

### Submitting a pull request
1. Fork the repo and create a branch from `main`
2. Make your changes inside `medextract/`
3. Add or update tests in `tests/test_extractor.py`
4. Ensure all 37+ tests pass: `pytest tests/ -v`
5. Open a PR with a clear description of what changed and why

---

## Code Style

- Python 3.10+
- No global state in library code — all config as function parameters with defaults
- No `logging.basicConfig()` in library modules — use `NullHandler`
- No hardcoded paths — always use function parameters
- Validate parameters at the start of public functions (`ValueError` with a clear message)
- Confidence values must be clamped to 1–10 before returning
- New features must include tests — mock Ollama with `unittest.mock.patch`

---

## Privacy Note

Do not commit real patient photos or health data to this repository. Use anonymised or synthetic test images only. Patient data is PHI and must be handled according to applicable regulations (HIPAA in the US).
