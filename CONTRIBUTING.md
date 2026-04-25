# Contributing to medextract

Thank you for your interest in contributing.

---

## Getting Started

```bash
git clone https://github.com/shaunakmirajgaonkar/Healthnexaa.git
cd Healthnexaa
pip install -e .
```

You also need [Ollama](https://ollama.com) running with the model pulled:

```bash
ollama serve
ollama pull medgemma1.5:4b
```

---

## How to Contribute

### Reporting a bug
Open an issue with:
- Python version and OS
- The exact command or code you ran
- The full error message or unexpected output
- A sample image if the issue is extraction-related (remove any personal health data)

### Suggesting a feature
Open an issue describing the use case. Common areas we'd welcome contributions:
- Support for glucometers or other medical devices
- Additional output fields
- Better handling of glare or blurry images
- Alternative local model support (LLaVA, Moondream, etc.)

### Submitting a pull request
1. Fork the repo and create a branch from `main`
2. Make your changes in `medextract/`
3. Test on at least one real image with `ollama serve` running
4. Open a PR with a clear description of what changed and why

---

## Code Style

- Python 3.10+
- No global state in library code — all config as function parameters
- No `logging.basicConfig()` in library modules — use `NullHandler`
- No hardcoded paths — use environment variables or function parameters

---

## Privacy Note

Do not commit real patient photos or health data to this repository. Use anonymised or synthetic test images only.
