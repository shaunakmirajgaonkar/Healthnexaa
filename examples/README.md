# Examples

These are the original standalone scripts used before the `medextract` library was built.
They are kept here for reference only.

**For new projects, use the library instead:**

```python
from medextract import extract_folder, analyze_image
```

See the main [README](../README.md) and [usage guide](../plans/usage-guide.md) for full documentation.

---

| Script | Model | Notes |
|---|---|---|
| `bp-monitor-extraction-gemini` | Gemini 1.5 Flash | Original Colab/Jupyter script, requires Gemini API key |
| `bp-monitor-extraction-gemini.py` | Gemini 1.5 Flash | Same as above, plain Python version |
| `fast-extraction-gemini.py` | Gemini 1.5 Flash | Faster variant with batch support |
| `image-extraction-ollama.py` | Ollama (MedGemma) | Local model, no API key needed |
| `image-extraction-ollama-parallel.py` | Ollama (MedGemma) | Local model with parallel processing |
