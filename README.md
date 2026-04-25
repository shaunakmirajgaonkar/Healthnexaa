# medextract

Extract blood pressure readings from medical device photos using a local AI model — no API key, no internet, no cost.

Built on [Ollama](https://ollama.com) + MedGemma, the library processes photos of BP monitors and returns structured data (systolic, diastolic, pulse, brand, AHA classification, and 10 more fields) ready for CSV export or direct use in Python.

---

## Features

- **Fully local** — runs on your machine via Ollama, no data sent to any server
- **No API key required** — free to use with no rate limits
- **14 fields extracted per image** — systolic, diastolic, pulse, brand, date, time, IHB, AFib, battery, glare, confidence, and more
- **Parallel processing** — multiple images processed simultaneously
- **AHA BP classification** — Normal → Elevated → Stage 1/2 → Hypertensive Crisis
- **Validation** — flags out-of-range and physiologically impossible readings
- **Works as a Python library or CLI tool**

---

## Prerequisites

Install [Ollama](https://ollama.com), then pull the model:

```bash
ollama serve
ollama pull medgemma1.5:4b
```

---

## Install

**From GitHub:**
```bash
pip install git+https://github.com/shaunakmirajgaonkar/Healthnexaa.git
```

**For local development:**
```bash
git clone https://github.com/shaunakmirajgaonkar/Healthnexaa.git
cd Healthnexaa
pip install -e .
```

**Dependencies installed automatically:** `ollama`, `pillow`, `pandas`

---

## Quickstart

### Python

```python
from medextract import extract_folder

rows = extract_folder("/path/to/bp-monitor-photos")

for row in rows:
    print(row["systolic"], "/", row["diastolic"], "—", row["bp_classification"])
```

### Single image

```python
from medextract import analyze_image, classify_bp

result = analyze_image("/path/to/photo.jpg")

if result:
    print("Systolic :", result["systolic"])
    print("Diastolic:", result["diastolic"])
    print("Pulse    :", result["pulse"])
    print("Brand    :", result["brand"])
    print("Category :", classify_bp(result["systolic"], result["diastolic"]))
    print("Confidence:", result["confidence"], "/ 10")
```

### Save to CSV

```python
import pandas as pd
from medextract import extract_folder

rows = extract_folder("/path/to/photos")
pd.DataFrame(rows).to_csv("results.csv", index=False)
```

### 1000+ images — progress bar and resume

```python
from medextract import extract_folder

# tqdm progress bar shown automatically
rows = extract_folder("/path/to/1000-photos", workers=5)
```

If the run crashes midway, resume without reprocessing done images:

```python
rows = extract_folder(
    "/path/to/photos",
    resume_csv="results.csv",  # skips files already in this CSV
)
```

### CLI

```bash
# Basic
python3 -m medextract.cli /path/to/photos --output results.csv --workers 4

# Resume a crashed run
python3 -m medextract.cli /path/to/photos --output results.csv --resume
```

---

## Output Fields

| Field | Type | Description |
|---|---|---|
| `file_name` | string | Source image filename |
| `systolic` | int | Top BP number (0 if unreadable) |
| `diastolic` | int | Bottom BP number (0 if unreadable) |
| `pulse` | int | Pulse / BPM (0 if unreadable) |
| `brand` | string | Device brand |
| `date` | string | Date shown on device |
| `time` | string | Time shown on device |
| `memory_slot` | string | M1, M2, or null |
| `ihb` | bool | Irregular heartbeat indicator |
| `afib` | bool | AFib indicator |
| `battery_low` | bool | Battery warning |
| `has_glare` | bool | Glare affecting readability |
| `confidence` | int 1–10 | Model confidence in the reading |
| `bp_classification` | string | AHA category |
| `extracted_at` | string | Timestamp of extraction |

---

## BP Classification (AHA Standard)

| Classification | Systolic | Diastolic |
|---|---|---|
| Normal | < 120 | < 80 |
| Elevated | 120–129 | < 80 |
| Stage 1 Hypertension | 130–139 | 80–89 |
| Stage 2 Hypertension | ≥ 140 | ≥ 90 |
| HYPERTENSIVE CRISIS | > 180 | > 120 |

---

## Supported Image Formats

`.png` `.jpg` `.jpeg` `.webp`

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

37 tests covering `classify_bp`, `validate_bp`, image loading, Ollama checks, input validation, resume, and `analyze_image` with mocked responses.

Tests run automatically on every push and pull request via GitHub Actions across Python 3.10, 3.11, and 3.12.

---

## Project Structure

```
medextract/          ← installable Python library
├── __init__.py      ← public API
├── extractor.py     ← core extraction logic
└── cli.py           ← command-line interface

tests/               ← pytest test suite (31 tests)
└── test_extractor.py

examples/            ← original standalone scripts (reference only)
├── README.md
└── *.py

plans/               ← project documentation
├── usage-guide.md
├── library-creation-guide.md
├── project-plan.md
└── medical-device-extraction-guide.md

pyproject.toml       ← package config
```

---

## License

MIT — see [LICENSE](LICENSE)
