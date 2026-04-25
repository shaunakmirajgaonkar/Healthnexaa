# Project Summary — medextract

## What We Built

Started with a **single messy script** that used the OpenAI API to extract 5 fields from BP monitor photos. Ended with a **fully published Python library** anyone in the world can install.

```bash
pip install medextract
```

A production-ready Python library that extracts blood pressure readings from medical device photos using a **local AI model** (Ollama + MedGemma). No API key, no internet, no cost.

---

## Journey — Phase by Phase

| Phase | What Happened |
|---|---|
| **Phase 1 — Initial Script** | Switched from OpenAI API → local Ollama + MedGemma. No more API keys, costs, or rate limits |
| **Phase 2 — Hardening** | Expanded from 5 fields → 14 fields. Added parallel processing, retry logic, BP classification |
| **Phase 3 — Python Library** | Converted script into installable `medextract` Python package with clean public API |
| **Phase 4 — Production Ready** | Added 37 tests, GitHub Actions CI, input validation, progress bar, resume support |
| **Phase 5 — PyPI Publish** | Published to PyPI — `pip install medextract` live worldwide |
| **v0.3.1 Bug Fix** | Found `bp_classification` missing from `analyze_image()` during real external user test — fixed and republished |

---

## What the Library Does

- Extracts **14 fields** per image — systolic, diastolic, pulse, brand, date, time, IHB, AFib, battery, glare, confidence, and more
- Classifies BP using **AHA standard** (Normal → Elevated → Stage 1 → Stage 2 → Hypertensive Crisis)
- Processes **1000+ images in parallel** with a live progress bar and ETA
- **Resumes crashed batches** without reprocessing completed images
- Works as a **Python import or CLI command**
- Runs **100% locally** — no data ever leaves the machine

---

## How to Use

### Install
```bash
pip install medextract
```

### Single image
```python
from medextract import analyze_image

result = analyze_image("/path/to/photo.jpg")
print(result["systolic"], "/", result["diastolic"])
print("Category:", result["bp_classification"])
print("Confidence:", result["confidence"], "/ 10")
```

### Whole folder
```python
from medextract import extract_folder
import pandas as pd

rows = extract_folder("/path/to/photos", workers=5)
pd.DataFrame(rows).to_csv("results.csv", index=False)
```

### CLI
```bash
python3 -m medextract.cli /path/to/photos --output results.csv
```

---

## Output Fields (14 per image)

| Field | Type | Description |
|---|---|---|
| `systolic` | int | Top BP number |
| `diastolic` | int | Bottom BP number |
| `pulse` | int | Pulse / BPM |
| `brand` | string | Device brand |
| `date` | string | Date shown on device |
| `time` | string | Time shown on device |
| `memory_slot` | string | M1, M2, or null |
| `ihb` | bool | Irregular heartbeat indicator |
| `afib` | bool | AFib indicator |
| `battery_low` | bool | Battery warning |
| `has_glare` | bool | Glare affecting readability |
| `confidence` | int 1–10 | Model confidence (clamped) |
| `bp_classification` | string | AHA category |
| `extracted_at` | string | Timestamp of extraction |

---

## Final Numbers

| Metric | Value |
|---|---|
| Tests | 37 / 37 passing |
| CI | GitHub Actions — Python 3.10, 3.11, 3.12 |
| PyPI | `pip install medextract` |
| Version | `0.3.1` |
| Fields extracted | 14 per image |
| Verified on real devices | Omron 140/80 ✅, Meditech BP-12 130/80 ✅ |
| Cost to run | $0 |

---

## Before vs After

| Before (script) | After (library) |
|---|---|
| OpenAI API — costs money, needs key | Local Ollama — free, no key, no internet |
| 5 fields extracted | 14 fields extracted |
| Single flat `.py` file | Installable `medextract` package |
| No tests | 37 tests — all passing |
| No input validation | Clear `ValueError` before processing starts |
| No progress feedback | tqdm progress bar with live BP readout |
| No resume | Resumes crashed batches from existing CSV |
| No CI | GitHub Actions across Python 3.10, 3.11, 3.12 |
| Run with `python script.py` | `pip install medextract` + import anywhere |

---

## Links

- **PyPI:** https://pypi.org/project/medextract/0.3.1/
- **GitHub:** https://github.com/shaunakmirajgaonkar/Healthnexaa
- **Install:** `pip install medextract`

---

*Built: April 2026 | Author: Shaunak Mirajgaonkar*
