# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo contains a single Python script (`bp-monitor-extraction`) that batch-processes blood pressure monitor photos using OpenAI's GPT-4o vision API and exports the extracted readings to a CSV.

The script is designed to run in a Jupyter/Colab environment — it uses `!pip install`, `display(df)`, and prints emoji progress indicators. It is **not** a package; there is no setup.py, pyproject.toml, or test suite.

## Running the script

Install dependencies first (or run the first cell in Colab):

```bash
pip install openai pillow pandas
```

Before running, set two values at the top of the file:
- `API_KEY` — your OpenAI key
- `IMAGE_FOLDER_PATH` — path to the folder of BP monitor images (`.png`, `.jpg`, `.jpeg`)

Run as a plain Python script:

```bash
python bp-monitor-extraction
```

Or open in Jupyter and run cell by cell.

Output is written to `medical_data_export.csv` in the working directory.

## Key design decisions

- **Retry with exponential backoff**: `analyze_screen()` retries up to 5 times on any API/parse error, doubling the wait each attempt (starting at 10s). Images that exceed retries are skipped and counted.
- **Rate-limit courtesy**: a 3-second sleep between images prevents hitting OpenAI rate limits.
- **Image pre-processing**: all images are resized to max 1024×1024 and converted to JPEG (RGB) before base64 encoding, to keep token costs down and handle unusual formats.
- **Strict JSON output**: the prompt instructs GPT-4o to return only a JSON object with fixed keys (`systolic`, `diastolic`, `pulse`, `brand`, `has_glare`). The parser strips markdown fences before calling `json.loads`.
- **Zero-value convention**: unreadable fields are returned as `0` (integers) or `"Unknown"` (brand), not `null`, so the CSV schema stays consistent.
