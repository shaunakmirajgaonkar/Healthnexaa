"""
BP Monitor Image Extraction
Uses Google Gemini 1.5 Flash (free tier) — get key at aistudio.google.com
"""

import io
import os
import time
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai

# ==========================================
# Configuration — edit these two lines
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
IMAGE_FOLDER   = Path(os.getenv("IMAGE_FOLDER", "/Users/shaunak/Desktop"))
OUTPUT_CSV     = "medical_data_export.csv"
FAILED_LOG     = "failed_images.txt"

# ==========================================
# Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """Analyze this blood pressure monitor image and extract every visible value.
Return ONLY a valid JSON object — no markdown, no extra text:
{
  "systolic":    <int, 0 if unreadable>,
  "diastolic":   <int, 0 if unreadable>,
  "pulse":       <int, 0 if unreadable>,
  "brand":       "<string, 'Unknown' if not visible>",
  "date":        "<YYYY-MM-DD or null>",
  "time":        "<HH:MM or null>",
  "memory_slot": "<M1, M2, or null>",
  "ihb":         <true if irregular heartbeat symbol shown, else false>,
  "afib":        <true if afib indicator shown, else false>,
  "error_code":  "<E1-E5 string or null>",
  "user":        "<User 1, User 2, or null>",
  "battery_low": <true if battery warning shown, else false>,
  "has_glare":   <true if glare affects readability, else false>,
  "confidence":  <int 1-10, your confidence in the reading>
}"""

COLUMN_ORDER = [
    "file_name", "systolic", "diastolic", "pulse", "brand",
    "date", "time", "memory_slot", "ihb", "afib",
    "error_code", "user", "battery_low", "has_glare", "confidence",
]

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_RETRIES = 5
RATE_LIMIT_SLEEP = 4  # seconds — keeps under 15 RPM free tier limit


# ==========================================
# Core extraction
# ==========================================
def load_image(image_path: Path) -> Image.Image | None:
    try:
        img = Image.open(image_path)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, Exception) as e:
        log.warning("Cannot open %s: %s", image_path.name, e)
        return None


def analyze_screen(image_path: Path) -> dict | None:
    img = load_image(image_path)
    if img is None:
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content([PROMPT, img])
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            data["file_name"] = image_path.name
            return data
        except Exception as e:
            wait = (2 ** attempt) * 5
            log.warning("Attempt %d/%d failed for %s: %s. Retrying in %ds.",
                        attempt, MAX_RETRIES, image_path.name, e, wait)
            time.sleep(wait)

    log.error("Skipping %s: all retries exhausted.", image_path.name)
    return None


def validate_bp(row: dict) -> list[str]:
    warnings = []
    s, d, p = row.get("systolic", 0), row.get("diastolic", 0), row.get("pulse", 0)
    if s and not (60 <= s <= 300):
        warnings.append(f"systolic {s} out of range")
    if d and not (30 <= d <= 200):
        warnings.append(f"diastolic {d} out of range")
    if p and not (30 <= p <= 250):
        warnings.append(f"pulse {p} out of range")
    if s and d and s <= d:
        warnings.append("systolic must be greater than diastolic")
    return warnings


def classify_bp(systolic: int, diastolic: int) -> str:
    if systolic > 180 or diastolic > 120:
        return "HYPERTENSIVE CRISIS"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    elif systolic >= 130 or diastolic >= 80:
        return "Stage 1 Hypertension"
    elif systolic >= 120 and diastolic < 80:
        return "Elevated"
    elif systolic > 0:
        return "Normal"
    return "Unknown"


# ==========================================
# Main
# ==========================================
def main():
    if not IMAGE_FOLDER.exists():
        log.error("Folder not found: %s", IMAGE_FOLDER)
        return

    images = sorted([f for f in IMAGE_FOLDER.iterdir() if f.suffix.lower() in EXTENSIONS])
    if not images:
        log.error("No images found in %s", IMAGE_FOLDER)
        return

    log.info("Found %d image(s) in %s", len(images), IMAGE_FOLDER)
    log.info("Estimated time: ~%ds (4s/image + retries)\n", len(images) * 4)

    extracted, failed = [], []

    for i, img_path in enumerate(images, 1):
        log.info("[%d/%d] Processing: %s", i, len(images), img_path.name)
        result = analyze_screen(img_path)

        if result:
            warnings = validate_bp(result)
            result["bp_classification"] = classify_bp(result.get("systolic", 0), result.get("diastolic", 0))
            result["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if warnings:
                log.warning("  Validation: %s", "; ".join(warnings))
            log.info("  BP %s/%s  Pulse %s  Brand: %s  Confidence: %s/10  [%s]",
                     result.get("systolic"), result.get("diastolic"),
                     result.get("pulse"), result.get("brand"),
                     result.get("confidence"), result.get("bp_classification"))
            extracted.append(result)
        else:
            failed.append(img_path.name)

        if i < len(images):
            time.sleep(RATE_LIMIT_SLEEP)

    # Save results
    log.info("\nExtraction complete: %d succeeded, %d failed.", len(extracted), len(failed))

    if failed:
        Path(FAILED_LOG).write_text("\n".join(failed))
        log.warning("Failed images logged to: %s", FAILED_LOG)

    if not extracted:
        log.warning("No data extracted. CSV not saved.")
        return

    df = pd.DataFrame(extracted)
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    final_cols = COLUMN_ORDER + ["bp_classification", "extracted_at"] + extra_cols
    df = df[[c for c in final_cols if c in df.columns]]

    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved to: %s\n", OUTPUT_CSV)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
