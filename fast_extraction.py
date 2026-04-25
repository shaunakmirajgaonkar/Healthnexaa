"""
High-Speed BP Monitor Extraction — Async Gemini API
Processes 500-1000+ images/min using concurrent async requests.

Setup:
  pip install google-generativeai pillow pandas aiofiles
  GEMINI_API_KEY="your-key" python fast_extraction.py

Get free key: https://aistudio.google.com
Upgrade to paid for 1000+ RPM: https://ai.google.dev/pricing
"""

import os
import json
import logging
import asyncio
import base64
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai

# ==========================================
# Configuration
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
IMAGE_FOLDER   = Path(os.getenv("IMAGE_FOLDER", "./images"))
OUTPUT_CSV     = "medical_data_export.csv"
FAILED_LOG     = "failed_images.txt"
MODEL          = "gemini-2.0-flash"
CONCURRENCY    = 50   # simultaneous requests — increase to 100+ on paid tier

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
model = genai.GenerativeModel(MODEL)

PROMPT = """You are a medical device reader. Analyze this blood pressure monitor image and extract every visible value.
Return ONLY a valid JSON object — no markdown, no extra text:
{
  "systolic":    <int, 0 if unreadable>,
  "diastolic":   <int, 0 if unreadable>,
  "pulse":       <int, 0 if unreadable>,
  "brand":       "<string, Unknown if not visible>",
  "date":        "<YYYY-MM-DD or null>",
  "time":        "<HH:MM or null>",
  "memory_slot": "<M1, M2, or null>",
  "ihb":         <true if irregular heartbeat symbol shown, else false>,
  "afib":        <true if afib indicator shown, else false>,
  "error_code":  "<E1-E5 string or null>",
  "user":        "<User 1, User 2, or null>",
  "battery_low": <true if battery warning shown, else false>,
  "has_glare":   <true if glare affects readability, else false>,
  "confidence":  <int 1-10>
}"""

COLUMN_ORDER = [
    "file_name", "systolic", "diastolic", "pulse", "brand",
    "date", "time", "memory_slot", "ihb", "afib",
    "error_code", "user", "battery_low", "has_glare", "confidence",
    "bp_classification", "extracted_at",
]

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ==========================================
# Image loading
# ==========================================
def load_image(image_path: Path) -> Image.Image | None:
    try:
        img = Image.open(image_path)
        img.thumbnail((512, 512), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, Exception) as e:
        log.warning("Cannot open %s: %s", image_path.name, e)
        return None


# ==========================================
# Async extraction
# ==========================================
async def analyze_screen(semaphore: asyncio.Semaphore, image_path: Path) -> dict | None:
    img = load_image(image_path)
    if img is None:
        return None

    async with semaphore:
        for attempt in range(1, 4):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content([PROMPT, img])
                )
                raw = response.text.strip().replace("```json", "").replace("```", "").strip()
                start, end = raw.find("{"), raw.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError("No JSON in response")
                data = json.loads(raw[start:end])
                data["file_name"] = image_path.name
                return data
            except Exception as e:
                log.warning("[%s] Attempt %d/3: %s", image_path.name, attempt, e)
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)

    log.error("Skipping %s: all retries exhausted.", image_path.name)
    return None


def classify_bp(systolic: int, diastolic: int) -> str:
    if systolic > 180 or diastolic > 120: return "HYPERTENSIVE CRISIS"
    elif systolic >= 140 or diastolic >= 90: return "Stage 2 Hypertension"
    elif systolic >= 130 or diastolic >= 80: return "Stage 1 Hypertension"
    elif systolic >= 120 and diastolic < 80: return "Elevated"
    elif systolic > 0: return "Normal"
    return "Unknown"


def validate_bp(row: dict) -> list[str]:
    warnings = []
    s, d, p = row.get("systolic", 0), row.get("diastolic", 0), row.get("pulse", 0)
    if s and not (60 <= s <= 300): warnings.append(f"systolic {s} out of range")
    if d and not (30 <= d <= 200): warnings.append(f"diastolic {d} out of range")
    if p and not (30 <= p <= 250): warnings.append(f"pulse {p} out of range")
    if s and d and s <= d: warnings.append("systolic must be > diastolic")
    return warnings


# ==========================================
# Main async runner
# ==========================================
async def main():
    if not IMAGE_FOLDER.exists():
        log.error("Folder not found: %s", IMAGE_FOLDER)
        return

    images = sorted([f for f in IMAGE_FOLDER.iterdir() if f.suffix.lower() in EXTENSIONS])
    if not images:
        log.error("No images found in %s", IMAGE_FOLDER)
        return

    log.info("Model      : %s", MODEL)
    log.info("Concurrency: %d simultaneous requests", CONCURRENCY)
    log.info("Images     : %d\n", len(images))

    semaphore = asyncio.Semaphore(CONCURRENCY)
    start_time = asyncio.get_event_loop().time()

    tasks = [analyze_screen(semaphore, img) for img in images]
    results = await asyncio.gather(*tasks)

    elapsed = asyncio.get_event_loop().time() - start_time
    rate = len(images) / elapsed * 60

    extracted, failed = [], []
    for img_path, result in zip(images, results):
        if result:
            warnings = validate_bp(result)
            result["bp_classification"] = classify_bp(result.get("systolic", 0), result.get("diastolic", 0))
            result["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if warnings:
                log.warning("[%s] %s", img_path.name, "; ".join(warnings))
            extracted.append(result)
        else:
            failed.append(img_path.name)

    log.info("\nDone: %d succeeded, %d failed in %.1fs (%.0f images/min)",
             len(extracted), len(failed), elapsed, rate)

    if failed:
        Path(FAILED_LOG).write_text("\n".join(failed))
        log.warning("Failed images: %s", FAILED_LOG)

    if not extracted:
        log.warning("No data extracted.")
        return

    df = pd.DataFrame(extracted)
    final_cols = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[final_cols]
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved: %s", OUTPUT_CSV)
    print(df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())