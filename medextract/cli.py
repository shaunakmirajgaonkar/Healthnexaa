"""medextract/cli.py — command-line entry point."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .extractor import extract_folder

COLUMN_ORDER = [
    "file_name", "systolic", "diastolic", "pulse", "brand",
    "date", "time", "memory_slot", "ihb", "afib",
    "error_code", "user", "battery_low", "has_glare", "confidence",
    "bp_classification", "extracted_at",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="medextract",
        description="Extract readings from BP monitor images using a local LLM.",
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--output", default="medical_data_export.csv", help="Output CSV path (default: medical_data_export.csv)")
    parser.add_argument("--model", default="medgemma1.5:4b", help="Ollama model to use (default: medgemma1.5:4b)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3)")
    parser.add_argument("--image-size", type=int, default=512, help="Max image dimension in pixels (default: 512)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per image (default: 3)")
    parser.add_argument("--resume", action="store_true", help="Skip images already in the output CSV and append new results")
    args = parser.parse_args()

    try:
        rows = extract_folder(
            args.folder,
            model=args.model,
            workers=args.workers,
            image_size=args.image_size,
            max_retries=args.max_retries,
            resume_csv=args.output if args.resume else None,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No data extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
