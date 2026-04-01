from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    infer_text_column,
    load_data,
    preprocess_text_column,
    save_csv,
)


def run_preprocessing(input_path: str | None = None) -> Path:
    print("[preprocessing] Loading raw data...")
    df, source_path = load_data(input_path=input_path, data_dir=DEFAULT_RAW_DIR)
    print(f"[preprocessing] Source file: {source_path}")
    print(f"[preprocessing] Shape before cleaning: {df.shape}")
    print("[preprocessing] Sample rows before cleaning:")
    print(df.head(3).to_string(index=False))

    text_column = infer_text_column(df)
    print(f"[preprocessing] Using text column: {text_column}")

    processed = preprocess_text_column(df, text_column=text_column)
    print(f"[preprocessing] Shape after cleaning: {processed.shape}")
    print("[preprocessing] Sample rows after cleaning:")
    preview_columns = [col for col in ["text", "cleaned_text", "token_count"] if col in processed.columns]
    print(processed[preview_columns].head(3).to_string(index=False))

    output_path = DEFAULT_PROCESSED_DIR / "cleaned.csv"
    save_csv(processed, output_path)
    print(f"[preprocessing] Saved cleaned data to: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess the raw NLP review dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to a CSV/JSON file or directory under data/raw.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_preprocessing(args.input)


if __name__ == "__main__":
    main()
