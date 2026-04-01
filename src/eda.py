from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import run_preprocessing  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_FIGURES_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    infer_text_column,
    load_data,
    plot_text_length_distribution,
    plot_top_words,
    plot_wordcloud,
    preprocess_text_column,
)


def load_clean_data(input_path: str | None = None) -> pd.DataFrame:
    if input_path is not None:
        df, source_path = load_data(input_path=input_path)
        print(f"[eda] Loaded cleaned data from: {source_path}")
        return df

    default_path = DEFAULT_PROCESSED_DIR / "cleaned.csv"
    if default_path.exists():
        df, source_path = load_data(input_path=default_path)
        print(f"[eda] Loaded cleaned data from: {source_path}")
        return df

    print("[eda] Cleaned data not found. Loading raw data and preprocessing as fallback.")
    raw_df, raw_path = load_data(data_dir=DEFAULT_RAW_DIR)
    text_column = infer_text_column(raw_df)
    df = preprocess_text_column(raw_df, text_column=text_column)
    print(f"[eda] Preprocessed raw data from: {raw_path}")
    return df


def run_eda(input_path: str | None = None) -> None:
    print("[eda] Starting EDA stage...")
    df = load_clean_data(input_path)
    print(f"[eda] Data shape: {df.shape}")
    print("[eda] Sample rows:")
    print(df[[col for col in ["text", "cleaned_text"] if col in df.columns]].head(3).to_string(index=False))

    text_column = "cleaned_text" if "cleaned_text" in df.columns else infer_text_column(df)
    texts = df[text_column].fillna("").astype(str)
    lengths = texts.str.len()
    word_counts = Counter(word for text in texts for word in text.split() if word)

    figures_dir = DEFAULT_FIGURES_DIR
    length_fig = figures_dir / "text_length_distribution.png"
    words_fig = figures_dir / "top_words.png"
    cloud_fig = figures_dir / "wordcloud.png"

    print("[eda] Generating text length histogram...")
    plot_text_length_distribution(lengths, length_fig, "Text Length Distribution")

    print("[eda] Generating top words bar plot...")
    plot_top_words(word_counts, words_fig, "Top 20 Most Common Words", top_n=20)

    print("[eda] Generating word cloud...")
    plot_wordcloud(" ".join(texts.tolist()), cloud_fig, "Word Cloud of Cleaned Text")

    print(f"[eda] Saved figures to: {figures_dir}")
    print(f"[eda] Average text length: {lengths.mean():.2f}")
    print(f"[eda] Median text length: {lengths.median():.2f}")
    print("[eda] Top 5 words:")
    for word, count in word_counts.most_common(5):
        print(f"[eda]   {word}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on cleaned text.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to cleaned CSV/JSON. Defaults to outputs/processed/cleaned.csv.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_eda(args.input)


if __name__ == "__main__":
    main()
