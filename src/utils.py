from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional
import json
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ADDITIONAL_STOP_WORDS = {
    "food",
    "taste",
    "product",
    "one",
    "use",
    "get",
    "also",
    "would",
    "like",
    "really",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS).union(ADDITIONAL_STOP_WORDS)


def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_csv(df: pd.DataFrame, output_path: Path | str, index: bool = False) -> Path:
    output = Path(output_path)
    ensure_directory(output.parent)
    df.to_csv(output, index=index)
    return output


def _candidate_files(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("*.csv", "*.json", "*.jsonl", "*.ndjson"):
        candidates.extend(sorted(directory.glob(pattern)))
    return candidates


def load_data(input_path: Path | str | None = None, data_dir: Path | str | None = None) -> tuple[pd.DataFrame, Path]:
    if input_path is not None:
        path = Path(input_path)
        if path.is_dir():
            candidates = _candidate_files(path)
            if not candidates:
                raise FileNotFoundError(f"No CSV/JSON files found in {path}")
            path = candidates[0]
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
    else:
        search_dir = Path(data_dir) if data_dir is not None else DEFAULT_RAW_DIR
        if not search_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {search_dir}")
        candidates = _candidate_files(search_dir)
        if not candidates:
            raise FileNotFoundError(f"No CSV/JSON files found in {search_dir}")
        path = candidates[0]

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported data format: {suffix}")

    return df, path


def infer_text_column(df: pd.DataFrame) -> str:
    candidate_columns = [
        "text",
        "review",
        "reviews",
        "content",
        "comment",
        "body",
        "message",
        "sentence",
    ]
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidate_columns:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for column in df.columns:
        if any(token in column.lower() for token in ("text", "review", "content", "comment", "body")):
            return column

    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if object_columns:
        return object_columns[0]

    raise ValueError("Could not infer a text column. Provide a dataset with a text-like column.")


def clean_text(text: object, stop_words: Optional[set[str]] = None) -> tuple[str, list[str]]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "", []

    normalized = str(text).lower()
    tokens = re.findall(r"[a-z0-9']+", normalized)
    stop_words = stop_words or STOP_WORDS
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens), filtered_tokens


def preprocess_text_column(df: pd.DataFrame, text_column: str, stop_words: Optional[set[str]] = None) -> pd.DataFrame:
    processed = df.copy()
    processed["text"] = processed[text_column].astype(str)

    cleaned_texts: list[str] = []
    tokens_column: list[str] = []
    token_counts: list[int] = []

    for raw_text in processed["text"]:
        cleaned_text, tokens = clean_text(raw_text, stop_words=stop_words)
        cleaned_texts.append(cleaned_text)
        tokens_column.append(" ".join(tokens))
        token_counts.append(len(tokens))

    processed["cleaned_text"] = cleaned_texts
    processed["tokens"] = tokens_column
    processed["token_count"] = token_counts
    return processed


def word_frequencies(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        counter.update(token for token in text.split() if token)
    return counter


def save_figure(fig: plt.Figure, output_path: Path | str, dpi: int = 300) -> Path:
    output = Path(output_path)
    ensure_directory(output.parent)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


def plot_text_length_distribution(lengths: pd.Series, output_path: Path | str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=40, kde=True, color="#264653", ax=ax)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Text Length (characters)")
    ax.set_ylabel("Count")
    return save_figure(fig, output_path)


def plot_top_words(word_counts: Counter, output_path: Path | str, title: str, top_n: int = 20) -> Path:
    top_words = word_counts.most_common(top_n)
    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts, y=words, ax=ax, palette="crest")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    return save_figure(fig, output_path)


def plot_wordcloud(text: str, output_path: Path | str, title: str) -> Path:
    text_to_use = text.strip() or "empty"
    wordcloud = WordCloud(width=1600, height=800, background_color="white", colormap="viridis").generate(text_to_use)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    return save_figure(fig, output_path)


def plot_sentiment_distribution(sentiments: pd.Series, output_path: Path | str, title: str) -> Path:
    counts = sentiments.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    return save_figure(fig, output_path)


def write_json(data: dict, output_path: Path | str) -> Path:
    output = Path(output_path)
    ensure_directory(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    return output
