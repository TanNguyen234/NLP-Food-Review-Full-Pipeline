from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import run_preprocessing  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_FIGURES_DIR,
    DEFAULT_PROCESSED_DIR,
    infer_text_column,
    load_data,
    plot_sentiment_distribution,
    save_csv,
)


DEFAULT_PRETRAINED_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def load_clean_data(input_path: str | None = None) -> pd.DataFrame:
    if input_path is not None:
        df, source_path = load_data(input_path=input_path)
        print(f"[sentiment] Loaded cleaned data from: {source_path}")
        return df

    default_path = DEFAULT_PROCESSED_DIR / "cleaned.csv"
    if default_path.exists():
        df, source_path = load_data(input_path=default_path)
        print(f"[sentiment] Loaded cleaned data from: {source_path}")
        return df

    print("[sentiment] Cleaned data not found. Running preprocessing fallback...")
    output_path = run_preprocessing(None)
    df, source_path = load_data(input_path=output_path)
    print(f"[sentiment] Loaded preprocessed data from: {source_path}")
    return df


def resolve_text_series(df: pd.DataFrame) -> pd.Series:
    if "cleaned_text" in df.columns:
        texts = df["cleaned_text"].fillna("").astype(str)
        if "text" in df.columns:
            raw_texts = df["text"].fillna("").astype(str)
            return texts.where(texts.str.strip().astype(bool), raw_texts)
        return texts

    text_column = infer_text_column(df)
    return df[text_column].fillna("").astype(str)


def load_sentiment_pipeline(checkpoint_dir: Path, pretrained_model: str):
    checkpoint_exists = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    model_path = checkpoint_dir if checkpoint_exists else pretrained_model

    if checkpoint_exists:
        print(f"[sentiment] Loading RoBERTa model from checkpoint: {checkpoint_dir}")
    else:
        print(f"[sentiment] No checkpoint found. Downloading pretrained model: {pretrained_model}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    import torch

    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )
    return sentiment_pipeline, model


def normalize_model_label(label: str, model) -> str:
    normalized = str(label).strip().lower()
    if normalized in SENTIMENT_ORDER:
        return normalized

    if normalized.startswith("label_"):
        try:
            index = int(normalized.split("_", 1)[1])
            mapped = model.config.id2label.get(index, normalized)
            mapped = str(mapped).strip().lower()
            if mapped in SENTIMENT_ORDER:
                return mapped
        except (ValueError, AttributeError):
            pass

    if normalized.isdigit():
        index = int(normalized)
        mapped = model.config.id2label.get(index, normalized)
        mapped = str(mapped).strip().lower()
        if mapped in SENTIMENT_ORDER:
            return mapped

    if "pos" in normalized:
        return "positive"
    if "neg" in normalized:
        return "negative"
    if "neu" in normalized:
        return "neutral"

    id2label = getattr(model.config, "id2label", {})
    if len(id2label) == 3:
        candidate = str(id2label.get(0, "negative")).strip().lower()
        if candidate in SENTIMENT_ORDER:
            return candidate

    return normalized


def predict_sentiments(texts: pd.Series, sentiment_pipeline, model, batch_size: int = 16) -> pd.DataFrame:
    normalized_texts = texts.fillna("").astype(str).tolist()
    sentiment_labels: list[str] = []
    sentiment_scores: list[float] = []

    total_batches = max(1, (len(normalized_texts) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, len(normalized_texts), batch_size), 1):
        batch_texts = normalized_texts[start : start + batch_size]
        print(f"[sentiment] Processing batch {batch_index}/{total_batches} ({len(batch_texts)} texts)...")
        batch_outputs = sentiment_pipeline(batch_texts)
        if isinstance(batch_outputs, dict):
            batch_outputs = [batch_outputs]

        for output in batch_outputs:
            label = normalize_model_label(output["label"], model)
            sentiment_labels.append(label)
            sentiment_scores.append(float(output["score"]))

    return pd.DataFrame({"sentiment_label": sentiment_labels, "sentiment_score": sentiment_scores})


def run_sentiment(input_path: str | None = None, checkpoint_dir: str | None = None, pretrained_model: str = DEFAULT_PRETRAINED_MODEL, batch_size: int = 16) -> None:
    print("[sentiment] Starting sentiment analysis stage...")
    df = load_clean_data(input_path)
    print(f"[sentiment] Data shape: {df.shape}")

    texts = resolve_text_series(df)

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else Path("checkpoints/roberta")
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    sentiment_pipeline, model = load_sentiment_pipeline(checkpoint_path, pretrained_model)
    predictions = predict_sentiments(texts, sentiment_pipeline, model, batch_size=batch_size)

    results = pd.concat([df.reset_index(drop=True), predictions], axis=1)
    output_path = DEFAULT_PROCESSED_DIR / "sentiment_results.csv"
    save_csv(results, output_path)
    print(f"[sentiment] Saved sentiment results to: {output_path}")

    figure_path = DEFAULT_FIGURES_DIR / "sentiment_distribution.png"
    plot_sentiment_distribution(results["sentiment_label"], figure_path, "Sentiment Distribution")
    print(f"[sentiment] Saved sentiment distribution chart to: {figure_path}")

    print("[sentiment] Sentiment label counts:")
    print(results["sentiment_label"].value_counts().to_string())
    print("[sentiment] Sample predictions:")
    preview_columns = [col for col in ["text", "cleaned_text", "sentiment_label", "sentiment_score"] if col in results.columns]
    print(results[preview_columns].head(5).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoBERTa sentiment analysis on cleaned text.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to cleaned CSV/JSON. Defaults to outputs/processed/cleaned.csv.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path("checkpoints/roberta")),
        help="Directory used to load/save a local RoBERTa checkpoint.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=DEFAULT_PRETRAINED_MODEL,
        help="Pretrained RoBERTa sentiment model to download when no local checkpoint exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for inference.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_sentiment(args.input, args.checkpoint_dir, args.pretrained_model, args.batch_size)


if __name__ == "__main__":
    main()
