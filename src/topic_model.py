from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import run_preprocessing  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    infer_text_column,
    load_data,
    save_csv,
)


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_clean_data(input_path: str | None = None) -> pd.DataFrame:
    if input_path is not None:
        df, source_path = load_data(input_path=input_path)
        print(f"[topic] Loaded cleaned data from: {source_path}")
        return df

    default_path = DEFAULT_PROCESSED_DIR / "cleaned.csv"
    if default_path.exists():
        df, source_path = load_data(input_path=default_path)
        print(f"[topic] Loaded cleaned data from: {source_path}")
        return df

    print("[topic] Cleaned data not found. Running preprocessing fallback...")
    output_path = run_preprocessing(None)
    df, source_path = load_data(input_path=output_path)
    print(f"[topic] Loaded preprocessed data from: {source_path}")
    return df


def build_topic_texts(df: pd.DataFrame) -> list[str]:
    if "cleaned_text" in df.columns:
        texts = df["cleaned_text"].fillna("").astype(str).tolist()
        fallback_texts = df.get("text", df.iloc[:, 0]).fillna("").astype(str).tolist()
        return [cleaned if cleaned.strip() else raw for cleaned, raw in zip(texts, fallback_texts)]

    text_column = infer_text_column(df)
    return df[text_column].fillna("").astype(str).tolist()


def load_or_train_topic_model(docs: list[str], checkpoint_dir: Path, embedding_model_name: str) -> BERTopic:
    embedding_model = SentenceTransformer(embedding_model_name)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    checkpoint_exists = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    if checkpoint_exists:
        print(f"[topic] Loading BERTopic model from checkpoint: {checkpoint_dir}")
        return BERTopic.load(str(checkpoint_dir), embedding_model=embedding_model)

    print("[topic] No checkpoint found. Training a new BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
        verbose=True,
        min_topic_size=10,
    )
    topics, _ = topic_model.fit_transform(docs)
    print(f"[topic] Trained BERTopic model on {len(docs)} documents.")
    topic_model.save(str(checkpoint_dir), serialization="safetensors", save_embedding_model=True)
    print(f"[topic] Saved BERTopic checkpoint to: {checkpoint_dir}")
    print(f"[topic] Unique topics found: {len(set(topics))}")
    return topic_model


def save_topic_outputs(topic_model: BERTopic, docs: list[str], output_dir: Path) -> tuple[Path, Path]:
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / "topic_info.csv"
    save_csv(topic_info, topic_info_path)

    topics, _ = topic_model.transform(docs)
    assignments = pd.DataFrame({"document_id": range(len(docs)), "text": docs, "topic": topics})
    assignments_path = output_dir / "topic_assignments.csv"
    save_csv(assignments, assignments_path)

    html_path = output_dir / "topic_visualization.html"
    try:
        fig = topic_model.visualize_topics()
    except Exception:
        fig = topic_model.visualize_barchart(top_n_topics=10)
    fig.write_html(str(html_path))

    return topic_info_path, html_path


def run_topic_model(input_path: str | None = None, checkpoint_dir: str | None = None, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    print("[topic] Starting topic modeling stage...")
    df = load_clean_data(input_path)
    print(f"[topic] Data shape: {df.shape}")

    docs = build_topic_texts(df)
    docs = [doc if doc.strip() else "empty document" for doc in docs]
    print(f"[topic] Documents prepared: {len(docs)}")

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else Path("checkpoints/bertopic")
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    topic_model = load_or_train_topic_model(docs, checkpoint_path, embedding_model_name)

    output_dir = DEFAULT_PROCESSED_DIR
    topic_info_path, html_path = save_topic_outputs(topic_model, docs, output_dir)

    topic_info = topic_model.get_topic_info()
    print("[topic] Top topics:")
    print(topic_info.head(10).to_string(index=False))
    print(f"[topic] Saved topic info CSV to: {topic_info_path}")
    print(f"[topic] Saved topic visualization HTML to: {html_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or load BERTopic for topic modeling.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to cleaned CSV/JSON. Defaults to outputs/processed/cleaned.csv.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path("checkpoints/bertopic")),
        help="Directory used to load/save the BERTopic checkpoint.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used by BERTopic.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_topic_model(args.input, args.checkpoint_dir, args.embedding_model)


if __name__ == "__main__":
    main()
