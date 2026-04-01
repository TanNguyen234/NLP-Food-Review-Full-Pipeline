from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from src.utils import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_PROCESSED_DIR,
    clean_text,
    ensure_directory,
    infer_text_column,
    preprocess_text_column,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BER_TOPIC_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bertopic"
ROBERTA_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "roberta"


st.set_page_config(page_title="NLP Pipeline", layout="wide")


def parse_raw_text(raw_text: str) -> pd.DataFrame:
    documents = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return pd.DataFrame({"text": documents})


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def build_input_dataframe(raw_text: str, uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    raw_text = raw_text.strip()

    if raw_text:
        return parse_raw_text(raw_text), "raw_text"

    if uploaded_file is not None:
        try:
            return load_csv_from_upload(uploaded_file), "csv"
        except Exception as error:
            st.error(f"Failed to read uploaded CSV: {error}")
            return None, None

    return None, None


def find_text_series(df: pd.DataFrame) -> pd.Series:
    if "text" in df.columns:
        return df["text"].fillna("").astype(str)

    text_column = infer_text_column(df)
    return df[text_column].fillna("").astype(str)


def render_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("1. Preprocessing")
    st.write(f"Original shape: {df.shape}")
    st.dataframe(df.head(5), use_container_width=True)

    text_column = infer_text_column(df)
    st.caption(f"Using text column: {text_column}")

    processed = preprocess_text_column(df, text_column=text_column)
    st.write(f"Processed shape: {processed.shape}")
    st.dataframe(processed[[col for col in ["text", "cleaned_text", "token_count"] if col in processed.columns]].head(5), use_container_width=True)

    output_path = DEFAULT_PROCESSED_DIR / "cleaned.csv"
    ensure_directory(output_path.parent)
    processed.to_csv(output_path, index=False)
    st.success(f"Saved cleaned data to {output_path}")
    return processed


def make_histogram(lengths: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(lengths, bins=30, kde=True, color="#2a9d8f", ax=ax)
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Text Length")
    ax.set_ylabel("Count")
    return fig


def make_top_words_chart(word_counts: Counter):
    top_words = word_counts.most_common(20)
    top_df = pd.DataFrame(top_words, columns=["word", "count"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_df, y="word", x="count", ax=ax, palette="crest")
    ax.set_title("Top 20 Most Common Words")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    return fig


def make_wordcloud_image(text: str):
    wc = WordCloud(width=1600, height=800, background_color="white", colormap="viridis").generate(text or "empty")
    return wc.to_array()


def render_eda(processed: pd.DataFrame) -> None:
    st.subheader("2. EDA")

    text_series = processed["cleaned_text"].fillna("").astype(str) if "cleaned_text" in processed.columns else find_text_series(processed)
    lengths = text_series.str.len()
    word_counts = Counter(word for text in text_series for word in text.split() if word)

    st.write("Text length statistics")
    st.write(
        {
            "documents": int(len(processed)),
            "average_length": float(lengths.mean()) if len(lengths) else 0.0,
            "median_length": float(lengths.median()) if len(lengths) else 0.0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
        }
    )

    histogram_fig = make_histogram(lengths)
    st.pyplot(histogram_fig, use_container_width=True)
    plt.close(histogram_fig)

    top_words_fig = make_top_words_chart(word_counts)
    st.pyplot(top_words_fig, use_container_width=True)
    plt.close(top_words_fig)

    wordcloud_image = make_wordcloud_image(" ".join(text_series.tolist()))
    st.image(wordcloud_image, caption="Word Cloud", use_container_width=True)

    st.write("Top 10 words")
    st.dataframe(pd.DataFrame(word_counts.most_common(10), columns=["word", "count"]), use_container_width=True)


def _topic_checkpoint_available() -> bool:
    return BER_TOPIC_CHECKPOINT.exists() and any(BER_TOPIC_CHECKPOINT.iterdir())


def render_topic_modeling(processed: pd.DataFrame) -> None:
    st.subheader("3. Topic Modeling")

    if not _topic_checkpoint_available():
        st.warning("BERTopic checkpoint not found in checkpoints/bertopic. Topic modeling is skipped.")
        return

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        docs = (
            processed["cleaned_text"].fillna("").astype(str).tolist()
            if "cleaned_text" in processed.columns
            else find_text_series(processed).tolist()
        )
        docs = [doc if doc.strip() else "empty document" for doc in docs]

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic.load(str(BER_TOPIC_CHECKPOINT), embedding_model=embedding_model)
        topic_info = topic_model.get_topic_info()
        topics, _ = topic_model.transform(docs)

        st.write("Top topics")
        st.dataframe(topic_info.head(10), use_container_width=True)
        st.write("Topic assignments")
        st.dataframe(pd.DataFrame({"text": docs, "topic": topics}).head(20), use_container_width=True)
    except Exception as error:
        st.warning(f"Topic model could not be loaded: {error}")


def _roberta_checkpoint_available() -> bool:
    return ROBERTA_CHECKPOINT.exists() and any(ROBERTA_CHECKPOINT.iterdir())


def normalize_sentiment_label(label: str, id2label: dict | None = None) -> str:
    label_text = str(label).strip().lower()
    if label_text in {"negative", "neutral", "positive"}:
        return label_text

    if label_text.startswith("label_") and id2label:
        try:
            index = int(label_text.split("_", 1)[1])
            mapped = str(id2label.get(index, label_text)).strip().lower()
            if mapped in {"negative", "neutral", "positive"}:
                return mapped
        except ValueError:
            pass

    if "neg" in label_text:
        return "negative"
    if "neu" in label_text:
        return "neutral"
    if "pos" in label_text:
        return "positive"
    return label_text


def render_sentiment(processed: pd.DataFrame) -> None:
    st.subheader("4. Sentiment Analysis")

    if not _roberta_checkpoint_available():
        st.warning("RoBERTa checkpoint not found in checkpoints/roberta. Sentiment analysis is skipped.")
        return

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        texts = (
            processed["cleaned_text"].fillna("").astype(str).tolist()
            if "cleaned_text" in processed.columns
            else find_text_series(processed).tolist()
        )

        tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_CHECKPOINT))
        model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_CHECKPOINT))
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )

        outputs = sentiment_pipe(texts)
        if isinstance(outputs, dict):
            outputs = [outputs]

        sentiment_df = pd.DataFrame(
            {
                "text": texts,
                "sentiment_label": [normalize_sentiment_label(item["label"], getattr(model.config, "id2label", {})) for item in outputs],
                "sentiment_score": [float(item["score"]) for item in outputs],
            }
        )

        st.write("Sentiment results")
        st.dataframe(sentiment_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        order = ["negative", "neutral", "positive"]
        counts = sentiment_df["sentiment_label"].value_counts().reindex(order, fill_value=0)
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Set2")
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as error:
        st.warning(f"Sentiment model could not be loaded: {error}")


def show_input_summary(df: pd.DataFrame) -> None:
    st.subheader("Input Summary")
    st.metric("Number of documents", len(df))
    st.dataframe(df.head(10), use_container_width=True)


def main() -> None:
    st.title("NLP Pipeline Explorer")
    st.caption("Raw text takes priority over CSV upload. Each non-empty line in the text box is treated as one document.")

    with st.form("input_form"):
        raw_text = st.text_area("Raw Text Input", height=220, placeholder="hello world\nthis is test")
        uploaded_file = st.file_uploader("CSV Upload", type=["csv"])
        submitted = st.form_submit_button("Run Analysis")

    if not submitted:
        st.info("Provide raw text or upload a CSV, then click Run Analysis.")
        return

    input_df, source_mode = build_input_dataframe(raw_text, uploaded_file)

    if input_df is None or input_df.empty:
        st.warning("Empty input. Provide raw text or upload a CSV file.")
        return

    if source_mode == "raw_text" and uploaded_file is not None and raw_text.strip():
        st.info("Raw text input was provided and will be used instead of the uploaded CSV.")

    st.header("Input Data")
    show_input_summary(input_df)

    try:
        processed = render_preprocessing(input_df)
    except Exception as error:
        st.error(f"Preprocessing failed: {error}")
        return

    st.header("Analysis")
    render_eda(processed)
    render_topic_modeling(processed)
    render_sentiment(processed)


if __name__ == "__main__":
    main()
