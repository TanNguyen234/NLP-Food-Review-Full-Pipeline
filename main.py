from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stage of the NLP food review pipeline.")
    parser.add_argument(
        "--step",
        required=True,
        choices=["preprocessing", "eda", "topic", "sentiment"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--input", type=str, default=None, help="Optional input file or directory path.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional checkpoint directory override.")
    parser.add_argument("--embedding-model", type=str, default=None, help="Optional BERTopic embedding model override.")
    parser.add_argument("--pretrained-model", type=str, default=None, help="Optional RoBERTa model override.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size for sentiment analysis.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.step == "preprocessing":
        from src.preprocessing import run_preprocessing

        run_preprocessing(args.input)
    elif args.step == "eda":
        from src.eda import run_eda

        run_eda(args.input)
    elif args.step == "topic":
        from src.topic_model import run_topic_model

        run_topic_model(args.input, args.checkpoint_dir, args.embedding_model or "all-MiniLM-L6-v2")
    elif args.step == "sentiment":
        from src.sentiment import run_sentiment

        run_sentiment(args.input, args.checkpoint_dir, args.pretrained_model or "cardiffnlp/twitter-roberta-base-sentiment-latest", args.batch_size)


if __name__ == "__main__":
    main()
