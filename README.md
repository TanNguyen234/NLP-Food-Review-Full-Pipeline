# NLP Food Review Full Pipeline

Production-style Python implementation of a four-stage NLP workflow for food reviews:

1. Text preprocessing
2. Exploratory data analysis
3. Topic modeling with BERTopic
4. Sentiment analysis with RoBERTa

## Project Layout

```
project_root/
├── data/
│   └── raw/
├── checkpoints/
│   ├── bertopic/
│   └── roberta/
├── outputs/
│   ├── figures/
│   └── processed/
├── src/
├── requirements.txt
├── README.md
└── main.py
```

## Setup

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Put your raw dataset in `data/raw/` as a CSV or JSON file. The scripts expect a text-like column and will look for a column named `text`, `review`, `content`, or a similar variant. If your file uses a different column name, it will try to infer the best match.

## Checkpoints

If you already have trained artifacts, place them here:

- `checkpoints/bertopic/` for BERTopic model files
- `checkpoints/roberta/` for a local RoBERTa sentiment model

If those folders are empty, the scripts will train or download models as needed.

## Run Each Stage

### 1. Preprocessing

```bash
python main.py --step preprocessing
```

Outputs:
- `outputs/processed/cleaned.csv`

### 2. EDA

```bash
python main.py --step eda
```

Outputs:
- `outputs/figures/text_length_distribution.png`
- `outputs/figures/top_words.png`
- `outputs/figures/wordcloud.png`

### 3. Topic Modeling

```bash
python main.py --step topic
```

Outputs:
- `outputs/processed/topic_info.csv`
- `outputs/processed/topic_assignments.csv`
- `outputs/processed/topic_visualization.html`

If `checkpoints/bertopic/` is empty, a new model will be trained and saved there.

### 4. Sentiment Analysis

```bash
python main.py --step sentiment
```

Outputs:
- `outputs/processed/sentiment_results.csv`
- `outputs/figures/sentiment_distribution.png`

If `checkpoints/roberta/` is empty, the pretrained RoBERTa model will be downloaded automatically.

## Direct Script Execution

Each stage can also be run directly:

```bash
python src/preprocessing.py
python src/eda.py
python src/topic_model.py
python src/sentiment.py
```

## CLI Options

All stages support `--input` to point to a specific CSV or JSON file. Topic modeling also supports `--checkpoint-dir` and `--embedding-model`. Sentiment analysis supports `--checkpoint-dir`, `--pretrained-model`, and `--batch-size`.
