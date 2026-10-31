"""Train the NLP triage model on the opensec-triage dataset.

Data source (first one found):
  1. data/train.jsonl (+ data/test.jsonl)            -> local files
  2. Hugging Face: tegridydev/opensec-triage          -> needs `pip install datasets`
  3. data/test.jsonl only                             -> 80/20 split (quick demo)

Classes: malicious, suspicious, benign, expected_admin, authorised_testing,
         misconfiguration, unknown

Run:  python train.py
Out:  models/triage_model.joblib, models/metrics.json
"""
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

os.makedirs("models", exist_ok=True)


def load_data():
    if os.path.exists("data/train.jsonl"):
        train = pd.read_json("data/train.jsonl", lines=True)
        test = pd.read_json("data/test.jsonl", lines=True) if os.path.exists("data/test.jsonl") else None
        if test is None:
            train, test = train_test_split(train, test_size=0.2, random_state=42, stratify=train["target"])
        return train, test, "local train/test files"
    try:
        from datasets import load_dataset

        ds = load_dataset("tegridydev/opensec-triage")
        return ds["train"].to_pandas(), ds["test"].to_pandas(), "Hugging Face (full dataset)"
    except Exception as e:  # no internet / datasets not installed
        print("Could not load from Hugging Face:", type(e).__name__)
    df = pd.read_json("data/test.jsonl", lines=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    return train, test, "data/test.jsonl split 80/20 (demo)"


train, test, source = load_data()
print(f"Data source: {source} | train={len(train)} test={len(test)}")

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=300_000)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ]
)
model.fit(train["text"], train["target"])
pred = model.predict(test["text"])

labels = sorted(train["target"].unique())
metrics = {
    "data_source": source,
    "n_train": len(train),
    "n_test": len(test),
    "accuracy": round(accuracy_score(test["target"], pred), 4),
    "f1_macro": round(f1_score(test["target"], pred, average="macro"), 4),
    "labels": labels,
    "confusion_matrix": confusion_matrix(test["target"], pred, labels=labels).tolist(),
    "report": classification_report(test["target"], pred, output_dict=True),
}
print(classification_report(test["target"], pred))
joblib.dump(model, "models/triage_model.joblib")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved models/triage_model.joblib and models/metrics.json")
