"""
train.py
Trains a language detection model on the Kaggle "Language Detection" dataset.

Dataset: https://www.kaggle.com/datasets/basilb2s/language-detection
Download "Language Detection.csv" and place it in the data/ folder before running.

Usage:
    python train.py
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/Language Detection.csv"
MODEL_PATH = "model.joblib"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Kaggle dataset columns are "Text" and "Language"
    df = df.rename(columns={c: c.strip() for c in df.columns})
    assert "Text" in df.columns and "Language" in df.columns, (
        f"Expected columns 'Text' and 'Language', got: {list(df.columns)}"
    )
    df = df.dropna(subset=["Text", "Language"])
    return df


def build_pipeline() -> Pipeline:
    # Character n-grams (1-4) capture language-specific letter patterns
    # better than word-level features, and work even on short snippets.
    # Wider range (up to 4 chars) helps separate closely related languages
    # like English/Dutch/German/Danish that share a lot of short n-grams.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
    )
    # LinearSVC draws much sharper decision boundaries than Naive Bayes for
    # this kind of task. class_weight="balanced" stops it from favoring the
    # largest class (English) just because it has more training examples.
    # Wrapped in CalibratedClassifierCV so we still get predict_proba()
    # (LinearSVC alone only gives decision scores, not probabilities) for
    # the confidence bar / top-5 chart in the UI.
    base_clf = LinearSVC(class_weight="balanced", max_iter=10000)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows across {df['Language'].nunique()} languages")
    print(df["Language"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["Text"], df["Language"], test_size=0.2, random_state=42, stratify=df["Language"]
    )

    print("\nTraining model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
