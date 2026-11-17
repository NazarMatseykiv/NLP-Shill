from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text).strip().lower()))


def build() -> Pipeline:
    vec = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=60_000, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 max_features=30_000, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w\w+\b")),
    ])
    return Pipeline([("tfidf", vec),
                     ("clf", LinearSVC(class_weight="balanced", max_iter=20000, random_state=SEED))])


GRID = {
    "clf__C": [0.25, 0.5, 1.0, 2.0, 4.0],
    "clf__loss": ["hinge", "squared_hinge"],
    "tfidf__char__ngram_range": [(2, 4), (2, 5), (1, 5)],
    "tfidf__char__min_df": [1, 2],
    "tfidf__word__ngram_range": [(1, 1), (1, 2)],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/intents_merged.csv")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = [normalize(t) for t in df["text"].astype(str)]
    labels = df["intent"].astype(str).tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

    search = GridSearchCV(
        build(), GRID, scoring="f1_macro", n_jobs=-1, refit=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
    search.fit(X_tr, y_tr)

    y_pred = search.best_estimator_.predict(X_te)
    holdout = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
    }

    results = pd.DataFrame(search.cv_results_)
    cols = [c for c in results.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "rank_test_score"]
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    results[cols].sort_values("rank_test_score").to_csv(
        models_dir / "grid_search_results.csv", index=False)

    out = {
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(X_tr), "n_test": len(X_te),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_f1_macro_on_train": float(search.best_score_),
        "holdout_metrics": holdout,
    }
    (models_dir / "tuning_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    final = build().set_params(**search.best_params_).fit(texts, labels)
    joblib.dump(final, models_dir / "intent_model_deploy.joblib")

    print(f"Best CV F1 (train only): {search.best_score_:.4f}")
    print(f"Holdout F1 macro:        {holdout['f1_macro']:.4f}")
    print(f"Best params: {search.best_params_}")


if __name__ == "__main__":
    main()
