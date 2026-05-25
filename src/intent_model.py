from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import normalize_text

INTENT_LABELS = ("create_order", "track_order", "cancel_order", "faq")


def load_intent_data(csv_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    texts = [normalize_text(t) for t in df["text"].astype(str)]
    labels = df["intent"].astype(str).tolist()
    return texts, labels


def build_intent_pipeline(
    model_name: str = "logistic_regression",
    feature_set: str = "combined",
) -> Pipeline:
    # local import to break circular dependency
    from .evaluation import build_best_pipeline

    return build_best_pipeline(model_name, feature_set)


def train_intent_model(
    csv_path: Path,
    model_path: Path,
    model_name: str = "logistic_regression",
    feature_set: str = "combined",
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    # local import to break circular dependency
    from .evaluation import evaluate_cross_validation, evaluate_holdout

    pipe = build_intent_pipeline(model_name, feature_set)
    holdout = evaluate_holdout(pipe, csv_path, test_size=test_size)
    cross_val = evaluate_cross_validation(pipe, csv_path, cv_folds=cv_folds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    texts, labels = load_intent_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, model_path)

    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "train_size": holdout["train_size"],
        "test_size": holdout["test_size"],
        "train_accuracy": holdout["train_accuracy"],
        "train_f1_macro": holdout["train_f1_macro"],
        "cv_folds": cross_val["cv_folds"],
        "cv_accuracy_mean": cross_val["cv_accuracy_mean"],
        "cv_accuracy_std": cross_val["cv_accuracy_std"],
        "cv_f1_macro_mean": cross_val["cv_f1_macro_mean"],
        "cv_f1_macro_std": cross_val["cv_f1_macro_std"],
        "cv_precision_macro_mean": cross_val["cv_precision_macro_mean"],
        "cv_precision_macro_std": cross_val["cv_precision_macro_std"],
        "cv_recall_macro_mean": cross_val["cv_recall_macro_mean"],
        "cv_recall_macro_std": cross_val["cv_recall_macro_std"],
        "report": holdout["classification_report"],
        "report_text": holdout["classification_report_text"],
        "confusion_matrix": holdout["confusion_matrix"],
        "labels": holdout["labels"],
        "accuracy": holdout["accuracy"],
        "f1_macro": holdout["f1_macro"],
    }


def predict_intent(pipe: Pipeline, text: str) -> tuple[str, float]:
    text_n = normalize_text(text)
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba([text_n])[0]
        idx = int(proba.argmax())
        return pipe.classes_[idx], float(proba[idx])

    if hasattr(clf, "decision_function"):
        scores = pipe.decision_function([text_n])[0]
        if scores.ndim == 1:
            # binary
            idx = int(scores.argmax())
        else:
            idx = int(np.argmax(scores))
        # pseudo-probability via softmax
        exp = np.exp(scores - np.max(scores))
        prob = exp / exp.sum()
        return pipe.classes_[idx], float(prob[idx])

    label = pipe.predict([text_n])[0]
    return label, 1.0
