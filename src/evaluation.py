from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from .model_registry import get_model_candidates


def resolve_data_path(root: Path) -> Path:
    merged = root / "data" / "intents_merged.csv"
    if merged.exists():
        return merged
    return root / "data" / "intents_train.csv"


def compare_models(
    csv_path: Path,
    feature_set: str = "combined",
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    from .intent_model import load_intent_dataset

    texts, labels, groups = load_intent_dataset(csv_path)
    splits = list(
        StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        ).split(texts, labels, groups)
    )

    train_idx, test_idx = splits[0]
    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    groups_train = [groups[i] for i in train_idx]
    cv = StratifiedGroupKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    results = []
    candidates = get_model_candidates(feature_set)

    for name, pipeline in candidates.items():
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            groups=groups_train,
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_time = time.perf_counter() - t0

        results.append(
            {
                "model": name,
                "feature_set": feature_set,
                "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
                "cv_accuracy_std": float(cv_scores["test_accuracy"].std()),
                "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(cv_scores["test_f1_macro"].std()),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "test_precision_macro": float(
                    precision_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "test_recall_macro": float(
                    recall_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "train_time_sec": round(train_time, 4),
                "predict_time_sec": round(predict_time, 6),
            }
        )

    results.sort(key=lambda x: x["cv_f1_macro_mean"], reverse=True)
    return {
        "data_path": str(csv_path),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_folds": cv_folds,
        "feature_set": feature_set,
        "ranking": results,
        "best_model": results[0]["model"] if results else None,
    }


def evaluate_cross_validation(pipeline, csv_path: Path, cv_folds: int = 5, random_state: int = 42) -> dict:
    # local import to avoid circular dependency
    from .intent_model import load_intent_dataset

    texts, labels, groups = load_intent_dataset(csv_path)
    cv = StratifiedGroupKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    cv_scores = cross_validate(
        pipeline,
        texts,
        labels,
        cv=cv,
        scoring=scoring,
        groups=groups,
        n_jobs=-1,
    )

    return {
        "cv_folds": cv_folds,
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_scores["test_accuracy"].std()),
        "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
        "cv_f1_macro_std": float(cv_scores["test_f1_macro"].std()),
        "cv_precision_macro_mean": float(cv_scores["test_precision_macro"].mean()),
        "cv_precision_macro_std": float(cv_scores["test_precision_macro"].std()),
        "cv_recall_macro_mean": float(cv_scores["test_recall_macro"].mean()),
        "cv_recall_macro_std": float(cv_scores["test_recall_macro"].std()),
    }


def evaluate_holdout(pipeline, csv_path: Path, test_size: float = 0.2) -> dict:
    # local import to avoid circular dependency
    from .intent_model import load_intent_dataset

    texts, labels, groups = load_intent_dataset(csv_path)
    n_splits = max(2, round(1 / test_size))
    train_idx, test_idx = next(
        StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        ).split(texts, labels, groups)
    )
    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "labels": labels_sorted,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "classification_report_text": classification_report(y_test, y_pred, zero_division=0),
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "train_f1_macro": float(
            f1_score(y_train, y_train_pred, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def build_best_pipeline(model_name: str, feature_set: str = "combined"):
    candidates = get_model_candidates(feature_set)
    if model_name not in candidates:
        raise ValueError(f"Unknown model: {model_name}")
    return candidates[model_name]


def save_model_card(path: Path, card: dict) -> None:
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
