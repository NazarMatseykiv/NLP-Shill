from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .menu import MenuMatcher
from .preprocess import normalize_text

INTENT_LABELS = ("create_order", "track_order", "cancel_order", "faq", "other")


def load_intent_dataset(
    csv_path: Path, menu: MenuMatcher | None = None
) -> tuple[list[str], list[str], list[str]]:
    df = pd.read_csv(csv_path)
    if menu is None:
        menu_path = csv_path.parent / "menu.json"
        if not menu_path.exists():
            menu_path = Path(__file__).resolve().parents[1] / "data" / "menu.json"
        menu = MenuMatcher(menu_path)
    texts = [normalize_text(t, menu) for t in df["text"].astype(str)]
    labels = df["intent"].astype(str).tolist()
    groups = (
        df["group"].astype(str).tolist()
        if "group" in df.columns
        else [f"{label}:{text}" for label, text in zip(labels, texts)]
    )
    return texts, labels, groups


def load_intent_data(csv_path: Path) -> tuple[list[str], list[str]]:
    texts, labels, _ = load_intent_dataset(csv_path)
    return texts, labels


def make_group_splits(
    texts: list[str],
    labels: list[str],
    groups: list[str],
    cv_folds: int,
    random_state: int = 42,
) -> list[tuple]:
    cv = StratifiedGroupKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    return list(cv.split(texts, labels, groups))


def calibrate_pipeline(
    pipeline,
    texts: list[str],
    labels: list[str],
    groups: list[str],
    cv_folds: int = 5,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=make_group_splits(texts, labels, groups, cv_folds),
        ensemble=False,
    )
    return calibrated.fit(texts, labels)


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
    model_params: dict | None = None,
) -> dict:
    # local import to break circular dependency
    from .evaluation import evaluate_cross_validation, evaluate_holdout

    pipe = build_intent_pipeline(model_name, feature_set)
    if model_params:
        pipe.set_params(**model_params)
    holdout = evaluate_holdout(pipe, csv_path, test_size=test_size)
    cross_val = evaluate_cross_validation(pipe, csv_path, cv_folds=cv_folds)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    texts, labels, groups = load_intent_dataset(csv_path)
    calibrated = calibrate_pipeline(pipe, texts, labels, groups, cv_folds)
    joblib.dump(calibrated, model_path)

    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "train_size": len(texts),
        "evaluation_train_size": holdout["train_size"],
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


def predict_intent(
    pipe: Pipeline, text: str, menu: MenuMatcher | None = None
) -> tuple[str, float]:
    text_n = normalize_text(text, menu)
    if not hasattr(pipe, "predict_proba"):
        raise ValueError("Модель не має каліброваних ймовірностей. Запустіть train.py.")
    proba = pipe.predict_proba([text_n])[0]
    idx = int(proba.argmax())
    return pipe.classes_[idx], float(proba[idx])
