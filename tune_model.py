import json
import hashlib
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.evaluation import build_best_pipeline, resolve_data_path
from src.intent_model import load_intent_dataset

ROOT = Path(__file__).resolve().parent


def load_best_config() -> tuple[str, str]:
    path = ROOT / "models" / "best_config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cfg["model_name"], cfg["feature_set"]
    return "logistic_regression", "combined"


def param_grid_for(model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {"clf__C": [0.25, 0.5, 1.0, 2.0, 4.0]}
    if model_name == "linear_svc":
        return {"clf__C": [0.25, 0.5, 1.0, 2.0]}
    if model_name == "complement_nb":
        return {"clf__alpha": [0.01, 0.05, 0.1, 0.3]}
    if model_name == "random_forest":
        return {
            "clf__n_estimators": [150, 250],
            "clf__max_depth": [16, 24, None],
        }
    return {}


def main() -> None:
    model_name, feature_set = load_best_config()
    if model_name not in ("logistic_regression", "linear_svc", "complement_nb", "random_forest"):
        model_name = "logistic_regression"

    data_path = resolve_data_path(ROOT)
    texts, labels, groups = load_intent_dataset(data_path)
    pipe = build_best_pipeline(model_name, feature_set)
    grid = param_grid_for(model_name)

    if not grid:
        print("Немає сітки гіперпараметрів — пропуск tune.")
        return

    print(f"GridSearch: {model_name} ({feature_set}), params={grid}")
    search = GridSearchCV(
        pipe,
        grid,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    search.fit(texts, labels, groups=groups)

    out = {
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "feature_set": feature_set,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
    }
    tuned_path = ROOT / "models" / "tuning_results.json"
    tuned_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best CV F1 macro: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    print("Тренувальну модель не змінено; застосуйте параметри через python train.py")
    print(f"Tuning log: {tuned_path}")


if __name__ == "__main__":
    main()
