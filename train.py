import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

from src.evaluation import resolve_data_path
from src.intent_model import train_intent_model

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "intent_model.joblib"
METRICS = ROOT / "models" / "metrics.json"
CONFIG = ROOT / "models" / "best_config.json"


def load_config() -> tuple[str, str]:
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        return cfg.get("model_name", "logistic_regression"), cfg.get(
            "feature_set", "combined"
        )
    return "logistic_regression", "combined"


def main() -> None:
    data_path = resolve_data_path(ROOT)
    if not data_path.exists():
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_merged_local.py")],
            check=False,
        )
        print("Підказка: python scripts/download_datasets.py")

    model_name, feature_set = load_config()
    tuning_path = ROOT / "models" / "tuning_results.json"
    model_params = {}
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        if (
            tuning.get("model_name") == model_name
            and tuning.get("feature_set") == feature_set
            and tuning.get("data_sha256")
            == hashlib.sha256(data_path.read_bytes()).hexdigest()
        ):
            model_params = tuning.get("best_params", {})
    print(f"Навчання: {model_name} + {feature_set} ({data_path.name})")
    if model_params:
        print(f"Параметри з останнього tune_model.py: {model_params}")

    stats = train_intent_model(
        data_path,
        MODEL,
        model_name=model_name,
        feature_set=feature_set,
        test_size=0.2,
        model_params=model_params,
    )

    METRICS.write_text(
        json.dumps(
            {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "model_name": model_name,
                "feature_set": feature_set,
                "data_path": str(data_path),
                "total_samples": stats["train_size"],
                "train_size": stats["train_size"],
                "evaluation_train_size": stats["evaluation_train_size"],
                "test_size": stats["test_size"],
                "cv_folds": stats["cv_folds"],
                "cv_accuracy_mean": stats["cv_accuracy_mean"],
                "cv_accuracy_std": stats["cv_accuracy_std"],
                "cv_f1_macro_mean": stats["cv_f1_macro_mean"],
                "cv_f1_macro_std": stats["cv_f1_macro_std"],
                "accuracy": stats["accuracy"],
                "f1_macro": stats["f1_macro"],
                "labels": stats["labels"],
                "confusion_matrix": stats["confusion_matrix"],
                "classification_report": stats["report"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Модель: {MODEL}")
    print(f"Accuracy: {stats['accuracy']:.4f}  F1 macro: {stats['f1_macro']:.4f}")
    print(f"Метрики: {METRICS}")
    print("\n--- Classification report ---\n")
    print(stats["report_text"])
    print(
        "\nДалі: python experiments/confidence_threshold.py -> "
        "python scripts/generate_ml_report.py"
    )


if __name__ == "__main__":
    main()
