import json
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
    print(f"Навчання: {model_name} + {feature_set} ({data_path.name})")

    stats = train_intent_model(
        data_path,
        MODEL,
        model_name=model_name,
        feature_set=feature_set,
        test_size=0.2,
    )

    METRICS.write_text(
        json.dumps(
            {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "model_name": model_name,
                "feature_set": feature_set,
                "train_size": stats["train_size"],
                "test_size": stats["test_size"],
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
    print("\nДалі: python tune_model.py  →  python scripts/generate_ml_report.py")


if __name__ == "__main__":
    main()
