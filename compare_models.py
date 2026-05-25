import json
from datetime import datetime
from pathlib import Path

from src.data_quality import analyze_dataset
from src.evaluation import compare_models, resolve_data_path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "model_comparison.json"


def main() -> None:
    data_path = resolve_data_path(ROOT)
    quality = analyze_dataset(data_path)
    (ROOT / "models" / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Якість даних: {ROOT / 'models' / 'data_quality.json'}")

    all_results = {}
    for feature_set in ("char", "word", "combined"):
        print(f"\n=== Feature set: {feature_set} ===")
        result = compare_models(data_path, feature_set=feature_set)
        all_results[feature_set] = result
        for row in result["ranking"]:
            print(
                f"  {row['model']:22}  CV F1={row['cv_f1_macro_mean']:.4f}  "
                f"test F1={row['test_f1_macro']:.4f}  acc={row['test_accuracy']:.4f}"
            )

    best = None
    for fs, res in all_results.items():
        top = res["ranking"][0]
        key = (top["cv_f1_macro_mean"], top["test_f1_macro"])
        candidate = {
            "model": top["model"],
            "feature_set": fs,
            "cv_f1_macro_mean": top["cv_f1_macro_mean"],
            "test_f1_macro": top["test_f1_macro"],
        }
        if best is None or key > (best["cv_f1_macro_mean"], best["test_f1_macro"]):
            best = candidate

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "data_quality_summary": {
            "total_samples": quality["total_samples"],
            "class_imbalance_ratio": quality["class_imbalance_ratio"],
            "duplicate_texts": quality["duplicate_texts"],
        },
        "by_feature_set": all_results,
        "recommended": best,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config_path = ROOT / "models" / "best_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": best["model"],
                "feature_set": best["feature_set"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nРекомендовано: {best['model']} + {best['feature_set']}")
    print(f"Збережено: {OUT}")
    print(f"Конфіг для train.py: {config_path}")
    print("Таблиця: python scripts/export_comparison_table.py")


if __name__ == "__main__":
    main()
