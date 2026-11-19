import json
from pathlib import Path

from src.data_quality import analyze_dataset
from src.evaluation import resolve_data_path
from src.menu import MenuMatcher
from src.slots import SlotExtractor
from sklearn.model_selection import cross_val_score

from src.intent_model import build_intent_pipeline, load_intent_data

ROOT = Path(__file__).resolve().parent

SLOT_TEST_CASES = [
    {
        "text": "Два борщі та салат на вул. Шевченка 12",
        "expected_dishes": ["Borscht / Борщ", "Caesar Salad"],
        "address_contains": "Шевченка",
    },
    {
        "text": "Two burgers and fries delivered to 12 Main Street pay by card, leave at the door",
        "expected_dishes": ["Burger", "French Fries"],
        "address_contains": "Main Street",
        "payment": "card",
        "comment_contains": "leave at the door",
    },
    {
        "text": "На вул. Шевченка 12 хочу два бургери та піцу, карткою",
        "expected_dishes": ["Burger", "Pizza Margherita"],
        "expected_quantities": {"Burger": 2, "Pizza Margherita": 1},
        "address_contains": "Шевченка",
        "payment": "card",
    },
    {
        "text": "На вул. Шевченка 12 хочу піцу пепероні та four cheese pizza, карткою",
        "expected_dishes": [
            "Pizza Pepperoni / Піца Пепероні",
            "Pizza Quattro Formaggi / Піца 4 сири",
        ],
        "exact_dishes": [
            "Pizza Pepperoni / Піца Пепероні",
            "Pizza Quattro Formaggi / Піца 4 сири",
        ],
        "address_contains": "Шевченка",
        "payment": "card",
    },
    {
        "text": "One pizza no onions at 5 Oak Avenue",
        "expected_dishes": ["Pizza Margherita"],
        "comment_contains": "no",
    },
]


def evaluate_intent_cv(model_name: str, feature_set: str) -> dict:
    data = resolve_data_path(ROOT)
    texts, labels = load_intent_data(data)
    pipe = build_intent_pipeline(model_name, feature_set)
    scores = cross_val_score(pipe, texts, labels, cv=5, scoring="f1_macro", n_jobs=-1)
    return {
        "model": model_name,
        "feature_set": feature_set,
        "cv_folds": 5,
        "f1_macro_mean": float(scores.mean()),
        "f1_macro_std": float(scores.std()),
        "f1_macro_scores": [float(s) for s in scores],
    }


def evaluate_slots() -> dict:
    menu = MenuMatcher(ROOT / "data" / "menu.json")
    extractor = SlotExtractor(menu)
    passed = 0
    details = []
    for case in SLOT_TEST_CASES:
        slots = extractor.extract(case["text"]).to_dict()
        dish_names = [d["dish_name"] for d in slots["dishes"]]
        ok = all(name in dish_names for name in case.get("expected_dishes", []))
        if case.get("exact_dishes"):
            ok = ok and set(dish_names) == set(case["exact_dishes"])
        quantities = {dish["dish_name"]: dish["quantity"] for dish in slots["dishes"]}
        ok = ok and all(
            quantities.get(name) == quantity
            for name, quantity in case.get("expected_quantities", {}).items()
        )
        if case.get("address_contains"):
            address = (slots["address"] or "").casefold()
            expected_address = case["address_contains"].casefold()
            ok = ok and expected_address in address
        if case.get("payment"):
            ok = ok and slots["payment"] == case["payment"]
        if case.get("comment_contains"):
            comment = (slots["comment"] or "").casefold()
            expected_comment = case["comment_contains"].casefold()
            ok = ok and expected_comment in comment
        passed += int(ok)
        details.append({"text": case["text"], "ok": ok, "slots": slots})
    return {
        "slot_tests_total": len(SLOT_TEST_CASES),
        "slot_tests_passed": passed,
        "slot_accuracy": passed / len(SLOT_TEST_CASES),
        "details": details,
    }


def main() -> None:
    data = resolve_data_path(ROOT)
    quality = analyze_dataset(data)
    print("Data quality:", json.dumps(quality["intent_distribution"], ensure_ascii=False))

    cfg_path = ROOT / "models" / "best_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model_name, feature_set = cfg["model_name"], cfg["feature_set"]
    else:
        model_name, feature_set = "logistic_regression", "combined"

    print(f"\nCV for selected model ({model_name}, {feature_set})...")
    intent_cv = evaluate_intent_cv(model_name, feature_set)
    print(json.dumps(intent_cv, ensure_ascii=False, indent=2))

    print("\nSlot extraction...")
    slot_eval = evaluate_slots()
    print(json.dumps(slot_eval, ensure_ascii=False, indent=2))

    out = ROOT / "models" / "evaluation_full.json"
    out.write_text(
        json.dumps(
            {
                "data_quality": quality,
                "intent_cv": intent_cv,
                "slots": slot_eval,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
