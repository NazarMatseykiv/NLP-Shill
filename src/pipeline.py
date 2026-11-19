from pathlib import Path

import joblib
import json

from .intent_model import predict_intent
from .menu import MenuMatcher
from .slots import SlotExtractor


class FoodOrderNLP:
    def __init__(self, root: Path):
        self.root = root
        self.menu = MenuMatcher(root / "data" / "menu.json")
        self.slots = SlotExtractor(self.menu)
        model_path = root / "models" / "intent_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не знайдена: {model_path}. Спочатку запустіть train.py"
            )
        self.intent_model = joblib.load(model_path)
        threshold_path = root / "reports" / "threshold.json"
        self.confidence_threshold = 0.55
        if threshold_path.exists():
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            chosen = threshold_data.get("chosen")
            if chosen is not None:
                threshold = float(chosen["threshold"])
                if not 0 <= threshold <= 1:
                    raise ValueError(f"Некоректний поріг впевненості: {threshold}")
                self.confidence_threshold = threshold

    def process(self, text: str) -> dict:
        predicted_intent, confidence = predict_intent(
            self.intent_model, text, self.menu
        )
        below_threshold = confidence < self.confidence_threshold
        intent = "other" if below_threshold else predicted_intent
        result = {
            "text": text,
            "intent": intent,
            "intent_confidence": round(confidence, 4),
        }
        if below_threshold and predicted_intent != "other":
            result["suggested_intent"] = predicted_intent
        if intent == "create_order":
            result["slots"] = self.slots.extract(text).to_dict()
        return result
