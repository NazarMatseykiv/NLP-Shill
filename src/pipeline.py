from pathlib import Path

import joblib

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

    def process(self, text: str) -> dict:
        intent, confidence = predict_intent(self.intent_model, text)
        result = {
            "text": text,
            "intent": intent,
            "intent_confidence": round(confidence, 4),
        }
        if intent == "create_order":
            result["slots"] = self.slots.extract(text).to_dict()
        return result
