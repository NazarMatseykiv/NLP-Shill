import json
import sys
from pathlib import Path

from src.pipeline import FoodOrderNLP

ROOT = Path(__file__).resolve().parent


def main() -> None:
    nlp = FoodOrderNLP(ROOT)
    print("Food Delivery NLP (укр). Порожній рядок — вихід.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        result = nlp.process(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nlp = FoodOrderNLP(ROOT)
        print(json.dumps(nlp.process(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
    else:
        main()
