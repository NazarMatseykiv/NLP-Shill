import json
from pathlib import Path

from rapidfuzz import fuzz, process

try:
    import pymorphy2
    from .preprocess import tokenize, lemmatize_token
    _MORPH = pymorphy2.MorphAnalyzer()
except Exception:
    _MORPH = None
    tokenize = None
    lemmatize_token = None


class MenuMatcher:
    def __init__(self, menu_path: Path):
        with menu_path.open(encoding="utf-8") as f:
            items = json.load(f)
        self.items = items
        self.alias_to_item: dict[str, dict] = {}
        for item in items:
            for alias in item.get("aliases", []) + [item["name"]]:
                key = alias.lower()
                self.alias_to_item[key] = item
                # add lemmatized alias if lemmatizer is available
                if _MORPH and tokenize and lemmatize_token:
                    toks = tokenize(alias)
                    if toks:
                        lemma = " ".join(lemmatize_token(t, _MORPH) for t in toks)
                        if lemma:
                            self.alias_to_item[lemma] = item
        self.choices = list(self.alias_to_item.keys())

    def match_dish(self, phrase: str, threshold: int = 82) -> dict | None:
        phrase = phrase.lower().strip()
        if not phrase:
            return None
        # exact match raw
        if phrase in self.alias_to_item:
            return self.alias_to_item[phrase]
        # try lemmatized phrase
        if _MORPH and tokenize and lemmatize_token:
            toks = tokenize(phrase)
            if toks:
                lemma = " ".join(lemmatize_token(t, _MORPH) for t in toks)
                if lemma in self.alias_to_item:
                    return self.alias_to_item[lemma]

        # fallback to fuzzy match
        result = process.extractOne(
            phrase,
            self.choices,
            scorer=fuzz.partial_ratio,
        )
        if result and result[1] >= threshold:
            return self.alias_to_item[result[0]]
        return None
