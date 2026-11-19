import json
import re
import unicodedata
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
        aliases = set()
        for item in items:
            for alias in item.get("aliases", []) + [item["name"]]:
                key = unicodedata.normalize("NFC", alias.strip().lower())
                self.alias_to_item[key] = item
                aliases.add(key)
                # add lemmatized alias if lemmatizer is available
                if _MORPH and tokenize and lemmatize_token:
                    toks = tokenize(alias)
                    if toks:
                        lemma = " ".join(lemmatize_token(t, _MORPH) for t in toks)
                        if lemma:
                            self.alias_to_item[lemma] = item
                            aliases.add(lemma)
        for alias in tuple(aliases):
            if re.fullmatch(r"[a-z]+", alias):
                if alias.endswith("y") and len(alias) > 1:
                    plural = alias[:-1] + "ies"
                elif alias.endswith(("s", "x", "ch", "sh")):
                    plural = alias + "es"
                else:
                    plural = alias + "s"
                aliases.add(plural)
                self.alias_to_item[plural] = self.alias_to_item[alias]
            elif alias.endswith("а"):
                plural = alias[:-1] + "и"
                aliases.add(plural)
                self.alias_to_item[plural] = self.alias_to_item[alias]
        self.choices = list(self.alias_to_item.keys())
        self._mask_patterns = [
            re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
            for alias in sorted(aliases, key=len, reverse=True)
        ]

    def mask_dishes(self, text: str) -> str:
        for pattern in self._mask_patterns:
            text = pattern.sub("<dish>", text)
        return text

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
