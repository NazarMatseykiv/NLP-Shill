import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .menu import MenuMatcher

WORD_RE = re.compile(r"[\w']+", re.UNICODE)
ORDER_ID_RE = re.compile(
    r"(?<!\w)(?:(?P<label>order|замовленн\w*|замовл\w*|заказ\w*|"
    r"zamovlenn\w*|zakaz\w*)\s*(?:№|#)?\s*|"
    r"(?:№|#)\s*|(?=[a-z]{1,3}-\d{2,}))"
    r"[a-z]{0,3}[- ]?\d{2,}(?!\w)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")


@lru_cache(maxsize=1)
def _default_menu():
    from .menu import MenuMatcher

    path = Path(__file__).resolve().parents[1] / "data" / "menu.json"
    return MenuMatcher(path) if path.exists() else None


def normalize_text(
    text: str,
    menu: "MenuMatcher | None" = None,
    *,
    mask_entities: bool = True,
) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower())
    if mask_entities:
        if menu is None:
            menu = _default_menu()
        if menu is not None:
            text = menu.mask_dishes(text)
        text = ORDER_ID_RE.sub(
            lambda match: (
                f"{match.group('label')} " if match.group("label") else ""
            )
            + "<id>",
            text,
        )
        text = NUMBER_RE.sub("<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text, mask_entities=False))


def lemmatize_token(token: str, morph) -> str:
    parsed = morph.parse(token)
    if not parsed:
        return token
    return parsed[0].normal_form
