import re
import unicodedata

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def lemmatize_token(token: str, morph) -> str:
    parsed = morph.parse(token)
    if not parsed:
        return token
    return parsed[0].normal_form
