import re
from dataclasses import dataclass, field

from .menu import MenuMatcher, _MORPH
from .preprocess import normalize_text, tokenize, lemmatize_token

QUANTITY_WORDS = {
  # Ukrainian
    "один": 1,
    "одна": 1,
    "одне": 1,
    "два": 2,
    "дві": 2,
    "три": 3,
    "чотири": 4,
    "п'ять": 5,
    "пять": 5,
    "шість": 6,
    "сім": 7,
    "вісім": 8,
    "дев'ять": 9,
    "десять": 10,
    # English
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

STOP_TOKENS = {
    # Ukrainian common small words
    "і",
    "та",
    "на",
    "в",
    "у",
    "з",
    "по",
    "до",
    "за",
    "але",
    "що",
    "то",
    "чи",
    "а",
    # English
    "and",
    "or",
    "to",
    "on",
    "in",
    "the",
    "a",
    "an",
    "хочу",
    "замовити",
    "замовлення",
    "потрібно",
    "потрібен",
    "потрібна",
    "привезіть",
    "доставте",
    "оформіть",
    "want",
    "order",
    "please",
    "hi",
    "hello",
    "could",
    "would",
    "bring",
    "delivered",
    "deliver",
    "want",
    "like",
    "send",
    "need",
    "can",
    "you",
    "your",
    "i'd",
    "i",
    "for",
    "at",
    "by",
    "address",
    "is",
    "the",
    "будь",
    "ласка",
    "доставки",
    "для",
    "додайте",
    "адреса",
}

PAYMENT_PATTERNS = [
    (re.compile(r"\b(карткою|картою|картка|безготівк\w*)\b", re.I), "card"),
    (re.compile(r"\b(готівк\w*|готівкою)\b", re.I), "cash"),
    (
        re.compile(
            r"\b(pay(?:ment)?\s+(?:by|with)\s+card|credit card|debit card|by card|card)\b",
            re.I,
        ),
        "card",
    ),
    (re.compile(r"\b(cash|cash on delivery|cod)\b", re.I), "cash"),
    (re.compile(r"\b(online|онлайн)\b", re.I), "online"),
]

ADDRESS_PATTERNS = [
    re.compile(
        r"\b(?:вул\.?|вулиц[яі]|просп\.?|проспект|пров\.?|провулок)\s+"
        r"[\w'’\-]+(?:\s+[\w'’\-]+){0,3}\s*,?\s*\d{1,5}(?:/\d+)?"
        r"(?:\s*(?:кв\.?|квартира)\s*\d+)?",
        re.I | re.UNICODE,
    ),
    re.compile(
        r"\b\d{1,5}\s+[\w'’\-]+(?:\s+[\w'’\-]+)?\s+"
        r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|drive|dr\.?|boulevard|blvd\.?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:to|at)\s+"
        r"[\w'’\-]+(?:\s+[\w'’\-]+){0,2}\s+"
        r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|drive|dr\.?|boulevard|blvd\.?)"
        r"\s+\d{1,5}(?:/\d+)?"
        r"(?:\s*(?:apt\.?|apartment|flat)\s*\d+)?",
        re.I,
    ),
    re.compile(
        r"\b[\w'’\-]+(?:\s+[\w'’\-]+)?\s+"
        r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|drive|dr\.?|boulevard|blvd\.?)"
        r"\s+\d{1,5}(?:/\d+)?"
        r"(?:\s*(?:apt\.?|apartment|flat)\s*\d+)?",
        re.I,
    ),
    re.compile(
        r"\b(?:на|по|адрес(?:а)?\s*:?)\s+"
        r"(?P<street>[^\W\d_]+(?:['’\-][^\W\d_]+)*(?:\s+[^\W\d_]+(?:['’\-][^\W\d_]+)*){0,2})"
        r"\s*,?\s+(?P<number>\d{1,5})(?:/\d+)?"
        r"(?:\s*(?:кв\.?|квартира)\s*\d+)?",
        re.I | re.UNICODE,
    ),
    re.compile(
        r"\b(?:хрещатик|саксаганського)\s*\d{1,5}\b",
        re.I | re.UNICODE,
    ),
]

TRANSLITERATION_DIGRAPHS = (
    ("shch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("yu", "ю"),
    ("ya", "я"),
    ("ye", "є"),
    ("yi", "ї"),
    ("ie", "є"),
    ("iu", "ю"),
)
TRANSLITERATION_CHARS = str.maketrans({
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "г",
    "i": "і",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
})


def _transliterate_ukrainian(text: str) -> str:
    def convert_word(match: re.Match) -> str:
        word = match.group()
        if not re.fullmatch(r"[a-z]+", word):
            return word
        for latin, cyrillic in TRANSLITERATION_DIGRAPHS:
            word = word.replace(latin, cyrillic)
        return word.translate(TRANSLITERATION_CHARS)

    return re.sub(r"[a-z]+", convert_word, text)

COMMENT_PATTERNS = [
    re.compile(r"\b(без\s+[\wіїєґ']+(?:\s+[\wіїєґ']+)?)\b", re.I | re.UNICODE),
    re.compile(r"\b(no\s+[\w']+(?:\s+[\w']+)?|without\s+[\w']+(?:\s+[\w']+)?)\b", re.I),
    re.compile(r"\b(extra\s+[\w']+)\b", re.I),
]

TIME_PATTERNS = [
    re.compile(r"\b(на\s+вечір|до\s+\d{1,2}[:.]\d{2}|о\s+\d{1,2}[:.]\d{2})\b", re.I),
    re.compile(r"\b(asap|right now|tonight|tomorrow morning)\b", re.I),
    re.compile(r"\b(зараз|якомога\s+швидше|терміново)\b", re.I),
]


@dataclass
class DishItem:
    dish_id: int
    dish_name: str
    quantity: int
    raw: str


@dataclass
class OrderSlots:
    dishes: list[DishItem] = field(default_factory=list)
    address: str | None = None
    payment: str | None = None
    comment: str | None = None
    time_hint: str | None = None

    def to_dict(self) -> dict:
        return {
            "dishes": [
                {
                    "dish_id": d.dish_id,
                    "dish_name": d.dish_name,
                    "quantity": d.quantity,
                    "raw": d.raw,
                }
                for d in self.dishes
            ],
            "address": self.address,
            "payment": self.payment,
            "comment": self.comment,
            "time_hint": self.time_hint,
        }


class SlotExtractor:
    """Rule + dictionary slot filling (UA + EN)."""

    def __init__(self, menu: MenuMatcher):
        self.menu = menu
        aliases = []
        for item in menu.items:
            aliases.extend(item.get("aliases", []))
            aliases.append(item["name"])
        # include lemmatized alias forms when lemmatizer available
        alias_set = {a.lower() for a in aliases if a}
        if _MORPH and tokenize and lemmatize_token:
            lemmata = set()
            for a in list(alias_set):
                toks = tokenize(a)
                if toks:
                    lemma = " ".join(lemmatize_token(t, _MORPH) for t in toks)
                    if lemma:
                        lemmata.add(lemma)
            alias_set.update(lemmata)
        # Longest match first; allow multi-word aliases
        aliases = sorted(alias_set, key=len, reverse=True)
        # determine max alias length in tokens for n-gram matching
        try:
            self.max_alias_len = max(len(tokenize(a)) for a in aliases) if aliases else 1
            # cap length to avoid matching large spans (addresses, comments)
            self.max_alias_len = min(self.max_alias_len, 4)
        except Exception:
            self.max_alias_len = 4
        self.dish_re = None

    def _parse_quantity_before(self, text: str, start: int) -> int:
        prefix = text[:start].strip()
        tokens = prefix.split()[-4:]
        for token in reversed(tokens):
            token_l = token.lower()
            if token_l in QUANTITY_WORDS:
                return QUANTITY_WORDS[token_l]
            if token_l.isdigit():
                return int(token_l)
        m = re.search(r"(\d{1,2})\s*$", prefix)
        if m:
            return int(m.group(1))
        return 1

    def extract(self, text: str) -> OrderSlots:
        text_n = normalize_text(text, mask_entities=False)
        slots = OrderSlots()

        payment_text = None
        payment_match = None
        for pattern, payment in PAYMENT_PATTERNS:
            m = pattern.search(text_n)
            if m:
                slots.payment = payment
                payment_text = m.group(0)
                payment_match = m
                break

        comment_text = None
        comment_end = payment_match.start() if payment_match else len(text_n)
        for pattern in COMMENT_PATTERNS:
            m = pattern.search(text_n, 0, comment_end)
            if m:
                slots.comment = m.group(1).strip()
                comment_text = m.group(0)
                break

        time_text = None
        for pattern in TIME_PATTERNS:
            m = pattern.search(text_n, 0, comment_end)
            if m:
                slots.time_hint = m.group(1).strip()
                time_text = m.group(0)
                break

        trailing_comment = None
        if payment_match:
            trailing_comment = text_n[payment_match.end() :]
            trailing_comment = trailing_comment.strip(" \t\r\n,.;:!?-—")
            if trailing_comment:
                slots.comment = (
                    f"{slots.comment}, {trailing_comment}"
                    if slots.comment
                    else trailing_comment
                )

        # remove detected payment/comment/time substrings from normalized text
        for s in (
            payment_text,
            comment_text,
            text_n[payment_match.end() :] if payment_match and trailing_comment else None,
            time_text,
        ):
            if s:
                text_n = text_n.replace(
                    normalize_text(s, mask_entities=False), " "
                )

        address_match = None
        address_text = text_n
        for candidate_text in (text_n, _transliterate_ukrainian(text_n)):
            for pattern in ADDRESS_PATTERNS:
                address_match = pattern.search(candidate_text)
                if address_match:
                    address_text = candidate_text
                    break
            if address_match:
                break
        if address_match:
            slots.address = re.sub(
                r"^(?:на|по|адрес(?:а)?\s*:|to|at)\s+",
                "",
                address_match.group().strip(" ,"),
                flags=re.I,
            )
            text_n = address_text
            text_n = (
                text_n[: address_match.start()]
                + " "
                + text_n[address_match.end() :]
            )

        tokens = tokenize(text_n)
        seen: set[int] = set()
        last_item_end = 0
        i = 0
        while i < len(tokens):
            # skip starting on explicit quantity tokens, digits, or stop-words (prepositions/conjunctions)
            ti = tokens[i].lower()
            max_n = min(self.max_alias_len, len(tokens) - i)
            exact_match = None
            exact_n = 0
            for n in range(max_n, 0, -1):
                phrase_tokens = tokens[i : i + n]
                phrase = " ".join(phrase_tokens)
                item = self.menu.alias_to_item.get(phrase)
                if item is None and _MORPH:
                    lemma_phrase = " ".join(
                        lemmatize_token(token, _MORPH) for token in phrase_tokens
                    )
                    item = self.menu.alias_to_item.get(lemma_phrase)
                if item and item["id"] not in seen:
                    exact_match = item
                    exact_n = n
                    break

            if (
                exact_match is None
                and (
                    ti in QUANTITY_WORDS
                    or tokens[i].isdigit()
                    or ti in STOP_TOKENS
                    or len(ti) < 3
                )
            ):
                i += 1
                continue

            matched_item = exact_match
            matched_n = exact_n
            if matched_item is None:
                item = self.menu.match_dish(tokens[i])
                if item and item["id"] not in seen:
                    matched_item = item
                    matched_n = 1

            if matched_item:
                qty = 1
                for t in reversed(tokens[max(last_item_end, i - 4) : i]):
                    tl = t.lower()
                    if tl in QUANTITY_WORDS:
                        qty = QUANTITY_WORDS[tl]
                        break
                    if tl.isdigit():
                        qty = int(tl)
                        break
                slots.dishes.append(
                    DishItem(
                        dish_id=matched_item["id"],
                        dish_name=matched_item["name"],
                        quantity=qty,
                        raw=" ".join(tokens[i : i + matched_n]),
                    )
                )
                seen.add(matched_item["id"])
                last_item_end = i + matched_n
                i += matched_n
            else:
                i += 1

        if not slots.dishes:
            transliterated_text = _transliterate_ukrainian(text_n)
            tokens = tokenize(transliterated_text)
            last_item_end = 0
            i = 0
            while i < len(tokens):
                ti = tokens[i].lower()
                max_n = min(self.max_alias_len, len(tokens) - i)
                matched_item = None
                matched_n = 0
                if (
                    ti not in QUANTITY_WORDS
                    and not tokens[i].isdigit()
                    and ti not in STOP_TOKENS
                    and len(ti) >= 3
                ):
                    for n in range(max_n, 0, -1):
                        phrase_tokens = tokens[i : i + n]
                        phrase = " ".join(phrase_tokens)
                        item = self.menu.alias_to_item.get(phrase)
                        if item is None and n == 1:
                            item = self.menu.match_dish(phrase)
                        if item:
                            matched_item = item
                            matched_n = n
                            break
                if matched_item:
                    quantity = 1
                    for token in reversed(tokens[max(last_item_end, i - 4) : i]):
                        if token.lower() in QUANTITY_WORDS:
                            quantity = QUANTITY_WORDS[token.lower()]
                            break
                        if token.isdigit():
                            quantity = int(token)
                            break
                    slots.dishes.append(
                        DishItem(
                            dish_id=matched_item["id"],
                            dish_name=matched_item["name"],
                            quantity=quantity,
                            raw=" ".join(tokens[i : i + matched_n]),
                        )
                    )
                    last_item_end = i + matched_n
                    i += matched_n
                else:
                    i += 1

        return slots
