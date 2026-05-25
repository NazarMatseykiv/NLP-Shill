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
}

PAYMENT_PATTERNS = [
    (re.compile(r"\b(карткою|картою|картка|безготівк\w*)\b", re.I), "card"),
    (re.compile(r"\b(готівк\w*|готівкою)\b", re.I), "cash"),
    (re.compile(r"\b(card|by card|credit card|debit card)\b", re.I), "card"),
    (re.compile(r"\b(cash|cash on delivery|cod)\b", re.I), "cash"),
    (re.compile(r"\b(online|онлайн)\b", re.I), "online"),
]

ADDRESS_PATTERNS = [
    re.compile(
        r"(?:вул\.?|вулиц[яі]\s+|просп\.?|проспект\s+|пров\.?|провулок\s+)"
        r"([a-zа-яіїєґ0-9\s\.\-]+?)(?:\s*(?:\d+)|\s*(?:кв\.?)|$)",
        re.I | re.UNICODE,
    ),
    re.compile(
        r"\b(?:deliver(?:y)?\s+to|delivered\s+to|address|at)\s+"
        r"(\d{1,5}\s+[\w\s\.\-]+?(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|drive|dr\.?))",
        re.I,
    ),
    re.compile(
        r"\b(\d{1,5}\s+[\w\s\.\-]+(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|drive))\b",
        re.I,
    ),
    re.compile(r"\b(хрещатик\s*\d+|саксаганського\s*\d+)\b", re.I | re.UNICODE),
]

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
        text_n = normalize_text(text)
        slots = OrderSlots()

        payment_text = None
        for pattern, payment in PAYMENT_PATTERNS:
            m = pattern.search(text_n)
            if m:
                slots.payment = payment
                payment_text = m.group(0)
                break

        comment_text = None
        for pattern in COMMENT_PATTERNS:
            m = pattern.search(text_n)
            if m:
                slots.comment = m.group(1).strip()
                comment_text = m.group(0)
                break

        time_text = None
        for pattern in TIME_PATTERNS:
            m = pattern.search(text_n)
            if m:
                slots.time_hint = m.group(1).strip()
                time_text = m.group(0)
                break

        # remove detected payment/comment/time substrings from normalized text
        for s in (payment_text, comment_text, time_text):
            if s:
                text_n = text_n.replace(normalize_text(s), " ")

        # tokenize and detect address heuristically from tokens (avoid regex over-capture)
        tokens = tokenize(text_n)
        ADDRESS_KEYWORDS = {
            "street",
            "st",
            "road",
            "rd",
            "avenue",
            "ave",
            "lane",
            "drive",
            "dr",
            "main",
            "blvd",
            "boulevard",
            "вул",
            "вулиця",
            "проспект",
        }
        addr_cut = None
        for idx, t in enumerate(tokens):
            tl = t.lower().strip(".,")
            if tl in ADDRESS_KEYWORDS:
                addr_cut = max(0, idx - 1)
                break
            if tl == "to" and idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                addr_cut = idx
                break
            if tokens[idx].isdigit() and idx + 1 < len(tokens) and tokens[idx + 1].lower().strip(".,") in ADDRESS_KEYWORDS:
                addr_cut = idx
                break

        if addr_cut is not None:
            # reconstruct address from tokens and trim tokens for dish extraction
            addr = " ".join(tokens[addr_cut:]).strip()
            # drop leading 'to' if present
            if addr.lower().startswith("to "):
                addr = addr[3:]
            slots.address = addr
            tokens = tokens[:addr_cut]

        seen: set[int] = set()
        # detect likely start of address tokens (English/UA keywords)
        ADDRESS_KEYWORDS = {
            "street",
            "st",
            "road",
            "rd",
            "avenue",
            "ave",
            "lane",
            "drive",
            "dr",
            "main",
            "blvd",
            "boulevard",
            "вул",
            "вулиця",
            "проспект",
        }
        addr_cut = None
        for idx, t in enumerate(tokens):
            if t.lower().strip(".,") in ADDRESS_KEYWORDS:
                addr_cut = max(0, idx - 1)
                break
        i = 0
        while i < len(tokens):
            if addr_cut is not None and i >= addr_cut:
                break
            # skip starting on explicit quantity tokens, digits, or stop-words (prepositions/conjunctions)
            ti = tokens[i].lower()
            if ti in QUANTITY_WORDS or tokens[i].isdigit() or ti in STOP_TOKENS or len(ti) < 3:
                i += 1
                continue
            matched = False
            # try longest n-gram first
            for n in range(1, self.max_alias_len + 1):
                if i + n > len(tokens):
                    continue
                phrase = " ".join(tokens[i : i + n])
                item = self.menu.match_dish(phrase)
                if item and item["id"] not in seen:
                    # parse quantity from tokens before i
                    qty = 1
                    for t in reversed(tokens[max(0, i - 4) : i]):
                        tl = t.lower()
                        if tl in QUANTITY_WORDS:
                            qty = QUANTITY_WORDS[tl]
                            break
                        if tl.isdigit():
                            qty = int(tl)
                            break
                    slots.dishes.append(
                        DishItem(
                            dish_id=item["id"],
                            dish_name=item["name"],
                            quantity=qty,
                            raw=phrase,
                        )
                    )
                    seen.add(item["id"])
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1

        return slots
