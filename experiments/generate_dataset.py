from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# Лексикон
DISHES = [
    ("борщ", "борщ", "borscht", "borscht"),
    ("піца маргарита", "піцу маргарита", "margherita pizza", "margherita pizzas"),
    ("піца пепероні", "піцу пепероні", "pepperoni pizza", "pepperoni pizzas"),
    ("салат цезар", "салат цезар", "caesar salad", "caesar salads"),
    ("бургер", "бургер", "burger", "burgers"),
    ("картопля фрі", "картоплю фрі", "fries", "fries"),
    ("паста карбонара", "пасту карбонара", "carbonara", "carbonaras"),
    ("вареники", "вареники", "dumplings", "dumplings"),
    ("сирники", "сирники", "syrniki", "syrniki"),
    ("суші-сет", "суші-сет", "sushi set", "sushi sets"),
    ("лате", "лате", "latte", "lattes"),
    ("американо", "американо", "americano", "americanos"),
    ("чізкейк", "чізкейк", "cheesecake", "cheesecakes"),
    ("омлет", "омлет", "omelette", "omelettes"),
    ("котлета по-київськи", "котлету по-київськи", "chicken kyiv", "chicken kyivs"),
    ("голубці", "голубці", "cabbage rolls", "cabbage rolls"),
    ("шаурма", "шаурму", "shawarma", "shawarmas"),
    ("крем-суп", "крем-суп", "cream soup", "cream soups"),
    ("лимонад", "лимонад", "lemonade", "lemonades"),
    ("мохіто", "мохіто", "mojito", "mojitos"),
    ("тірамісу", "тірамісу", "tiramisu", "tiramisu"),
    ("гречана каша з грибами", "гречану кашу з грибами", "buckwheat with mushrooms", "buckwheat with mushrooms"),
    ("coca-cola", "колу", "cola", "colas"),
    ("хот-дог", "хот-дог", "hot dog", "hot dogs"),
]

STREETS_UA = ["вул. Шевченка", "вулиця Лесі Українки", "пр. Свободи", "вул. Головна", "вул. Руська",
              "вул. Кобилянської", "вул. Франка", "вул. Незалежності", "вул. Соборна", "вул. Хотинська",
              "вул. Героїв Майдану", "Університетська", "бульвар Лесі Українки", "пл. Центральна"]
STREETS_EN = ["Main Street", "Park Avenue", "Elm Road", "Baker Street", "Oak Lane", "River Street",
              "Maple Drive", "High Street", "Station Road", "Church Lane"]
DISTRICTS_UA = ["Садгору", "Рогізну", "центр міста", "Магала", "передмістя", "університетський кампус"]
DISTRICTS_EN = ["the city center", "the suburbs", "the campus", "the airport area", "the old town"]

PAY_UA = ["карткою", "готівкою", "оплата карткою", "оплачу кур'єру готівкою", "онлайн", "оплата при отриманні",
          "Apple Pay", "розрахуюсь на місці", "кеш", "терміналом"]
PAY_EN = ["by card", "cash", "pay online", "cash on delivery", "Apple Pay", "I'll pay the courier", "card payment"]
TIME_UA = ["на 19:00", "на 18:30", "якнайшвидше", "через годину", "завтра на обід", "на сьогодні ввечері", "до 20:00"]
TIME_EN = ["at 7 pm", "asap", "in an hour", "tomorrow for lunch", "tonight", "by 8 pm"]

FILL_PRE_UA = ["Привіт!", "Добрий день,", "Слухайте,", "Доброго вечора!", "Вітаю,", "Підкажіть,", "Алло,", ""]
FILL_PRE_EN = ["Hi,", "Hello!", "Hey,", "Good evening,", "Quick question:", "Excuse me,", ""]
FILL_SUF_UA = ["дякую", "будь ласка", "пж", "заздалегідь дякую", "дуже треба", ""]
FILL_SUF_EN = ["thanks", "please", "plz", "thank you!", "asap pls", ""]
EMOJI = ["🙏", "🍕", "😊", "👍", "🔥", "😅"]

UA_NEIGH = {"а": "вп", "в": "аіу", "о": "лд", "н": "еуг", "е": "нкг", "и": "тіш", "т": "иь", "р": "оп", "с": "ча",
            "к": "уе", "м": "іи", "д": "лж", "л": "од", "п": "ар", "у": "кв", "і": "ви", "г": "нш"}
EN_NEIGH = {"a": "sq", "e": "wr", "o": "ip", "n": "bm", "t": "ry", "r": "et", "s": "ad", "i": "uo", "l": "k",
            "d": "sf", "c": "xv", "u": "yi", "p": "o", "m": "n", "h": "gj", "g": "fh"}

TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ye", "ж": "zh",
            "з": "z", "и": "y", "і": "i", "ї": "yi", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
            "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
            "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "yu", "я": "ya", "'": "", "ʼ": ""}


def translit(text: str) -> str:
    return "".join(TRANSLIT.get(c.lower(), c) for c in text)


def load_menu(path: str | None) -> list[tuple]:
    """Якщо є menu.json — використовуємо реальні назви (форма однакова)."""
    if not path or not Path(path).exists():
        return DISHES
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return DISHES
    out = []
    for it in items:
        name = it.get("name") if isinstance(it, dict) else None
        if isinstance(name, dict):  # {"ua": "...", "en": "..."}
            ua = name.get("ua") or name.get("uk") or next(iter(name.values()))
            en = name.get("en") or ua
        elif isinstance(name, str):
            ua = en = name
        else:
            continue
        out.append((ua, ua, en, en))
    return out or DISHES

# Генератори складових
class Gen:
    def __init__(self, rng: random.Random, dishes):
        self.r, self.dishes = rng, dishes

    def qty(self) -> int:
        return self.r.choices([1, 2, 3, 4, 5], [45, 30, 14, 7, 4])[0]

    def item_ua(self) -> str:
        nom, acc, *_ = self.r.choice(self.dishes)
        n = self.qty()
        if n == 1:
            return self.r.choice([acc, f"{acc} ×1", f"одну порцію: {nom}"])
        return self.r.choice([f"{acc} ×{n}", f"{acc} - {n} шт", f"{acc} ({n} шт)", f"{n} × {nom}", f"{nom} x{n}"])

    def item_en(self) -> str:
        _, _, sg, pl = self.r.choice(self.dishes)
        n = self.qty()
        if n == 1:
            return self.r.choice([f"a {sg}", f"one {sg}", f"{sg} x1"])
        return self.r.choice([f"{n} {pl}", f"{sg} x{n}", f"{n}x {sg}"])

    def items(self, lang: str) -> str:
        k = self.r.choices([1, 2, 3], [45, 40, 15])[0]
        f = self.item_ua if lang == "UA" else self.item_en
        parts = [f() for _ in range(k)]
        if k == 1:
            return parts[0]
        sep = self.r.choice([" і ", ", ", " та ", " + "]) if lang == "UA" else self.r.choice([" and ", ", ", " + "])
        return sep.join(parts)

    def addr(self, lang: str) -> str:
        n = self.r.randint(1, 120)
        if lang == "UA":
            a = f"{self.r.choice(STREETS_UA)} {n}"
            if self.r.random() < 0.3:
                a += f", кв. {self.r.randint(1, 90)}"
        else:
            a = f"{n} {self.r.choice(STREETS_EN)}"
            if self.r.random() < 0.3:
                a += f", apt {self.r.randint(1, 60)}"
        return a

    def order_no(self) -> str:
        n = self.r.randint(100, 9999)
        return self.r.choice([f"{n}", f"№{n}", f"SH-{n}", f"#{n}"])

    def pay(self, lang): return self.r.choice(PAY_UA if lang == "UA" else PAY_EN)
    def time(self, lang): return self.r.choice(TIME_UA if lang == "UA" else TIME_EN)

# Шаблони
T = {
    "create_order": {
        "UA": [
            "Хочу замовити {items} на {addr}, {pay}",
            "Замовте мені {items}, адреса: {addr}",
            "Можна {items} до {addr}? {pay}",
            "Доставте {items} на {addr} {time}",
            "{items} на {addr}",
            "Прошу оформити замовлення: {items}. Доставка: {addr}. Оплата — {pay}",
            "Мені {items}, {addr}, {pay}",
            "Зробіть замовлення на {items}",
            "Хочу {items}",
            "Можна замовити {items} з доставкою на {addr}?",
            "Додайте до замовлення {items}",
            "Оформіть доставку: {items} на {addr}",
            "Потрібно {items} {time}, {addr}",
            "Давайте {items}, привезіть на {addr}",
            "Беру {items}. {pay}",
            "Привезіть на {addr} {items} {time}",
            "{items}, {pay}, {addr}",
            "Я б замовив {items}, якщо встигнете {time}",
        ],
        "EN": [
            "I'd like to order {items} to {addr}, {pay}",
            "Can I get {items} delivered to {addr}?",
            "Please deliver {items} to {addr} {time}",
            "Order: {items}, address {addr}, {pay}",
            "{items} to {addr}",
            "I want {items}",
            "Could you add {items} to my order?",
            "One order please - {items}. {pay}",
            "Send {items} to {addr} {time}",
            "Place an order for {items}",
            "I'll take {items}, {pay}",
            "Get me {items} at {addr}",
            "{items}, {addr}, {pay}",
            "Can you bring {items} {time}? Address is {addr}",
        ],
        "MIX": [
            "Хочу order: {items_en} на {addr_ua}",
            "Привіт, {items_en} будь ласка на {addr_ua}",
            "Замов мені {items_en} to {addr_en}, {pay_ua}",
            "I want {items_ua}, адреса {addr_ua}",
            "Please deliver {items_ua} на {addr_ua}",
        ],
    },
    "track_order": {
        "UA": [
            "Де моє замовлення?",
            "Коли привезуть замовлення {no}?",
            "Який статус замовлення {no}?",
            "Кур'єр уже виїхав?",
            "Скільки ще чекати на замовлення?",
            "Замовлення {no} досі не привезли",
            "Де кур'єр?",
            "Чи далеко мій кур'єр?",
            "Підкажіть, коли буде доставка?",
            "Відстежити замовлення {no}",
            "Чекаю вже годину, де моя їжа?",
            "Статус доставки?",
            "Чи прийняли моє замовлення?",
            "Коли очікувати кур'єра?",
            "Перевірте, будь ласка, де замовлення {no}",
            "Мій заказ {no} коли буде?",
            "Скільки хвилин до прибуття кур'єра?",
            "Чому так довго? Замовлення {no}",
            "Чи вже готується моє замовлення?",
        ],
        "EN": [
            "Where is my order?",
            "What's the status of order {no}?",
            "When will my food arrive?",
            "Has the courier left yet?",
            "Track order {no}",
            "How much longer until my delivery?",
            "My order {no} is late, where is it?",
            "Is my order on the way?",
            "Can you check my delivery status?",
            "ETA for my order please",
            "Where's the courier?",
            "Any update on order {no}?",
            "It's been an hour, where is my food?",
            "Is my order still being prepared?",
        ],
        "MIX": [
            "Де мій order {no}? it's late",
            "Статус my delivery будь ласка",
            "Where is кур'єр?",
            "Коли arrive замовлення {no}?",
        ],
    },
    "cancel_order": {
        "UA": [
            "Скасуйте замовлення",
            "Прошу скасувати замовлення {no}",
            "Я передумав, відмініть замовлення",
            "Не потрібно більше замовлення {no}, скасуйте",
            "Скасуйте будь ласка, помилився адресою",
            "Відмінити замовлення {no}",
            "Анулюйте моє замовлення",
            "Хочу відмовитись від замовлення {no}",
            "Забудьте про замовлення, не треба",
            "Скасуйте {dish}, не хочу",
            "Замовлення оформив випадково, скасуйте",
            "Більше не актуально, відмініть {no}",
            "Мені вже не треба доставка, скасуйте",
            "Видаліть моє замовлення {no}",
            "Не встигну забрати - скасуйте",
        ],
        "EN": [
            "Cancel my order",
            "Please cancel order {no}",
            "I changed my mind, cancel it",
            "I don't want the order anymore",
            "Cancel {no} please",
            "Withdraw my order",
            "I ordered by mistake, please cancel",
            "Stop the order, I no longer need it",
            "Cancel the {dish_en} please",
            "Please remove order {no}",
            "I won't be home, cancel the delivery",
        ],
        "MIX": [
            "Cancel замовлення {no} будь ласка",
            "Скасуйте my order, thanks",
            "Please відмініть {no}",
        ],
    },
    "faq": {
        "UA": [
            "Скільки коштує доставка?",
            "Що є в меню?",
            "До котрої години ви працюєте?",
            "Чи є вегетаріанські страви?",
            "Які способи оплати доступні?",
            "Чи є безкоштовна доставка?",
            "Скільки зазвичай займає доставка?",
            "Чи доставляєте ви в {district}?",
            "Як замовити без реєстрації?",
            "Чи є у вас знижки?",
            "Чи є щось без глютену?",
            "Які алергени у {dish}?",
            "Чи можна оплатити карткою кур'єру?",
            "Яка мінімальна сума замовлення?",
            "Чи приймаєте промокоди?",
            "Скільки коштує {dish}?",
            "Чи можна замовити на завтра?",
            "Як з вами зв'язатися?",
            "Чи працюєте у свята?",
            "Чи є дитяче меню?",
            "Де можна подивитися меню перед замовленням?",
            "Як швидко ви зазвичай привозите замовлення?",
        ],
        "EN": [
            "How much is delivery?",
            "What's on the menu?",
            "What time do you close?",
            "Do you have vegan options?",
            "Which payment methods do you accept?",
            "Do you deliver to {district}?",
            "How do I order without an account?",
            "Is there a minimum order amount?",
            "How much is the {dish_en}?",
            "Do you have gluten-free options?",
            "Can I pay the courier by card?",
            "What are your opening hours on weekends?",
            "Do you accept promo codes?",
            "How long does delivery usually take?",
            "Is it possible to order for tomorrow?",
        ],
        "MIX": [
            "Do you deliver до {district_ua}?",
            "Чи є vegan options?",
            "What are ваші working hours?",
        ],
    },
}


# Шум
def typo(text: str, rate: float, rng: random.Random) -> str:
    chars, out, i = list(text), [], 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "dup", "neigh"])
            if op == "swap" and i + 1 < len(chars):
                out.extend([chars[i + 1], c]); i += 2; continue
            if op == "dup":
                out.extend([c, c])
            elif op == "neigh":
                nb = (UA_NEIGH if c.lower() in UA_NEIGH else EN_NEIGH).get(c.lower())
                out.append(rng.choice(nb) if nb else c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def decorate(text: str, lang: str, rng: random.Random) -> str:
    ua = lang in ("UA", "MIX")
    if rng.random() < 0.25:
        pre = rng.choice(FILL_PRE_UA if ua else FILL_PRE_EN)
        text = f"{pre} {text}".strip()
    if rng.random() < 0.2:
        suf = rng.choice(FILL_SUF_UA if ua else FILL_SUF_EN)
        text = f"{text} {suf}".strip()
    if rng.random() < 0.3:
        text = re.sub(r"[?!.,:;]", "", text)
    if rng.random() < 0.05:
        text = f"{text} {rng.choice(EMOJI)}"
    if rng.random() < 0.04:
        text = text.upper()
    if rng.random() < 0.35:
        text = typo(text, rng.choice([0.02, 0.04, 0.07]), rng)
    return re.sub(r"\s+", " ", text).strip()


def norm_key(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t.strip().lower()))


# Основна генерація
def fill(tpl: str, g: Gen, lang: str) -> str:
    """Підставляє плейсхолдери. Для MIX частини беруться з різних мов."""
    sub = {
        "{items}": lambda: g.items("UA" if lang == "UA" else "EN"),
        "{items_ua}": lambda: g.items("UA"),
        "{items_en}": lambda: g.items("EN"),
        "{addr}": lambda: g.addr("UA" if lang == "UA" else "EN"),
        "{addr_ua}": lambda: g.addr("UA"),
        "{addr_en}": lambda: g.addr("EN"),
        "{pay}": lambda: g.pay("UA" if lang == "UA" else "EN"),
        "{pay_ua}": lambda: g.pay("UA"),
        "{time}": lambda: g.time("UA" if lang == "UA" else "EN"),
        "{no}": g.order_no,
        "{dish}": lambda: g.r.choice(g.dishes)[0],
        "{dish_en}": lambda: g.r.choice(g.dishes)[2],
        "{district}": lambda: g.r.choice(DISTRICTS_UA if lang == "UA" else DISTRICTS_EN),
        "{district_ua}": lambda: g.r.choice(DISTRICTS_UA),
    }
    for key, fn in sub.items():
        while key in tpl:
            tpl = tpl.replace(key, fn(), 1)
    return tpl


def generate(n_per_intent: int, seed: int, menu: str | None) -> list[dict]:
    rng = random.Random(seed)
    g = Gen(rng, load_menu(menu))
    rows, seen = [], set()
    langs = ["UA"] * 50 + ["EN"] * 32 + ["MIX"] * 10 + ["TRANSLIT"] * 8
    quota = {"create_order": int(n_per_intent * 1.3)}
    for intent, by_lang in T.items():
        target, tries = quota.get(intent, n_per_intent), 0
        got = 0
        while got < target and tries < target * 40:
            tries += 1
            lang = rng.choice(langs)
            base = "UA" if lang == "TRANSLIT" else lang
            tpls = by_lang[base]
            idx = rng.randrange(len(tpls))
            text = fill(tpls[idx], g, base)
            text = decorate(text, base, rng)
            if lang == "TRANSLIT":
                text = translit(text)
            key = norm_key(text)
            if key in seen or len(key) < 3:
                continue
            seen.add(key)
            rows.append({"text": text, "intent": intent, "lang": lang, "source": "generated",
                         "group": f"{intent}:{base}:{idx}"})
            got += 1

    hard = [
        ("faq", "Як замовити піцу без реєстрації?"), ("faq", "Чи можна скасувати замовлення після оплати?"),
        ("faq", "How do I cancel an order?"), ("faq", "Скільки часу займає доставка замовлення?"),
        ("faq", "Can I track my order online?"), ("faq", "Де можна відстежити замовлення?"),
        ("faq", "Чи можна замовити доставку вночі?"), ("faq", "What happens if my order is late?"),
        ("track_order", "Замовлення вже в дорозі?"), ("track_order", "Мені ще не привезли замовлення"),
        ("track_order", "Is the delivery on its way?"), ("track_order", "Order status?"),
        ("cancel_order", "Не треба доставляти замовлення"), ("cancel_order", "No need to deliver it anymore, cancel"),
        ("create_order", "Піца пепероні і борщ, Шевченка 5"), ("create_order", "2 burgers, 12 Main Street, cash"),
        ("create_order", "Те саме що минулого разу, будь ласка, на ту ж адресу"),
        ("create_order", "Same as last time please, same address"),
    ]
    for i, (intent, text) in enumerate(hard):
        key = norm_key(text)
        if key not in seen:
            seen.add(key)
            rows.append({"text": text, "intent": intent, "lang": "UA" if re.search("[а-яі]", text.lower()) else "EN",
                         "source": "hard_case", "group": f"hard:{i}"})

    curated_faq = [
        ("payment", "UA", "Якщо візьму борщ і салат, можна розрахуватися Apple Pay?"),
        ("payment", "UA", "Кур'єр має термінал чи краще підготувати готівку?"),
        ("payment", "UA", "Гроші списуються одразу під час оформлення чи вже після доставки?"),
        ("payment", "EN", "Can I split the bill between cash and a card?"),
        ("payment", "EN", "Do you charge the card when I place the order or when it arrives?"),
        ("payment", "MIX", "Чи проходить Google Pay для замовлення онлайн?"),
        ("allergens", "UA", "У піці пепероні є молоко або інші алергени?"),
        ("allergens", "UA", "Чи готуєте картоплю фрі в тій самій олії, що й страви з глютеном?"),
        ("allergens", "UA", "Де подивитися повний склад вареників, зокрема можливі сліди горіхів?"),
        ("allergens", "EN", "Is the Caesar salad safe for someone with an egg allergy?"),
        ("allergens", "EN", "Could you tell me whether the burger buns contain sesame?"),
        ("allergens", "MIX", "Does Pizza Margherita contain lactose?"),
        ("delivery", "UA", "Підкажіть, до котрої години сьогодні возите на Садгору?"),
        ("delivery", "UA", "Скільки зазвичай їде кур'єр до університетського кампусу?"),
        ("delivery", "UA", "На околиці теж доставляєте чи тільки в межах центру?"),
        ("delivery", "EN", "Would a delivery to the airport area take much longer than usual?"),
        ("delivery", "EN", "Do you still deliver to the suburbs after 9 pm?"),
        ("delivery", "EN", "Is there a separate delivery charge for the old town?"),
        ("delivery", "MIX", "А в Рогізну кур'єр їде у вихідні?"),
        ("delivery", "TRANSLIT", "do yakoyi hodyny dostavliaiete v tsentr?"),
        ("minimum", "UA", "Яку найменшу суму треба набрати, щоб кур'єр виїхав?"),
        ("minimum", "UA", "Якщо замовлю лише каву, оформлення пройде чи є мінімалка?"),
        ("minimum", "UA", "Є мінімальна вартість кошика для доставки на Шевченка?"),
        ("minimum", "EN", "Is there a minimum spend before delivery becomes available?"),
        ("minimum", "EN", "Could I order just one portion of fries, or is there a minimum?"),
        ("hours", "UA", "Ви приймаєте замовлення після опівночі у п'ятницю?"),
        ("hours", "UA", "А на Великдень кухня працюватиме за звичним графіком?"),
        ("hours", "UA", "О котрій відкриваєтесь у неділю?"),
        ("hours", "EN", "Are you open on public holidays, and when does the kitchen close?"),
        ("hours", "EN", "Can I place an order before your doors open for lunch?"),
        ("hours", "TRANSLIT", "koli vy vidkryvaietesia v nediliu?"),
        ("refunds", "UA", "Як повернути гроші, якщо оплатив онлайн, а замовлення скасували?"),
        ("refunds", "UA", "Піца приїхала холодна — куди написати щодо компенсації?"),
        ("refunds", "UA", "Скільки чекати на повернення коштів після помилкового списання?"),
        ("refunds", "UA", "Замовлення ще не привезли, але статус завис: як подати скаргу?"),
        ("refunds", "EN", "What is your refund policy if an item is missing from my order?"),
        ("refunds", "EN", "The burger arrived damaged; can I request a refund or replacement?"),
        ("hard_negative", "UA", "Чи можна оплатити замовлення карткою, якщо в кошику тільки борщ?"),
        ("hard_negative", "UA", "У вас є алергени в салаті цезарі, чи можна замовити без яєць?"),
        ("hard_negative", "UA", "Чи доставляєте піцу на Героїв Майдану і скільки це коштує?"),
        ("hard_negative", "UA", "Яка мінімальна сума замовлення на вареники?"),
        ("hard_negative", "UA", "Чи повертаєте кошти, коли замовлення запізнилося?"),
        ("hard_negative", "EN", "Can I get a refund for a cold Pizza Pepperoni, and how long does it take?"),
        ("hard_negative", "EN", "What time can I order a burger, and do you deliver to the campus?"),
        ("hard_negative", "EN", "Is the minimum order amount different if I only want coffee?"),
        ("hard_negative", "MIX", "Замовлення з French Fries можна оплатити готівкою кур'єру?"),
        ("hard_negative", "TRANSLIT", "chi ye alerheny u borshchi?"),
        ("hard_negative", "TRANSLIT", "yak povernuty hroshi za zamovlennia?"),
        ("hard_negative", "TRANSLIT", "yak zminyty adresu v zamovlenni?"),
    ]
    for topic, lang, text in curated_faq:
        key = norm_key(text)
        if key not in seen:
            seen.add(key)
            rows.append({
                "text": text,
                "intent": "faq",
                "lang": lang,
                "source": "curated",
                "group": f"faq:curated:{topic}",
            })

    curated_other = [
        ("weather", "UA", "Якою буде погода в Одесі на вихідних?"),
        ("weather", "UA", "Чи варто завтра брати парасолю?"),
        ("weather", "EN", "Will it be sunny in London this afternoon?"),
        ("weather", "EN", "What is the temperature in Toronto right now?"),
        ("jokes", "UA", "Розкажи короткий жарт про програмістів."),
        ("jokes", "UA", "Можеш придумати смішний тост?"),
        ("jokes", "EN", "Tell me a clean joke for a birthday party."),
        ("jokes", "EN", "Make up a funny one-liner about cats."),
        ("travel", "UA", "Як дістатися від вокзалу до аеропорту?"),
        ("travel", "UA", "Коли відправляється наступний потяг до Львова?"),
        ("travel", "EN", "Find me a train from Berlin to Prague."),
        ("travel", "EN", "How long is the flight to Madrid?"),
        ("math", "UA", "Скільки буде 17 помножити на 23?"),
        ("math", "UA", "Порахуй, будь ласка, 145 мінус 68."),
        ("math", "EN", "What is the square root of 256?"),
        ("math", "EN", "Convert 72 Fahrenheit to Celsius."),
        ("translation", "UA", "Переклади слово «umbrella» українською."),
        ("translation", "UA", "Як англійською сказати «дякую за допомогу»?"),
        ("translation", "EN", "Translate bonjour into English."),
        ("translation", "EN", "What does the Polish word proszę mean?"),
        ("entertainment", "UA", "Порадь детективний серіал на вечір."),
        ("entertainment", "UA", "Який альбом випустив гурт Океан Ельзи першим?"),
        ("entertainment", "EN", "Recommend a science-fiction movie from the 1990s."),
        ("entertainment", "EN", "Who directed the film Spirited Away?"),
        ("finance", "UA", "Який зараз курс євро до гривні?"),
        ("finance", "UA", "Поясни простими словами, що таке складний відсоток."),
        ("finance", "EN", "What was the price of gold yesterday?"),
        ("finance", "EN", "How do I make a monthly household budget?"),
        ("writing", "UA", "Допоможи скласти ввічливий лист викладачу."),
        ("writing", "UA", "Напиши чотири рядки про осінній ліс."),
        ("writing", "EN", "Draft a short thank-you note for my neighbor."),
        ("writing", "EN", "Suggest a title for a presentation about climate change."),
        ("knowledge", "UA", "Хто написав «Місто» Валер'яна Підмогильного?"),
        ("knowledge", "UA", "Яка столиця Португалії?"),
        ("knowledge", "EN", "Why is the sky blue?"),
        ("knowledge", "EN", "Who was the first person to walk on the Moon?"),
        ("account", "UA", "Як змінити пароль у пошті?"),
        ("account", "UA", "Не можу увійти у свій банківський застосунок, що робити?"),
        ("account", "EN", "How do I recover access to my social media account?"),
        ("account", "EN", "Can you help me set up two-factor authentication?"),
        ("devices", "UA", "Як під'єднати бездротові навушники до ноутбука?"),
        ("devices", "UA", "Чому телефон швидко розряджається?"),
        ("devices", "EN", "My printer is offline; how can I reconnect it?"),
        ("devices", "EN", "Which laptop is good for photo editing?"),
        ("sports", "UA", "Хто виграв учорашній матч збірної України?"),
        ("sports", "UA", "Поясни правила офсайду у футболі."),
        ("sports", "EN", "When is the next Formula One race?"),
        ("sports", "EN", "Who holds the women's marathon world record?"),
        ("account", "UA", "Як відновити доступ до електронної пошти, якщо забув пароль?"),
        ("account", "EN", "Could you help me change the password for my email account?"),
        ("account", "EN", "I cannot sign in to my bank app; how do I unlock it?"),
        ("media", "UA", "Увімкни спокійну музику для роботи."),
        ("media", "EN", "Play a relaxing jazz playlist while I work."),
        ("media", "EN", "Can you recommend a podcast about history?"),
        ("shopping", "UA", "У якому магазині продають недорогі ноутбуки?"),
        ("shopping", "EN", "Where can I compare prices for a new laptop?"),
        ("shopping", "EN", "Find a store that sells wireless headphones."),
        ("time_tools", "UA", "Нагадай мені зателефонувати о сьомій ранку."),
        ("time_tools", "EN", "Set a reminder for my dentist appointment tomorrow."),
        ("travel", "UA", "Де купити квиток на потяг до Києва?"),
        ("travel", "UA", "Скільки коштує проїзд автобусом до аеропорту?"),
        ("travel", "EN", "Where can I book a flight to Warsaw?"),
        ("food_general", "UA", "Як приготувати борщ удома з нуля?"),
        ("food_general", "UA", "Хто першим придумав суші?"),
        ("food_general", "EN", "How do I make pizza dough at home?"),
        ("food_general", "EN", "Translate the word burger into French."),
        ("food_general", "EN", "What is the history of pad thai?"),
        ("writing", "EN", "Write a short poem about a rainy afternoon."),
        ("knowledge", "UA", "Чому листя змінює колір восени?"),
        ("sports", "EN", "What time does the tennis final start tomorrow?"),
    ]
    for topic, lang, text in curated_other:
        key = norm_key(text)
        if key not in seen:
            seen.add(key)
            rows.append({
                "text": text,
                "intent": "other",
                "lang": lang,
                "source": "curated",
                "group": f"other:curated:{topic}",
            })

    curated_translit_tracking = [
        "de moie zamovlennia?",
        "koly pryide kurier?",
        "perevir status zamovlennia 7281",
        "kuryer vzhe vyikhav?",
        "skilky shche chekati na dostavku?",
        "mii zakaz SH-8421 de?",
    ]
    for text in curated_translit_tracking:
        key = norm_key(text)
        if key not in seen:
            seen.add(key)
            rows.append({
                "text": text,
                "intent": "track_order",
                "lang": "TRANSLIT",
                "source": "curated",
                "group": "track_order:curated:translit",
            })
    return rows


def merge_existing(rows: list[dict], path: str) -> list[dict]:
    df = pd.read_csv(path)
    seen = {norm_key(r["text"]) for r in rows}
    added = []
    for t, y in zip(df["text"].astype(str), df["intent"].astype(str)):
        key = norm_key(t)
        if key in seen:
            continue
        seen.add(key)
        toks = key.split()
        has_cyr, has_lat = bool(re.search("[а-яіїєґ]", key)), bool(re.search("[a-z]", key))
        lang = "MIX" if has_cyr and has_lat else ("UA" if has_cyr else "EN")
        group = f"orig:{y}:{' '.join(toks[:2])}"
        added.append({"text": t, "intent": y, "lang": lang, "source": "original", "group": group})
    return rows + added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default=None, help="ваш intents_merged.csv")
    ap.add_argument("--menu", default=None, help="data/menu.json (реальні назви страв)")
    ap.add_argument("--n-per-intent", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/intents_augmented.csv")
    args = ap.parse_args()

    rows = generate(args.n_per_intent, args.seed, args.menu)
    if args.existing:
        rows = merge_existing(rows, args.existing)
    random.Random(args.seed).shuffle(rows)

    df = pd.DataFrame(rows, columns=["text", "intent", "lang", "source", "group"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Збережено {len(df)} рядків у {args.out}")
    print("\nЗа інтентами:\n", df["intent"].value_counts().to_string())
    print("\nЗа мовами:\n", df["lang"].value_counts().to_string())
    print("\nЗа джерелом:\n", df["source"].value_counts().to_string())
    print(f"\nУнікальних груп (шаблонів): {df['group'].nunique()}")


if __name__ == "__main__":
    main()
