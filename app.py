import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline import FoodOrderNLP

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="Shill", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_nlp: FoodOrderNLP | None = None


def get_nlp() -> FoodOrderNLP:
    global _nlp
    if _nlp is None:
        _nlp = FoodOrderNLP(ROOT)
    return _nlp


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)





EXAMPLES = [
    {"label": "Нове замовлення", "text": "Два салати, один борщ та 4 сири на вул. Шевченка 12, карткою"},
    {"label": "Англійською", "text": "Two burgers and 1 fries to 12 Groove Street, cash"},
    {"label": "Статус", "text": "Де моє замовлення, коли привезуть?"},
    {"label": "Скасувати", "text": "Скасуйте замовлення, будь ласка"},
    {"label": "Питання", "text": "Що є в меню і скільки коштує доставка?"},
]

MENU_PRICES = {
    1: 89,
    2: 75,
    4: 110,
    5: 145,
    6: 65,
    7: 55,
    8: 45,
    9: 160,
    10: 135,
    11: 70,
    12: 220,
    13: 125,
    14: 140,
    15: 155,
    16: 95,
    17: 50,
    18: 40,
    19: 150,
    20: 85,
    21: 60,
    22: 120,
    23: 70,
    24: 140,
    25: 95,
    26: 110,
    27: 80,
    28: 65,
    29: 249,
    30: 229,
    31: 239,
    32: 259,
    33: 219,
}

ACTIVE_DELIVERIES = [
    {
        "id": "SH-1041",
        "customer": "Олена К.",
        "address": "вул. Шевченка 12",
        "items": "2× Борщ, Піца",
        "status": "в дорозі",
        "courier": "Ігор М.",
        "eta": "18:40",
    },
    {
        "id": "SH-1040",
        "customer": "Andrii P.",
        "address": "12 Main Street",
        "items": "Burger, Fries",
        "status": "готується",
        "courier": "—",
        "eta": "19:05",
    },
    {
        "id": "SH-1039",
        "customer": "Марія С.",
        "address": "пр. Перемоги 5",
        "items": "Салат, Кава",
        "status": "очікує кур'єра",
        "courier": "Петро Л.",
        "eta": "18:55",
    },
]

INTENT_UI = {
    "create_order": {"title": "Нове замовлення", "color": "#16a34a"},
    "track_order": {"title": "Статус доставки", "color": "#2563eb"},
    "cancel_order": {"title": "Скасування", "color": "#dc2626"},
    "faq": {"title": "Питання клієнта", "color": "#7c3aed"},
    "other": {"title": "Потрібне уточнення", "color": "#64748b"},
}


ORDER_ID_PATTERN = re.compile(
    r"(?:SH[- ]?\d+|замовл(?:ення|енням)?(?:\s*№?\s*)?#?\d+|\b№?\s*\d{3,4}\b)",
    re.I | re.UNICODE,
)


def extract_order_id(text: str) -> str | None:
    match = ORDER_ID_PATTERN.search(text)
    if not match:
        return None
    value = match.group(0)
    sh_match = re.search(r"SH[- ]?\d+", value, re.I)
    if sh_match:
        return sh_match.group(0).replace(" ", "-").upper()
    digits = re.findall(r"\d+", value)
    if not digits:
        return None
    candidate = digits[-1]
    for delivery in ACTIVE_DELIVERIES:
        if delivery["id"].endswith(candidate):
            return delivery["id"]
    return candidate


def find_delivery(order_id: str | None) -> dict | None:
    if not order_id:
        return None
    normalized = order_id.upper().replace(" ", "-")
    for delivery in ACTIVE_DELIVERIES:
        if delivery["id"].upper() == normalized:
            return delivery
    return None


def build_track_result(text: str) -> tuple[str, dict]:
    order_id = extract_order_id(text)
    delivery = find_delivery(order_id)
    if not order_id:
        message = "У повідомленні немає номера замовлення. Уточніть, будь ласка, номер замовлення."
        return message, {"order_id": None, "delivery": None, "needs_order_id": True}
    if not delivery:
        message = (
            f"Не вдалося знайти активне замовлення №{order_id}. "
            "Перевірте номер і уточніть його у клієнта."
        )
        return message, {"order_id": order_id, "delivery": None, "needs_order_id": False}
    message = (
        f"Замовлення {delivery['id']} зараз має статус «{delivery['status']}». "
        f"Очікуваний час: {delivery['eta']}."
    )
    return message, {"order_id": delivery["id"], "delivery": delivery, "needs_order_id": False}


def build_cancel_result(text: str) -> tuple[str, dict]:
    order_id = extract_order_id(text)
    delivery = find_delivery(order_id)
    if not order_id:
        message = "Уточніть, будь ласка, номер замовлення, яке потрібно скасувати."
        return message, {"order_id": None, "delivery": None, "needs_order_id": True}
    if not delivery:
        message = (
            f"Активне замовлення №{order_id} не знайдено. "
            "Перевірте номер перед скасуванням."
        )
        return message, {"order_id": order_id, "delivery": None, "needs_order_id": False}
    message = f"Підтвердити скасування активного замовлення {delivery['id']}?"
    return message, {"order_id": delivery["id"], "delivery": delivery, "needs_confirmation": True}


def build_faq_result(text: str) -> tuple[str, dict]:
    lowered = text.lower()
    topic = "general"
    if any(word in lowered for word in ("доставк", "delivery", "кур'єр", "courier", "привез")):
        topic = "delivery"
        message = "Середній час доставки становить 35–60 хвилин."
    elif any(word in lowered for word in ("меню", "страв", "dish", "menu", "їжа")):
        topic = "menu"
        message = "Ознайомитися зі списком доступних страв можна у розділі «Меню»."
    elif any(word in lowered for word in ("оплат", "карт", "готів", "cash", "card")):
        topic = "payment"
        message = "Доступна оплата карткою та готівкою кур'єру."
    elif any(word in lowered for word in ("скільки кошту", "ціна", "price", "cost")):
        topic = "price"
        message = "Точну ціну страви можна перевірити у розділі «Меню»."
    elif any(word in lowered for word in ("працю", "відкрит", "годин", "open", "hours")):
        topic = "hours"
        message = "Час роботи уточнюється оператором."
    else:
        message = "Давайте уточнимо, яке саме питання вас цікавить."
    return message, {"topic": topic}


@app.on_event("startup")
def startup() -> None:
    get_nlp()


@app.get("/api/health")
def health() -> dict:
    return {"ok": (ROOT / "models" / "intent_model.joblib").exists()}


@app.get("/api/examples")
def examples() -> list[dict]:
    return EXAMPLES


@app.get("/api/menu")
def menu() -> list[dict]:
    path = ROOT / "data" / "menu.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "price": MENU_PRICES.get(item["id"], 99),
            "available": True,
        }
        for item in items
    ]


@app.get("/api/deliveries")
def deliveries() -> list[dict]:
    return ACTIVE_DELIVERIES


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest) -> dict:
    result = get_nlp().process(body.text)
    meta = INTENT_UI.get(result["intent"], {"title": "Інше", "color": "#64748b"})
    result["intent_title"] = meta["title"]
    result["intent_color"] = meta["color"]

    if result["intent"] == "create_order" and result.get("slots"):
        dishes = result["slots"].get("dishes") or []
        total_qty = sum(d["quantity"] for d in dishes)
        if dishes:
            result["summary"] = f"Розпізнано {len(dishes)} позицій на {total_qty} порцій. Перевірте адресу та оплату."
        else:
            result["summary"] = "Замовлення без позицій у меню — потрібне уточнення."
    elif result["intent"] == "track_order":
        result["summary"] = "Клієнт цікавиться, де замовлення або коли буде доставка."
        response_text, action = build_track_result(body.text)
        result["response_text"] = response_text
        result["action"] = {"type": "track", **action}
    elif result["intent"] == "cancel_order":
        result["summary"] = "Клієнт просить скасувати замовлення."
        response_text, action = build_cancel_result(body.text)
        result["response_text"] = response_text
        result["action"] = {"type": "cancel", **action}
    elif result["intent"] == "faq":
        response_text, action = build_faq_result(body.text)
        result["summary"] = "Загальне питання щодо сервісу або меню."
        result["response_text"] = response_text
        result["action"] = {"type": "faq", **action}
    elif result["intent"] == "other":
        result["summary"] = "Не вдалося впевнено визначити тему. Уточніть, будь ласка, запит."
    else:
        result["summary"] = "Загальне питання щодо сервісу або меню."

    return result


@app.post("/api/cancel/{order_id}")
def cancel_delivery(order_id: str) -> dict:
    normalized = order_id.upper().replace(" ", "-")
    for index, delivery in enumerate(ACTIVE_DELIVERIES):
        if delivery["id"].upper() == normalized:
            cancelled = ACTIVE_DELIVERIES.pop(index)
            return {
                "ok": True,
                "order_id": cancelled["id"],
                "message": f"Замовлення {cancelled['id']} скасовано.",
            }
    return {"ok": False, "order_id": order_id, "message": "Активне замовлення не знайдено."}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
