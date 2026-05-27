import json
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
    {"label": "Нове замовлення", "text": "Два салати та 3 піци на вул. Шевченка 12, карткою"},
    {"label": "Англійською", "text": "Two burgers and 1 fries to 12 Groove Street, cash"},
    {"label": "Статус", "text": "Де моє замовлення, коли привезуть?"},
    {"label": "Скасувати", "text": "Скасуйте замовлення, будь ласка"},
    {"label": "Питання", "text": "Що є в меню і скільки коштує доставка?"},
]

MENU_PRICES = {
    1: 89,
    2: 75,
    3: 199,
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
}


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
    elif result["intent"] == "cancel_order":
        result["summary"] = "Клієнт просить скасувати замовлення."
    else:
        result["summary"] = "Загальне питання щодо сервісу або меню."

    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
