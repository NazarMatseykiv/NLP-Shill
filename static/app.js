const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const PAYMENT_LABELS = {
  card: "Картка",
  cash: "Готівка",
  online: "Онлайн",
};

const PAGES = {
  orders: {
    title: "Прийом замовлень",
    lead: "Введіть повідомлення клієнта — система розпізнає тип звернення та склад замовлення",
  },
  deliveries: {
    title: "Активні доставки",
    lead: "Поточні замовлення в обробці та доставці",
  },
  menu: {
    title: "Меню",
    lead: "Каталог страв, доступних для замовлення",
  },
};

const STORAGE_KEY = "shill_local_deliveries";
let pendingOrder = null;
let menuItems = [];

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Помилка з'єднання");
  }
  return res.json();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function updateClock() {
  $("#clock").textContent = new Date().toLocaleString("uk-UA", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusClass(status) {
  const s = status.toLowerCase();
  if (s.includes("дороз") || s.includes("дорозі")) return "transit";
  if (s.includes("готу")) return "cooking";
  if (s.includes("очіку")) return "waiting";
  return "new";
}

function getLocalDeliveries() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveLocalDelivery(order) {
  const list = getLocalDeliveries();
  list.unshift(order);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list.slice(0, 20)));
}

function switchPage(page) {
  $$(".nav-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.page === page);
  });
  $$(".page-view").forEach((el) => {
    el.classList.add("hidden");
  });
  $(`#page-${page}`).classList.remove("hidden");

  const meta = PAGES[page];
  $("#page-title").textContent = meta.title;
  $("#page-lead").textContent = meta.lead;

  if (page === "deliveries") loadDeliveries();
  if (page === "menu") renderMenu(menuItems, $("#menu-search").value);
}

function renderMenu(items, query = "") {
  const q = query.trim().toLowerCase();
  const filtered = q
    ? items.filter((i) => i.name.toLowerCase().includes(q))
    : items;
  const tbody = $("#menu-body");
  tbody.innerHTML = "";
  for (const item of filtered) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.id}</td>
      <td>${escapeHtml(item.name)}</td>
      <td class="price">${item.price} ₴</td>
      <td><span class="badge-ok">${item.available ? "В наявності" : "Немає"}</span></td>`;
    tbody.appendChild(tr);
  }
}

async function loadDeliveries() {
  const server = await api("/api/deliveries");
  const local = getLocalDeliveries();
  const all = [...local, ...server];

  $("#deliveries-count").textContent = `${all.length} замовлень`;
  const tbody = $("#deliveries-body");
  const empty = $("#deliveries-empty");
  tbody.innerHTML = "";

  if (!all.length) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");

  for (const d of all) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(d.id)}</strong></td>
      <td>${escapeHtml(d.customer)}</td>
      <td>${escapeHtml(d.address)}</td>
      <td>${escapeHtml(d.items)}</td>
      <td><span class="status-tag ${statusClass(d.status)}">${escapeHtml(d.status)}</span></td>
      <td>${escapeHtml(d.courier)}</td>
      <td>${escapeHtml(d.eta)}</td>`;
    tbody.appendChild(tr);
  }
}

function renderResult(data) {
  pendingOrder = null;
  $("#confirm-order").classList.add("hidden");

  $("#empty").classList.add("hidden");
  $("#result").classList.remove("hidden");

  const badge = $("#intent-badge");
  badge.textContent = data.intent_title || "—";
  badge.style.background = data.intent_color || "#64748b";

  const conf = Math.round((data.intent_confidence || 0) * 100);
  $("#confidence").textContent = `Впевненість моделі: ${conf}%`;
  $("#conf-bar").style.width = `${conf}%`;
  $("#summary").textContent = data.summary || "";

  const orderBlock = $("#order-block");
  const noticeBlock = $("#notice-block");
  const meta = $("#meta-grid");
  const tbody = $("#order-body");

  orderBlock.classList.add("hidden");
  noticeBlock.classList.add("hidden");
  meta.innerHTML = "";
  tbody.innerHTML = "";

  if (data.intent === "create_order" && data.slots) {
    orderBlock.classList.remove("hidden");
    const s = data.slots;
    const payment = PAYMENT_LABELS[s.payment] || s.payment || "—";

    const fields = [
      ["Адреса доставки", s.address || "—"],
      ["Спосіб оплати", payment],
      ["Коментар", s.comment || "—"],
      ["Час", s.time_hint || "—"],
    ];

    for (const [label, value] of fields) {
      const el = document.createElement("div");
      el.className = "info-item";
      el.innerHTML = `<span>${label}</span><strong>${escapeHtml(value)}</strong>`;
      meta.appendChild(el);
    }

    let total = 0;
    const itemParts = [];
    for (const d of s.dishes || []) {
      total += d.quantity;
      itemParts.push(`${d.quantity}× ${d.dish_name}`);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(d.dish_name)}</td>
        <td>${d.quantity}</td>
        <td>${escapeHtml(d.raw || "—")}</td>`;
      tbody.appendChild(tr);
    }

    $("#order-total").textContent = String(total);

    if (!s.dishes?.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="3" style="color:var(--muted)">Позиції не знайдено — уточніть у клієнта</td>`;
      tbody.appendChild(tr);
    } else if (s.address) {
      pendingOrder = {
        address: s.address,
        items: itemParts.join(", "),
        payment,
      };
      $("#confirm-order").classList.remove("hidden");
    }
  } else {
    noticeBlock.classList.remove("hidden");
    const responseText = data.response_text || "Обробіть звернення вручну за стандартною процедурою.";
    const action = data.action || {};
    let actionButtons = `
      <button type="button" class="btn secondary action-copy">Скопіювати відповідь</button>
    `;

    if (action.type === "cancel" && action.delivery && action.needs_confirmation) {
      actionButtons += `
        <button type="button" class="btn danger action-cancel">Скасувати замовлення</button>
      `;
    }
    if (action.type === "track" && action.delivery) {
      actionButtons += `
        <button type="button" class="btn primary action-open-deliveries">Відкрити доставку</button>
      `;
    }

    noticeBlock.innerHTML = `
      <strong>Готова відповідь клієнту</strong>
      <textarea class="action-response" rows="3">${escapeHtml(responseText)}</textarea>
      <div class="action-buttons">${actionButtons}</div>
      <p class="action-status"></p>
    `;

    const copyButton = noticeBlock.querySelector(".action-copy");
    copyButton.addEventListener("click", async () => {
      const text = noticeBlock.querySelector(".action-response").value;
      try {
        await navigator.clipboard.writeText(text);
        noticeBlock.querySelector(".action-status").textContent = "Відповідь скопійовано.";
      } catch {
        noticeBlock.querySelector(".action-status").textContent =
          "Не вдалося скопіювати. Виділіть текст вручну.";
      }
    });

    const cancelButton = noticeBlock.querySelector(".action-cancel");
    if (cancelButton) {
      cancelButton.addEventListener("click", async () => {
        try {
          const result = await api(`/api/cancel/${encodeURIComponent(action.order_id)}`, {
            method: "POST",
          });
          noticeBlock.querySelector(".action-status").textContent = result.message;
          if (result.ok) {
            cancelButton.disabled = true;
            cancelButton.textContent = "Скасовано";
          }
        } catch (error) {
          noticeBlock.querySelector(".action-status").textContent = error.message;
        }
      });
    }

    const deliveriesButton = noticeBlock.querySelector(".action-open-deliveries");
    if (deliveriesButton) {
      deliveriesButton.addEventListener("click", () => switchPage("deliveries"));
    }
  }

  const { intent_title, intent_color, summary, ...raw } = data;
  $("#json-out").textContent = JSON.stringify(raw, null, 2);
}

function confirmPendingOrder() {
  if (!pendingOrder) return;
  const num = 1042 + getLocalDeliveries().length;
  const now = new Date();
  const eta = new Date(now.getTime() + 35 * 60_000);
  const order = {
    id: `SH-${num}`,
    customer: "Новий клієнт",
    address: pendingOrder.address,
    items: pendingOrder.items,
    status: "очікує кур'єра",
    courier: "—",
    eta: eta.toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }),
  };
  saveLocalDelivery(order);
  pendingOrder = null;
  $("#confirm-order").classList.add("hidden");
  alert(`Замовлення ${order.id} додано в активні доставки.`);
}

async function analyze() {
  const text = $("#input").value.trim();
  if (!text) return;

  const btn = $("#analyze");
  btn.disabled = true;
  btn.textContent = "Обробка…";

  try {
    const data = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    renderResult(data);
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Обробити";
  }
}

async function init() {
  updateClock();
  setInterval(updateClock, 30_000);

  $$(".nav-item").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchPage(link.dataset.page);
    });
  });

  try {
    const health = await api("/api/health");
    const el = $("#health");
    if (health.ok) {
      el.textContent = "Сервіс активний";
      el.classList.add("ok");
    } else {
      el.textContent = "Сервіс недоступний";
      el.classList.add("bad");
    }
  } catch {
    $("#health").textContent = "Немає з'єднання";
    $("#health").classList.add("bad");
  }

  menuItems = await api("/api/menu");
  $("#menu-search").addEventListener("input", (e) => {
    renderMenu(menuItems, e.target.value);
  });

  const examples = await api("/api/examples");
  const chips = $("#chips");
  for (const ex of examples) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.title = ex.text;
    btn.textContent = ex.label;
    btn.addEventListener("click", () => {
      $("#input").value = ex.text;
      switchPage("orders");
      analyze();
    });
    chips.appendChild(btn);
  }

  $("#analyze").addEventListener("click", analyze);
  $("#confirm-order").addEventListener("click", confirmPendingOrder);
  $("#clear").addEventListener("click", () => {
    $("#input").value = "";
    $("#result").classList.add("hidden");
    $("#empty").classList.remove("hidden");
    pendingOrder = null;
    $("#confirm-order").classList.add("hidden");
  });
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
  });
}

init();
