import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

HISTORY_JSON = os.path.join(OUTPUT_DIR, "filtered_pairs_history.json")
POSITIONS_JSON = os.path.join(OUTPUT_DIR, "positions_state.json")  # локальный журнал открытых/закрытых

def ensure_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- история сканера ----------

def append_history(payload: Dict[str, Any]):
    ensure_dirs()
    record = {"ts": now_iso(), **payload}
    if not os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump([record], f, ensure_ascii=False, indent=2)
        return
    try:
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    arr.append(record)
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# ---------- локальное состояние позиций ----------

def positions_load() -> Dict[str, Any]:
    ensure_dirs()
    if not os.path.exists(POSITIONS_JSON):
        state = {"open": {}, "last_closed": {}}
        positions_save(state)
        return state
    with open(POSITIONS_JSON, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"open": {}, "last_closed": {}}

def positions_save(state: Dict[str, Any]):
    ensure_dirs()
    with open(POSITIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def positions_mark_opened(state: Dict[str, Any], key: str, symbol: str, side: str, qty: float, entry: float, tp: float, sl: float, order_id: str):
    state["open"][key] = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "order_id": order_id,
        "opened_at": now_iso(),
    }

def positions_mark_closed(state: Dict[str, Any], key: str):
    rec = state["open"].pop(key, None)
    if rec:
        state["last_closed"][rec["symbol"]] = now_iso()

def positions_sync_with_exchange(state: Dict[str, Any], exch_positions: List[Dict[str, Any]]) -> List[str]:
    """
    Сопоставляем открытые позиции из локального state с тем, что вернула биржа.
    Если позиции в локальном списке нет на бирже — считаем закрытой (TP/SL/ручное).
    Возвращает список ключей, которые были закрыты.
    """
    existing_keys = set(state["open"].keys())
    exch_keys = set()

    for p in exch_positions:
        try:
            sym = p.get("symbol")
            side = p.get("side")  # "Buy"/"Sell"
            k = f"{sym}:{'LONG' if side == 'Buy' else 'SHORT'}"
            exch_keys.add(k)
            # можем дополнять entry или qty, если хотим
        except Exception:
            pass

    closed = []
    for k in list(existing_keys):
        if k not in exch_keys:
            positions_mark_closed(state, k)
            closed.append(k)
    return closed
