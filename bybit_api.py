# bybit_api.py
# Полноценный клиент для Bybit V5 (unified), с корректной подписью Sign-Type=2
# - Поддержка mainnet (headers-sign) и demo (params-sign)
# - Безопасные ретраи, паузы между запросами
# - Утилиты для округления цены/кол-ва по спецификациям инструмента
# - Ордера (market), установка TP/SL, «мягкие» операции для PnL-чекера
# - Мягкий get_open_positions: при retCode!=0 не падает, а логирует и возвращает []

import time
import logging
import math
import hmac
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from urllib.parse import urlencode

DEFAULT_BYBIT_BASE = "https://api.bybit.com"  # V5 unified


def _hmac_sha256(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _normalize_value_for_sign(v: Any) -> str:
    # Bybit ожидает строки без URL-эскейпа, булевы как 'true'/'false'
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def _build_query_string_for_sign(params: Dict[str, Any]) -> str:
    # Строго алфавитная сортировка ключей, значения НЕ URL-кодируем
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    items = [(k, _normalize_value_for_sign(clean[k])) for k in sorted(clean.keys())]
    return "&".join([f"{k}={v}" for k, v in items])


class BybitAPI:
    """
    Поддержка mainnet/demo:
      sign_style: "headers" (Sign-Type=2) или "params" (api_key,timestamp,sign).
      position_mode: "one_way" | "hedge" (даёт правильный positionIdx).
    """

    def __init__(
        self,
        category: str = "linear",
        sleep_ms: int = 250,
        max_retries: int = 3,
        retry_backoff_sec: int = 2,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        recv_window: int = 60000,
        base_url: Optional[str] = None,
        sign_style: str = "headers",
        position_mode: str = "one_way",
    ):
        self.category = category
        self.sleep_ms = sleep_ms
        self.session = requests.Session()
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.recv_window = recv_window
        self.base_url = (base_url or DEFAULT_BYBIT_BASE).rstrip("/")
        self.sign_style = (sign_style or "headers").strip().lower()
        self.position_mode = (position_mode or "one_way").strip().lower()
        self._detected_mode: Optional[str] = None

    # ───────── utils/sign ─────────
    def _sleep(self):
        time.sleep(self.sleep_ms / 1000.0)

    def _req_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error retCode={data.get('retCode')} retMsg={data.get('retMsg')} data={data}")
        return data

    def _headers_private(self, method: str, query_params: Dict[str, Any], body_str: Optional[str]) -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        method_up = method.upper()
        if method_up == "GET":
            payload = _build_query_string_for_sign(query_params or {})
        else:
            payload = body_str or "{}"  # уже сериализованная строка JSON
        pre_sign = ts + self.api_key + str(self.recv_window) + payload
        sign = _hmac_sha256(self.api_secret, pre_sign)
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
        }

    def _params_signed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Стиль подписи через параметры (чаще нужен на demo).
        Алфавитная сортировка ключей; pre_sign — строка вида a=b&c=d ...
        """
        p = {k: v for k, v in (params or {}).items() if v is not None}
        p["api_key"] = self.api_key
        p["timestamp"] = str(int(time.time() * 1000))
        p["recv_window"] = str(self.recv_window)
        items = [(k, _normalize_value_for_sign(p[k])) for k in sorted(p.keys())]
        pre_sign = "&".join([f"{k}={v}" for k, v in items])
        p["sign"] = _hmac_sha256(self.api_secret, pre_sign)
        return p

    def _send(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, private: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        method_up = method.upper()
        params = params or {}

        # ── demo-режим: подпись через параметры запроса ──
        if self.sign_style == "params":
            signed = self._params_signed(params)
            if method_up == "GET":
                r = self.session.get(url, params=signed, timeout=20)
            else:
                r = self.session.post(url, json=signed, timeout=20)
            return r.json()

        # ── mainnet: Sign-Type=2 в заголовках ──
        if not private:
            if method_up == "GET":
                r = self.session.get(url, params=params, timeout=20)
            else:
                r = self.session.post(url, json=params, timeout=20)
            return r.json()

        if method_up == "GET":
            # Для подписи формируем строку параметров БЕЗ URL-эскейпа
            headers = self._headers_private("GET", params, body_str=None)
            r = self.session.get(url, params=params, headers=headers, timeout=20)
            return r.json()

        # POST/PUT/DELETE — сериализуем тело один раз и используем его же при подписи и отправке
        body_clean = {k: v for k, v in (params or {}).items() if v is not None}
        body_str = json.dumps(body_clean, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
        headers = self._headers_private(method_up, query_params={}, body_str=body_str)
        r = self.session.post(url, headers=headers, data=body_str, timeout=20)
        return r.json()

    # ───────── Public ─────────
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_instruments(self) -> List[Dict[str, Any]]:
        params = {"category": self.category, "limit": 1000}
        out: List[Dict[str, Any]] = []
        cursor = None
        while True:
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            data = self._send("GET", "/v5/market/instruments-info", p, private=False)
            self._req_check(data)
            result = data.get("result", {}) or {}
            for it in result.get("list", []) or []:
                status = str(it.get("status", "")).lower()
                symbol = it.get("symbol", "") or ""
                quote = (it.get("quoteCoin") or it.get("quoteSymbol") or "").upper()
                settle = (it.get("settleCoin") or "").upper()
                contract_type = str(it.get("contractType", "")).lower()
                is_usdt = ("USDT" in quote) or ("USDT" in settle) or symbol.endswith("USDT")
                is_trading = ("trading" in status)
                is_linear_perp = True if self.category != "linear" or not contract_type else ("perpetual" in contract_type)
                if is_usdt and is_trading and is_linear_perp:
                    out.append(it)
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
            self._sleep()
        logging.info(f"Найдено {len(out)} торгуемых USDT-перпетуалов.")
        self._sleep()
        return out

    def build_symbol_info_map(self, instruments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        info = {}
        for it in instruments:
            sym = it.get("symbol")
            pf = it.get("priceFilter", {}) or {}
            lf = it.get("lotSizeFilter", {}) or {}
            info[sym] = {
                "tickSize": float(pf.get("tickSize", 0.01)),
                "minOrderQty": float(lf.get("minOrderQty", 0.0)),
                "qtyStep": float(lf.get("qtyStep", 0.001)),
                "maxOrderQty": float(lf.get("maxOrderQty", 0.0)),
            }
        return info

    def round_price(self, symbol: str, price: float, info: Dict[str, Any]) -> float:
        tick = float(info.get("tickSize", 0.01)) if info else 0.01
        if tick <= 0:
            return float(f"{price:.6f}")
        return math.floor(price / tick) * tick

    def clamp_price_safe(self, price: float, info: Dict[str, Any]) -> float:
        tick = float(info.get("tickSize", 0.01)) if info else 0.01
        return max(tick, self.round_price("", price, info))

    def round_qty(self, symbol: str, qty: float, info: Dict[str, Any]) -> float:
        """Квантуем по qtyStep и ограничиваем min/maxOrderQty."""
        step = float(info.get("qtyStep", 0.001)) if info else 0.001
        min_q = float(info.get("minOrderQty", 0.0)) if info else 0.0
        max_q = float(info.get("maxOrderQty", 0.0)) if info else 0.0
        if step <= 0:
            step = 0.001
        q = math.floor(qty / step) * step
        if max_q > 0 and q > max_q:
            q = math.floor(max_q / step) * step
        if q < min_q:
            return 0.0
        return float(f"{q:.10f}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_tickers(self) -> List[Dict[str, Any]]:
        data = self._send("GET", "/v5/market/tickers", {"category": self.category}, private=False)
        self._req_check(data)
        self._sleep()
        return data.get("result", {}).get("list", []) or []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict[str, Any]]:
        params = {"category": self.category, "symbol": symbol, "interval": interval, "limit": limit}
        data = self._send("GET", "/v5/market/kline", params, private=False)
        self._req_check(data)
        lst = data.get("result", {}).get("list", []) or []
        lst_sorted = sorted(lst, key=lambda x: int(x[0]))
        self._sleep()
        # list format: [start, open, high, low, close, volume, turnover]
        out = []
        for x in lst_sorted:
            out.append({
                "ts": int(x[0]),
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5]),
                "turnover": float(x[6]) if len(x) > 6 and x[6] not in (None, "") else math.nan
            })
        return out

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_open_interest(self, symbol: str, interval: str = "1h") -> Optional[float]:
        params = {"category": self.category, "symbol": symbol, "interval": interval, "limit": 1}
        data = self._send("GET", "/v5/market/open-interest", params, private=False)
        self._req_check(data)
        self._sleep()
        lst = data.get("result", {}).get("list", []) or []
        if not lst:
            return None
        try:
            return float(lst[-1].get("openInterest"))
        except Exception:
            return None

    # ───────── Private ─────────
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_wallet_balance(self, account_type: Optional[str] = None) -> Dict[str, Any]:
        last_exc = None
        for t in [account_type or "UNIFIED", "CONTRACT"]:
            try:
                data = self._send("GET", "/v5/account/wallet-balance", {"accountType": t, "coin": "USDT"}, private=True)
                if str(data.get("retCode")) == "0":
                    return data.get("result", {}) or {}
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("wallet-balance failed")

    def get_available_usdt(self, account_type_env: Optional[str] = None) -> float:
        res = self.get_wallet_balance(account_type_env or "UNIFIED")
        try:
            lst = res.get("list", [])
            if not lst:
                return 0.0
            for c in lst[0].get("coin", []):
                if (c.get("coin") or "").upper() == "USDT":
                    # availableToWithdraw чаще всего соответствует доступным средствам
                    return float(c.get("availableToWithdraw") or c.get("walletBalance") or 0.0)
        except Exception:
            pass
        return 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает только активные позиции (size > 0).
        На mainnet сначала пробуем БЕЗ settleCoin, затем с ним (чтобы не ловить 10004).
        На demo — наоборот.
        """
        base = {"category": self.category, "limit": 200}
        is_mainnet = "api.bybit.com" in (self.base_url or "")

        def _try(params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            data = self._send("GET", "/v5/position/list", params, private=True)
            rc = int(data.get("retCode", -1))
            if rc != 0:
                logging.warning(f"[get_open_positions] retCode={rc} retMsg={data.get('retMsg')} params={params}")
                return None
            lst = (data.get("result") or {}).get("list") or []
            return [p for p in lst if float(p.get("size") or 0) > 0]

        # Формируем порядки попыток под mainnet/demo
        tries: List[Dict[str, Any]] = []
        if symbol:
            a = {**base, "symbol": symbol}
            b = {**base, "symbol": symbol, "settleCoin": settle_coin or "USDT"}
            tries = ([a, b] if is_mainnet else [b, a])
        else:
            a = {**base}
            b = {**base, "settleCoin": settle_coin or "USDT"}
            tries = ([a, b] if is_mainnet else [b, a])

        for params in tries:
            try:
                res = _try(params)
                if res is not None:
                    return res
            except Exception as e:
                logging.warning(f"[get_open_positions] exception with params={params}: {e}")
                continue
        return []


    def count_open_positions(self) -> int:
        lst = self.get_open_positions(symbol=None, settle_coin="USDT")
        return len(lst)

    def has_open_position(self, symbol: str) -> bool:
        try:
            return len(self.get_open_positions(symbol)) > 0
        except Exception as e:
            logging.warning(f"has_open_position failed for {symbol}: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def set_leverage(self, symbol: str, buy_leverage: int, sell_leverage: int):
        body = {"category": self.category, "symbol": symbol, "buyLeverage": str(buy_leverage), "sellLeverage": str(sell_leverage)}
        data = self._send("POST", "/v5/position/set-leverage", body, private=True)
        rc = int(data.get("retCode", -1))
        if rc not in (0, 110043):  # 110043 = leverage not modified
            raise RuntimeError(f"Bybit error retCode={data.get('retCode')} retMsg={data.get('retMsg')} data={data}")
        self._sleep()

    def _detect_mode(self, symbol: str) -> str:
        if self._detected_mode:
            return self._detected_mode
        try:
            data = self._send("GET", "/v5/position/list", {"category": self.category, "symbol": symbol}, private=True)
            if str(data.get("retCode")) == "0":
                for p in (data.get("result", {}) or {}).get("list", []) or []:
                    if int(p.get("positionIdx", 0)) in (1, 2):
                        self._detected_mode = "hedge"
                        break
                if not self._detected_mode:
                    self._detected_mode = "one_way"
                return self._detected_mode
        except Exception:
            pass
        self._detected_mode = self.position_mode if self.position_mode in ("one_way", "hedge") else "one_way"
        return self._detected_mode

    def _position_idx(self, symbol: str, side: str) -> int:
        # side: "Buy"/"Sell"
        mode = self._detect_mode(symbol)
        if mode == "hedge":
            return 1 if side.lower() == "buy" else 2
        return 0

    # ───────── Orders ─────────
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def create_market_order(self, symbol: str, side: str, qty: float, take_profit: float, stop_loss: float) -> Tuple[str, Optional[float]]:
        order_side = "Buy" if side.lower() in ("buy", "long") else "Sell"
        position_idx = self._position_idx(symbol, order_side)
        body = {
            "category": self.category,
            "symbol": symbol,
            "side": order_side,
            "orderType": "Market",
            "qty": str(qty),
            "positionIdx": position_idx,
            "takeProfit": str(take_profit),
            "stopLoss": str(stop_loss),
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
            "reduceOnly": False,
        }
        data = self._send("POST", "/v5/order/create", body, private=True)
        self._req_check(data)
        result = data.get("result", {}) or {}
        order_id = result.get("orderId") or ""
        entry_price = None
        try:
            for it in self.get_tickers():
                if it.get("symbol") == symbol:
                    entry_price = float(it.get("lastPrice"))
                    break
        except Exception:
            pass
        self._sleep()
        return order_id, entry_price

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def create_market_order_simple(self, symbol: str, side: str, qty: float) -> Tuple[str, Optional[float]]:
        order_side = "Buy" if side.lower() in ("buy", "long") else "Sell"
        position_idx = self._position_idx(symbol, order_side)
        body = {
            "category": self.category,
            "symbol": symbol,
            "side": order_side,
            "orderType": "Market",
            "qty": str(qty),
            "positionIdx": position_idx,
            "reduceOnly": False,
        }
        data = self._send("POST", "/v5/order/create", body, private=True)
        self._req_check(data)
        result = data.get("result", {}) or {}
        order_id = result.get("orderId") or ""
        entry_price = None
        try:
            for it in self.get_tickers():
                if it.get("symbol") == symbol:
                    entry_price = float(it.get("lastPrice"))
                    break
        except Exception:
            pass
        self._sleep()
        return order_id, entry_price

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def set_trading_stop(self, symbol: str, side: str, take_profit: float, stop_loss: float):
        order_side = "Buy" if side.lower() in ("buy", "long") else "Sell"
        position_idx = self._position_idx(symbol, order_side)
        body = {
            "category": self.category,
            "symbol": symbol,
            "positionIdx": position_idx,
            "takeProfit": str(take_profit),
            "stopLoss": str(stop_loss),
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
        }
        data = self._send("POST", "/v5/position/trading-stop", body, private=True)
        self._req_check(data)
        self._sleep()

    # --- Доп. операции для «мягкого» закрытия и PnL-чекера ---

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def set_take_profit_only(self, symbol: str, side: str, take_profit: float):
        """
        Обновляет только TP у позиции (без SL).
        """
        order_side = "Buy" if side.lower() in ("buy", "long") else "Sell"
        position_idx = self._position_idx(symbol, order_side)
        body = {
            "category": self.category,
            "symbol": symbol,
            "positionIdx": position_idx,
            "takeProfit": str(take_profit),
            "tpTriggerBy": "LastPrice",
        }
        data = self._send("POST", "/v5/position/trading-stop", body, private=True)
        self._req_check(data)
        self._sleep()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def close_position_market(self, symbol: str, side: str, qty: float) -> str:
        """
        Закрытие позиции маркетом с reduceOnly=true.
        Если текущая позиция LONG (side='Buy'), то закрываем ордером Sell, и наоборот.
        """
        # «Сторона ордера» должна быть ПРОТИВОПОЛОЖНА стороне позиции
        close_side = "Sell" if side.lower() == "buy" else "Buy"
        position_idx = self._position_idx(symbol, close_side)
        body = {
            "category": self.category,
            "symbol": symbol,
            "side": close_side,
            "orderType": "Market",
            "qty": str(qty),
            "positionIdx": position_idx,
            "reduceOnly": True,
        }
        data = self._send("POST", "/v5/order/create", body, private=True)
        self._req_check(data)
        result = data.get("result", {}) or {}
        order_id = result.get("orderId") or ""
        self._sleep()
        return order_id

    # ───────── Closed PnL (24h) ─────────
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_closed_pnl(self, symbol: Optional[str], start_time_ms: int, end_time_ms: int) -> List[Dict[str, Any]]:
        params = {"category": self.category, "startTime": start_time_ms, "endTime": end_time_ms, "limit": 50}
        if symbol:
            params["symbol"] = symbol
        out: List[Dict[str, Any]] = []
        cursor = None
        while True:
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            data = self._send("GET", "/v5/position/closed-pnl", p, private=True)
            self._req_check(data)
            result = data.get("result", {}) or {}
            out.extend(result.get("list", []) or [])
            cursor = result.get("nextPageCursor")
            if not cursor:
                break
            self._sleep()
        return out

    def had_closed_within(self, symbol: str, hours: int = 24) -> bool:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - hours * 3600 * 1000
        try:
            return len(self.get_closed_pnl(symbol, start_ms, end_ms)) > 0
        except Exception as e:
            logging.warning(f"closed-pnl check failed for {symbol}: {e}")
            return False
