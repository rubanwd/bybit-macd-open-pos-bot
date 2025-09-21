# main.py
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from tenacity import RetryError

from bybit_api import BybitAPI
from indicators import macd, rsi, atr
from reporter import build_report_txt, write_report_file
from telegram_utils import TelegramClient


TF_TO_BYBIT = {
    "5M": "5",
    "15M": "15",
    "30M": "30",
    "1H": "60",
    "4H": "240",
    "6H": "360",
    "12H": "720",
    "1D": "D",
    "1W": "W",
    "1M": "M",
}
LONG_TF_CODES = {"W", "M"}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def env_int(n, d):
    try:
        return int(os.getenv(n, d))
    except Exception:
        return d


def env_float(n, d):
    try:
        return float(os.getenv(n, d))
    except Exception:
        return d


def parse_timeframes(s):
    if not s:
        return ["1D", "1W"]
    items = [x.strip().upper() for x in s.split(",") if x.strip()]
    return [x for x in items if x in TF_TO_BYBIT] or ["1D", "1W"]


def check_sort_tf(s, tfs):
    s = (s or "").strip().upper()
    if s in tfs:
        return s
    logging.warning(f"SORT_TF={s} не найден в TIMEFRAMES. Использую {tfs[0]}.")
    return tfs[0]


def kline_to_df(kl):
    return pd.DataFrame(kl)[["open", "high", "low", "close"]].astype(float)


def compute_indicators(df, mf, ms, msig, rper, aper):
    ml, sl, h = macd(df["close"], mf, ms, msig)
    rs = rsi(df["close"], rper)
    asr = atr(df["high"], df["low"], df["close"], aper)
    return ml, sl, h, rs, asr


def classify_trend(ml, sl, h):
    bull = (ml.iloc[-1] > sl.iloc[-1]) and (h.iloc[-1] > 0)
    bear = (ml.iloc[-1] < sl.iloc[-1]) and (h.iloc[-1] < 0)
    return "BULL" if bull else ("BEAR" if bear else "NEUTRAL")


def rsi_sum(item, tfs):
    return sum(float(item.get(f"rsi_{tf}", 0.0)) for tf in tfs)


def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def fmt2(x: float) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"

def snapshot_rsi(api: BybitAPI, symbol: str, rsi_period: int) -> Dict[str, float]:
    """
    Возвращает RSI для таймфреймов 5M, 15M, 1H.
    Если эти RSI уже были вычислены в ходе отбора (it["rsi_5M"] и т.п.) — возьмём их,
    иначе быстро дольём свечи и посчитаем тут.
    """
    out: Dict[str, float] = {}
    for tf_code in ["5M", "15M", "1H"]:
        try:
            interval = TF_TO_BYBIT[tf_code]
            # нужно ≥ 50 баров для устойчивости
            kl = api.get_klines(symbol, interval, limit=60)
            df = kline_to_df(kl)
            rs = rsi(df["close"], rsi_period)
            out[tf_code] = float(rs.iloc[-1])
        except Exception:
            # если совсем не смогли — 0
            out[tf_code] = 0.0
    return out

def fmt_tp_sl(value: float) -> str:
    """
    Отформатировать число так, чтобы после ведущих нулей в дробной части
    осталось ровно три значащих цифры (без округления, просто отсечение).
    Примеры:
      0.0455500001 -> 0.0455
      0.0441400000 -> 0.0441
      0.4555000000 -> 0.455
      0.0000455    -> 0.0000455
    """
    s = f"{float(value):.18f}".rstrip("0")
    if "." not in s:
        return s
    intp, frac = s.split(".", 1)
    # считаем ведущие нули в дробной части
    i = 0
    while i < len(frac) and frac[i] == "0":
        i += 1
    # оставляем три значащих цифры после первых нулей
    keep = frac[i:i + 3]
    if not keep:
        # на всякий случай, если все нули
        return intp + "." + frac
    new_frac = frac[:i] + keep
    return intp + "." + new_frac




def main_loop():
    load_dotenv()
    setup_logging()

    # === ENV ===
    SCAN_INTERVAL_MINUTES = env_int("SCAN_INTERVAL_MINUTES", 10)
    CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")
    TOP_N = env_int("TOP_N", 100)
    LIMIT = env_int("KLINES_LIMIT", 200)

    MACD_FAST = env_int("MACD_FAST", 12)
    MACD_SLOW = env_int("MACD_SLOW", 26)
    MACD_SIGNAL = env_int("MACD_SIGNAL", 9)
    RSI_PERIOD = env_int("RSI_PERIOD", 14)
    ATR_PERIOD = env_int("ATR_PERIOD", 14)

    SLEEP_MS = env_int("PER_REQUEST_SLEEP_MS", 250)
    MAX_RETRIES = env_int("MAX_RETRIES", 3)
    RETRY_BACKOFF = env_int("RETRY_BACKOFF_SEC", 2)

    WORKERS = env_int("WORKERS", 8)
    USE_TICKERS_PREFILTER = env_int("USE_TICKERS_PREFILTER", 1)
    PREFILTER_MULTIPLIER = env_int("PREFILTER_MULTIPLIER", 1)

    ENABLE_TRADING = env_int("ENABLE_TRADING", 0) == 1
    ORDER_VALUE_PCT = env_float("ORDER_VALUE_PCT", 0.20)
    MAX_ORDER_NOTIONAL_USDT = env_float("MAX_ORDER_NOTIONAL_USDT", 100.0)
    LEVERAGE = env_int("LEVERAGE", 10)
    MAX_OPEN_POS = env_int("MAX_OPEN_POSITIONS", 3)
    ATR_TF_FOR_SLTP = (os.getenv("ATR_TF_FOR_SLTP") or "1H").upper()
    TP_ATR_MULT = env_float("TP_ATR_MULT", 1.5)
    SL_ATR_MULT = env_float("SL_ATR_MULT", 1.0)

    TIMEFRAMES = parse_timeframes(os.getenv("TIMEFRAMES", "1H,4H,1D"))
    SORT_TF = check_sort_tf(os.getenv("SORT_TF", TIMEFRAMES[0]), TIMEFRAMES)
    if ATR_TF_FOR_SLTP not in TIMEFRAMES:
        TIMEFRAMES.append(ATR_TF_FOR_SLTP)

    logging.info(f"Активные ТФ: {', '.join(TIMEFRAMES)} | отсечка TopN по ATR%: {SORT_TF}")

    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    TRD_TOKEN = os.getenv("TELEGRAM_TRADES_BOT_TOKEN")
    TRD_CHAT = os.getenv("TELEGRAM_TRADES_CHAT_ID")
    if not TG_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы в .env")
    tg_report = TelegramClient(TG_TOKEN, TG_CHAT_ID)
    tg_trades = TelegramClient(TRD_TOKEN, TRD_CHAT) if (ENABLE_TRADING and TRD_TOKEN and TRD_CHAT) else None

    api_key = os.getenv("BYBIT_API_KEY") or ""
    api_secret = os.getenv("BYBIT_API_SECRET") or ""
    recv_window = env_int("BYBIT_RECV_WINDOW", 60000)
    base_url = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").strip()
    sign_style = (os.getenv("BYBIT_SIGN_STYLE") or "headers").strip().lower()  # demo → "params"
    position_mode = (os.getenv("POSITION_MODE") or "one_way").strip().lower()
    account_type_env = (os.getenv("BYBIT_ACCOUNT_TYPE") or "UNIFIED").strip().upper()

    logging.info(f"Bybit base: {base_url} | sign_style={sign_style} | Trading={'ON' if ENABLE_TRADING else 'OFF'}")

    api = BybitAPI(
        category=CATEGORY,
        sleep_ms=SLEEP_MS,
        max_retries=MAX_RETRIES,
        retry_backoff_sec=RETRY_BACKOFF,
        api_key=api_key,
        api_secret=api_secret,
        recv_window=recv_window,
        base_url=base_url,
        sign_style=sign_style,
        position_mode=position_mode,
    )

    while True:
        logging.info("=== Новый цикл ===")
        try:
            # 1) инструменты и мета
            instruments = api.get_instruments()
            symbols = [it["symbol"] for it in instruments]
            sym_info = api.build_symbol_info_map(instruments)
            logging.info(f"Всего символов: {len(symbols)}")

            # 2) префильтр по /tickers
            tickers = api.get_tickers()
            tick_map = {t["symbol"]: t for t in tickers if t.get("symbol") in symbols}
            if USE_TICKERS_PREFILTER:
                rows = []
                for sym, t in tick_map.items():
                    try:
                        high = float(t.get("highPrice24h") or 0)
                        low = float(t.get("lowPrice24h") or 0)
                        last = float(t.get("lastPrice") or 0)
                        if last <= 0 or high <= 0 or low <= 0:
                            continue
                        rows.append({"symbol": sym, "range24h_pct": (high - low) / last})
                    except Exception:
                        continue
                rows = sorted(rows, key=lambda x: x["range24h_pct"], reverse=True)
                pre_top = [r["symbol"] for r in rows[:max(TOP_N * PREFILTER_MULTIPLIER, TOP_N)]]
                logging.info(f"Префильтр по /tickers: выбрано {len(pre_top)} (multiplier={PREFILTER_MULTIPLIER}).")
            else:
                pre_top = symbols

            # 3) индикаторы
            def load_pair(sym: str) -> Tuple[str, Optional[Dict]]:
                try:
                    trends, rsis, atr_abs, atr_pct = {}, {}, {}, {}
                    last_close_for_price = None
                    for tf in TIMEFRAMES:
                        interval = TF_TO_BYBIT[tf]
                        limit = min(LIMIT, 120) if interval in LONG_TF_CODES else LIMIT
                        kl = api.get_klines(sym, interval, limit=limit)
                        if (interval in LONG_TF_CODES and len(kl) < 30) or (interval not in LONG_TF_CODES and len(kl) < 50):
                            return sym, None
                        df = kline_to_df(kl)
                        m, s, h, rs, asr = compute_indicators(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD, ATR_PERIOD)
                        trends[tf] = classify_trend(m, s, h)
                        rsis[tf] = float(rs.iloc[-1])
                        last_close = float(df["close"].iloc[-1])
                        last_close_for_price = last_close
                        aa = float(asr.iloc[-1])
                        ap = (aa / last_close) if last_close else 0.0
                        atr_abs[tf] = aa
                        atr_pct[tf] = ap

                    uniq = set(trends.values())
                    if "NEUTRAL" in uniq or len(uniq) != 1:
                        return sym, None
                    common = uniq.pop()

                    last_market = None
                    t = tick_map.get(sym)
                    if t:
                        try:
                            last_market = float(t.get("lastPrice"))
                        except Exception:
                            pass
                    last_price = last_market or last_close_for_price or 0.0

                    return sym, {
                        "common_trend": common,
                        "atr_abs_map": atr_abs,
                        "atr_pct_map": atr_pct,
                        "rsi_map": rsis,
                        "last_price": last_price,
                        "atr_sort_abs": atr_abs.get(SORT_TF, 0.0),
                        "atr_sort_pct": atr_pct.get(SORT_TF, 0.0),
                    }
                except Exception as e:
                    logging.debug(f"[{sym}] ошибка загрузки/индикаторов: {e}")
                    return sym, None

            results: Dict[str, Dict] = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for f in as_completed([ex.submit(load_pair, s) for s in pre_top]):
                    sym, data = f.result()
                    if data:
                        results[sym] = data

            bull_list, bear_list = [], []
            for sym, d in results.items():
                it = {
                    "symbol": f"{sym.replace('USDT', '')}/USDT",
                    "exchange_symbol": sym,
                    "last_price": d["last_price"],
                    "atr_abs": d["atr_sort_abs"],
                    "atr_pct": d["atr_sort_pct"],
                }
                for tf in TIMEFRAMES:
                    it[f"rsi_{tf}"] = d["rsi_map"].get(tf, 0.0)
                    it[f"atr_pct_{tf}"] = d["atr_pct_map"].get(tf, 0.0)
                    it[f"atr_abs_{tf}"] = d["atr_abs_map"].get(tf, 0.0)

                if d["common_trend"] == "BULL":
                    bull_list.append(it)
                elif d["common_trend"] == "BEAR":
                    bear_list.append(it)

            bull_list = sorted(bull_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
            bear_list = sorted(bear_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
            logging.info(f"Финальный отбор ({'+'.join(TIMEFRAMES)}): BULL={len(bull_list)} BEAR={len(bear_list)}")

            def add_oi(it):
                sym = it["exchange_symbol"]
                try:
                    it["oi"] = api.get_open_interest(sym, "1h") or 0.0
                except Exception:
                    it["oi"] = 0.0
                return it

            with ThreadPoolExecutor(max_workers=min(6, WORKERS)) as ex:
                bull_list = list(ex.map(add_oi, bull_list))
                bear_list = list(ex.map(add_oi, bear_list))

            bull_sorted = sorted(bull_list, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)
            bear_sorted = sorted(bear_list, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)

            report_text = build_report_txt(bull_sorted, bear_sorted, timeframes=TIMEFRAMES, sort_tf=SORT_TF, tz="Europe/Kyiv")
            filepath = write_report_file(report_text)
            tg_report.send_document(filepath, caption=f"BYBIT MACD Scanner — отчёт ({' & '.join(TIMEFRAMES)})")

            # ───────────────── Торговый блок ─────────────────
            if ENABLE_TRADING:
                private_ok = True
                try:
                    # простая проверка приватного доступа
                    _ = api.get_open_positions("BTCUSDT")
                except Exception as e:
                    private_ok = False
                    logging.error(f"Приватный API недоступен (торговый блок пропущен): {e}")

                if private_ok:
                    # >>> ГЛАВНЫЙ ФИКС ЛИМИТА <<<
                    # считаем ВСЕ открытые позиции аккаунта (без фильтра по символам)
                    try:
                        # считаем ВСЕ открытые позиции аккаунта (без символа), с учетом settleCoin=USDT
                        current_open_count = api.count_open_positions()
                    except Exception as e:
                        logging.error(f"Не удалось получить список всех открытых позиций: {e}")
                        # ВАЖНО: чтобы не пробить лимит, на ошибке приравниваем к MAX_OPEN_POS
                        current_open_count = MAX_OPEN_POS

                    if current_open_count >= MAX_OPEN_POS:
                        logging.info(
                            f"Достигнут лимит открытых позиций: {current_open_count}/{MAX_OPEN_POS}. Новые входы пропущены."
                        )
                    else:
                        # вход максимум одной позиции за цикл
                        prefer_side = "BULL" if len(bull_sorted) >= len(bear_sorted) else "BEAR"

                        def cands_bull():
                            # для LONG выбираем наименьшую сумму RSI
                            return sorted(bull_sorted, key=lambda it: rsi_sum(it, TIMEFRAMES))

                        def cands_bear():
                            # для SHORT — наибольшую сумму RSI
                            return sorted(bear_sorted, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)

                        opened = False
                        for side in [prefer_side, "BEAR" if prefer_side == "BULL" else "BULL"]:
                            cands = cands_bull() if side == "BULL" else cands_bear()
                            for it in cands:
                                # повторно проверим лимит (на всякий случай, если что-то открылось параллельно)
                                try:
                                    current_open_count = len(api.get_open_positions())
                                except Exception:
                                    pass
                                if current_open_count >= MAX_OPEN_POS:
                                    logging.info(
                                        f"Достигнут лимит открытых позиций: {current_open_count}/{MAX_OPEN_POS}. "
                                        f"Останавливаю входы в этом цикле."
                                    )
                                    opened = True  # просто чтобы выйти из внешнего цикла
                                    break

                                sym = it["exchange_symbol"]
                                info = sym_info.get(sym, {})

                                # нет активной позиции по этому символу + не было закрытой за 24ч
                                if api.has_open_position(sym):
                                    continue
                                try:
                                    if api.had_closed_within(sym, 24):
                                        continue
                                except Exception as e:
                                    logging.warning(f"closed-pnl check failed for {sym}: {e}")

                                last_price = float(it["last_price"])
                                if last_price <= 0:
                                    continue

                                # размер ордера
                                try:
                                    usdt_avail = api.get_available_usdt(account_type_env)
                                except Exception as e:
                                    logging.error(f"Баланс недоступен: {e}")
                                    break
                                if usdt_avail <= 5:
                                    logging.info("Недостаточно средств.")
                                    break

                                notional = min(max(1.0, usdt_avail * ORDER_VALUE_PCT), MAX_ORDER_NOTIONAL_USDT)
                                qty = api.round_qty(sym, (notional * LEVERAGE) / last_price, info)
                                if qty <= 0:
                                    continue

                                atr_abs = float(it.get(f"atr_abs_{ATR_TF_FOR_SLTP}", 0.0))
                                if atr_abs <= 0:
                                    continue

                                if side == "BULL":
                                    raw_tp = last_price + atr_abs * TP_ATR_MULT
                                    raw_sl = last_price - atr_abs * SL_ATR_MULT
                                    order_side = "Buy"
                                else:
                                    raw_tp = last_price - atr_abs * TP_ATR_MULT
                                    raw_sl = last_price + atr_abs * SL_ATR_MULT
                                    order_side = "Sell"

                                tp = api.clamp_price_safe(raw_tp, info)
                                sl = api.clamp_price_safe(raw_sl, info)

                                try:
                                    api.set_leverage(sym, LEVERAGE, LEVERAGE)
                                    order_id, entry_price = api.create_market_order(sym, order_side, qty, tp, sl)
                                except RetryError as re:
                                    logging.error(f"Ошибка создания ордера по {sym}: {re.last_attempt.exception()}")
                                    try:
                                        order_id, entry_price = api.create_market_order_simple(sym, order_side, qty)
                                        api.set_trading_stop(sym, order_side, tp, sl)
                                    except Exception as e2:
                                        logging.error(f"Фолбэк тоже не удался для {sym}: {e2}")
                                        break
                                except Exception as e:
                                    logging.error(f"Ошибка создания ордера по {sym}: {e}")
                                    break

                                if tg_trades:
                                    try:
                                        # --- подготовим красивое уведомление ---
                                        atr_abs = float(it.get(f"atr_abs_{ATR_TF_FOR_SLTP}", 0.0))
                                        last_px = float(it["last_price"] or 0.0)
                                        atr_pct = (atr_abs / last_px * 100.0) if last_px > 0 else 0.0

                                        # RSI снимок на 5M/15M/1H (на текущую минуту)
                                        rsi_snap = snapshot_rsi(api, sym, RSI_PERIOD)
                                        rsi_5m = rsi_snap.get("5M", 0.0)
                                        rsi_15m = rsi_snap.get("15M", 0.0)
                                        rsi_1h = rsi_snap.get("1H", 0.0)

                                        msg = (
                                            f"🔔 Открыта позиция {sym} {order_side}\n"
                                            f"Цена: {entry_price or last_price:.8f}\n"
                                            f"Кол-во: {qty}\n"
                                            f"Плечо: x{LEVERAGE}\n"
                                            f"TP: {fmt_tp_sl(tp)}\n"
                                            f"SL: {fmt_tp_sl(sl)}\n"
                                            f"ATR({ATR_TF_FOR_SLTP})={fmt2(atr_pct)}%\n"
                                            f"RSI ➜ 5M: {fmt2(rsi_5m)} | 15M: {fmt2(rsi_15m)} | 1H: {fmt2(rsi_1h)}\n"
                                            f"Время: {now_iso()}"
                                        )

                                        if tg_trades:
                                            try:
                                                tg_trades.send_message(msg)
                                            except Exception as e:
                                                logging.error(f"Ошибка sendMessage: {e}")

                                    except Exception as e:
                                        logging.error(f"Ошибка sendMessage: {e}")

                                logging.info(f"Открыт ордер {order_id} по {sym} ({order_side}) qty={qty}")
                                opened = True
                                break
                            if opened:
                                break

        except Exception as e:
            logging.exception(f"Фатальная ошибка цикла: {e}")

        logging.info(f"Сон на {SCAN_INTERVAL_MINUTES} мин...")
        time.sleep(SCAN_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main_loop()
