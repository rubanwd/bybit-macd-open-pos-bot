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


# ----------------- CONST / HELPERS -----------------

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
    "1MN": "M",  # month, используем ключ 1MN, чтобы не путать с 1 минутой
}
LONG_TF_CODES = {"W", "M"}  # недельный и месячный дают мало баров, режем limit


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


def parse_timeframes(s: str) -> List[str]:
    if not s:
        return ["1D", "1W"]
    items = [x.strip().upper() for x in s.split(",") if x.strip()]
    # допускаем 1MN как «месяц»
    valid = []
    for x in items:
        if x in TF_TO_BYBIT:
            valid.append(x)
        elif x == "1M":  # если пользователь указал 1M как месяц — поддержим
            valid.append("1MN")
    return valid or ["1D", "1W"]


def check_sort_tf(s, tfs):
    s = (s or "").strip().upper()
    if s == "1M":  # аналогично 1M → 1MN
        s = "1MN"
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


def fmt_tp_sl(value: float) -> str:
    """
    Читаемый формат TP/SL: оставляем три значащих цифры после ведущих нулей.
    """
    s = f"{float(value):.18f}".rstrip("0")
    if "." not in s:
        return s
    intp, frac = s.split(".", 1)
    i = 0
    while i < len(frac) and frac[i] == "0":
        i += 1
    keep = frac[i:i + 3]
    if not keep:
        return intp + "." + frac
    return intp + "." + (frac[:i] + keep)


def snapshot_rsi(api: BybitAPI, symbol: str, rsi_period: int) -> Dict[str, float]:
    """
    Быстрая подзагрузка RSI для 5M, 15M, 1H (60 баров).
    """
    out: Dict[str, float] = {}
    for tf_code in ["5M", "15M", "1H"]:
        try:
            interval = TF_TO_BYBIT[tf_code]
            kl = api.get_klines(symbol, interval, limit=60)
            df = kline_to_df(kl)
            rs = rsi(df["close"], rsi_period)
            out[tf_code] = float(rs.iloc[-1])
        except Exception:
            out[tf_code] = 0.0
    return out


def compute_atr_abs(api: BybitAPI, symbol: str, tf_code: str, atr_period: int) -> float:
    """
    Подсчёт ATR(abs) для указанного TF (если его нет в item).
    """
    try:
        interval = TF_TO_BYBIT[tf_code]
        kl = api.get_klines(symbol, interval, limit=120)
        df = kline_to_df(kl)
        aser = atr(df["high"], df["low"], df["close"], atr_period)
        return float(aser.iloc[-1])
    except Exception:
        return 0.0


# ------------------------ MAIN LOOP ------------------------

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
    if ATR_TF_FOR_SLTP == "1M":
        ATR_TF_FOR_SLTP = "1MN"
    TP_ATR_MULT = env_float("TP_ATR_MULT", 1.5)
    SL_ATR_MULT = env_float("SL_ATR_MULT", 1.0)

    # RSI-правила порогов (оставляем прежние дефолты, можно менять в .env)
    BULL_LONG_RSI_MAX_5M  = env_float("BULL_LONG_RSI_MAX_5M", 70.0)
    BULL_LONG_RSI_MAX_15M = env_float("BULL_LONG_RSI_MAX_15M", 70.0)
    BULL_LONG_RSI_MAX_1H  = env_float("BULL_LONG_RSI_MAX_1H", 70.0)
    BULL_SHORT_RSI_MIN_5M  = env_float("BULL_SHORT_RSI_MIN_5M", 85.0)
    BULL_SHORT_RSI_MIN_15M = env_float("BULL_SHORT_RSI_MIN_15M", 85.0)
    BULL_SHORT_RSI_MIN_1H  = env_float("BULL_SHORT_RSI_MIN_1H", 85.0)

    BEAR_SHORT_RSI_MIN_5M  = env_float("BEAR_SHORT_RSI_MIN_5M", 30.0)
    BEAR_SHORT_RSI_MIN_15M = env_float("BEAR_SHORT_RSI_MIN_15M", 30.0)
    BEAR_SHORT_RSI_MIN_1H  = env_float("BEAR_SHORT_RSI_MIN_1H", 30.0)
    BEAR_LONG_RSI_MAX_5M  = env_float("BEAR_LONG_RSI_MAX_5M", 15.0)
    BEAR_LONG_RSI_MAX_15M = env_float("BEAR_LONG_RSI_MAX_15M", 15.0)
    BEAR_LONG_RSI_MAX_1H  = env_float("BEAR_LONG_RSI_MAX_1H", 15.0)

    TIMEFRAMES = parse_timeframes(os.getenv("TIMEFRAMES", "1H,4H,1D"))
    SORT_TF = check_sort_tf(os.getenv("SORT_TF", TIMEFRAMES[0]), TIMEFRAMES)
    REOPEN_COOLDOWN_HOURS = env_int("REOPEN_COOLDOWN_HOURS", 24)
    if ATR_TF_FOR_SLTP not in TIMEFRAMES:
        TIMEFRAMES.append(ATR_TF_FOR_SLTP)

    # Описание тренд-правила для сообщений
    TREND_RULE_HUMAN = "trend rule: 1H veto + confirmation by (15M OR 5M)"

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
    sign_style = (os.getenv("BYBIT_SIGN_STYLE") or "headers").strip().lower()
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
            # 1) Инструменты и мета
            instruments = api.get_instruments()
            symbols = [it["symbol"] for it in instruments]
            sym_info = api.build_symbol_info_map(instruments)
            logging.info(f"Всего символов: {len(symbols)}")

            # 2) Префильтр по /tickers
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

            # 3) Индикаторы + тренды
            def load_pair(sym: str) -> Tuple[str, Optional[Dict]]:
                try:
                    trends, rsis, atr_abs, atr_pct = {}, {}, {}, {}
                    last_close_for_price = None
                    for tf in TIMEFRAMES:
                        tf_key = "1MN" if tf == "1M" else tf
                        interval = TF_TO_BYBIT[tf_key]
                        limit = min(LIMIT, 120) if interval in LONG_TF_CODES else LIMIT
                        kl = api.get_klines(sym, interval, limit=limit)
                        # порог баров
                        if (interval in LONG_TF_CODES and len(kl) < 30) or (interval not in LONG_TF_CODES and len(kl) < 50):
                            return sym, None

                        df = kline_to_df(kl)
                        m, s, h, rs, asr = compute_indicators(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD, ATR_PERIOD)
                        trends[tf_key] = classify_trend(m, s, h)
                        rsis[tf_key] = float(rs.iloc[-1])

                        last_close = float(df["close"].iloc[-1])
                        last_close_for_price = last_close
                        aa = float(asr.iloc[-1])
                        ap = (aa / last_close) if last_close else 0.0
                        atr_abs[tf_key] = aa
                        atr_pct[tf_key] = ap

                    # рыночная цена (если есть)
                    last_market = None
                    t = tick_map.get(sym)
                    if t:
                        try:
                            last_market = float(t.get("lastPrice"))
                        except Exception:
                            pass
                    last_price = last_market or last_close_for_price or 0.0

                    return sym, {
                        "trends": trends,                 # карта трендов по ТФ
                        "atr_abs_map": atr_abs,
                        "atr_pct_map": atr_pct,
                        "rsi_map": rsis,
                        "last_price": last_price,
                        "atr_sort_abs": atr_abs.get(SORT_TF if SORT_TF != "1M" else "1MN", 0.0),
                        "atr_sort_pct": atr_pct.get(SORT_TF if SORT_TF != "1M" else "1MN", 0.0),
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

            # 4) Формируем BULL/BEAR по правилу: 1H veto + (15M OR 5M)
            bull_list, bear_list = [], []
            for sym, d in results.items():
                it = {
                    "symbol": f"{sym.replace('USDT', '')}/USDT",
                    "exchange_symbol": sym,
                    "last_price": d["last_price"],
                    "atr_abs": d["atr_sort_abs"],
                    "atr_pct": d["atr_sort_pct"],
                }
                # переносим все RSI/ATR
                for tf in TIMEFRAMES:
                    k = "1MN" if tf == "1M" else tf
                    it[f"rsi_{tf}"] = d["rsi_map"].get(k, 0.0)
                    it[f"atr_pct_{tf}"] = d["atr_pct_map"].get(k, 0.0)
                    it[f"atr_abs_{tf}"] = d["atr_abs_map"].get(k, 0.0)

                tr = d["trends"]
                tr5  = tr.get("5M")
                tr15 = tr.get("15M")
                tr1h = tr.get("1H")

                is_bull = (tr1h == "BULL") and (tr15 == "BULL" or tr5 == "BULL")
                is_bear = (tr1h == "BEAR") and (tr15 == "BEAR" or tr5 == "BEAR")

                it["trend_5M"] = tr5 or "NEUTRAL"
                it["trend_15M"] = tr15 or "NEUTRAL"
                it["trend_1H"] = tr1h or "NEUTRAL"
                it["trend_rule"] = TREND_RULE_HUMAN

                if is_bull:
                    bull_list.append(it)
                elif is_bear:
                    bear_list.append(it)
                # иначе — отбрасываем

            # 5) Ограничение по TOP_N (отсортировано по ATR% SORT_TF)
            bull_list = sorted(bull_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
            bear_list = sorted(bear_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
            logging.info(f"Финальный отбор ({'+'.join(TIMEFRAMES)}): BULL={len(bull_list)} BEAR={len(bear_list)}")

            # 6) Добавим Open Interest
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

            # 7) Сортировки для отчёта
            bull_sorted = sorted(bull_list, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)
            bear_sorted = sorted(bear_list, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)

            # 8) Отчёт в Telegram (добавили детали тренд-правила)
            report_text = build_report_txt(
                bull_sorted,
                bear_sorted,
                timeframes=TIMEFRAMES,
                sort_tf=SORT_TF,
                tz="Europe/Kyiv",
            )
            filepath = write_report_file(report_text)
            report_caption = (
                f"BYBIT MACD Scanner — отчёт ({' & '.join(TIMEFRAMES)}). "
                f"{TREND_RULE_HUMAN}. MACD({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}); "
                f"RSI={RSI_PERIOD}; ATR={ATR_PERIOD}."
            )
            tg_report.send_document(filepath, caption=report_caption)

            # ─────────── Торговый блок ───────────
            if ENABLE_TRADING:
                private_ok = True
                try:
                    # простая проверка приватного доступа
                    _ = api.get_open_positions("BTCUSDT")
                except Exception as e:
                    private_ok = False
                    logging.error(f"Приватный API недоступен (торговый блок пропущен): {e}")

                if private_ok:
                    # учитываем лимит всех позиций
                    try:
                        current_open_count = api.count_open_positions()
                    except Exception as e:
                        logging.error(f"Не удалось получить список всех открытых позиций: {e}")
                        current_open_count = MAX_OPEN_POS

                    if current_open_count >= MAX_OPEN_POS:
                        logging.info(
                            f"Достигнут лимит открытых позиций: {current_open_count}/{MAX_OPEN_POS}. Новые входы пропущены."
                        )
                    else:
                        prefer_source = "BULL" if len(bull_sorted) >= len(bear_sorted) else "BEAR"

                        def cands_bull():
                            return sorted(bull_sorted, key=lambda it: rsi_sum(it, TIMEFRAMES))

                        def cands_bear():
                            return sorted(bear_sorted, key=lambda it: rsi_sum(it, TIMEFRAMES), reverse=True)

                        opened = False
                        for source in [prefer_source, "BEAR" if prefer_source == "BULL" else "BULL"]:
                            cands = cands_bull() if source == "BULL" else cands_bear()
                            for it in cands:
                                # на каждом шаге проверим лимит ещё раз
                                try:
                                    current_open_count = api.count_open_positions()
                                except Exception:
                                    current_open_count = MAX_OPEN_POS
                                if current_open_count >= MAX_OPEN_POS:
                                    logging.info(
                                        f"Лимит открытых позиций: {current_open_count}/{MAX_OPEN_POS}. "
                                        f"Останавливаю входы в этом цикле."
                                    )
                                    opened = True
                                    break

                                sym = it["exchange_symbol"]
                                info = sym_info.get(sym, {})

                                # нет активной позиции + кулдаун
                                if api.has_open_position(sym):
                                    continue
                                try:
                                    if api.had_closed_within(sym, REOPEN_COOLDOWN_HOURS):
                                        continue
                                except Exception as e:
                                    logging.warning(f"closed-pnl check failed for {sym}: {e}")
                                    if os.getenv("COOLDOWN_BLOCK_ON_ERROR", "1").lower() not in ("0", "false"):
                                        continue

                                last_price = float(it["last_price"])
                                if last_price <= 0:
                                    continue

                                # Размер ордера
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

                                atr_abs_for_sltp = float(it.get(f"atr_abs_{ATR_TF_FOR_SLTP}", 0.0))
                                if atr_abs_for_sltp <= 0:
                                    atr_abs_for_sltp = compute_atr_abs(api, sym, ATR_TF_FOR_SLTP, ATR_PERIOD)
                                    if atr_abs_for_sltp <= 0:
                                        continue

                                # Выбор стороны с учётом RSI-правил (AND по 5M/15M/1H)
                                rsi_snap = snapshot_rsi(api, sym, RSI_PERIOD)
                                r5, r15, r1h = rsi_snap.get("5M", 0.0), rsi_snap.get("15M", 0.0), rsi_snap.get("1H", 0.0)

                                order_side: Optional[str] = None
                                if source == "BULL":
                                    cond_long  = (r5  <= BULL_LONG_RSI_MAX_5M and
                                                  r15 <= BULL_LONG_RSI_MAX_15M and
                                                  r1h <= BULL_LONG_RSI_MAX_1H)
                                    cond_short = (r5  >  BULL_SHORT_RSI_MIN_5M and
                                                  r15 >  BULL_SHORT_RSI_MIN_15M and
                                                  r1h >  BULL_SHORT_RSI_MIN_1H)
                                    if   cond_long:  order_side = "Buy"
                                    elif cond_short: order_side = "Sell"
                                    else:            continue
                                else:  # source == "BEAR"
                                    cond_short = (r5  >= BEAR_SHORT_RSI_MIN_5M and
                                                  r15 >= BEAR_SHORT_RSI_MIN_15M and
                                                  r1h >= BEAR_SHORT_RSI_MIN_1H)
                                    cond_long  = (r5  <  BEAR_LONG_RSI_MAX_5M and
                                                  r15 <  BEAR_LONG_RSI_MAX_15M and
                                                  r1h <  BEAR_LONG_RSI_MAX_1H)
                                    if   cond_short: order_side = "Sell"
                                    elif cond_long:  order_side = "Buy"
                                    else:            continue

                                # TP/SL по ATR
                                if order_side == "Buy":
                                    raw_tp = last_price + atr_abs_for_sltp * TP_ATR_MULT
                                    raw_sl = last_price - atr_abs_for_sltp * SL_ATR_MULT
                                else:
                                    raw_tp = last_price - atr_abs_for_sltp * TP_ATR_MULT
                                    raw_sl = last_price + atr_abs_for_sltp * SL_ATR_MULT

                                tp = api.clamp_price_safe(raw_tp, info)
                                sl = api.clamp_price_safe(raw_sl, info)

                                # Отправка на биржу
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

                                # -------- Уведомление в trades-канал (детали тренда + метрики) --------
                                try:
                                    # ATR% для 5M/15M/1H
                                    atr5_abs  = it.get("atr_abs_5M")  or compute_atr_abs(api, sym, "5M",  ATR_PERIOD)
                                    atr15_abs = it.get("atr_abs_15M") or compute_atr_abs(api, sym, "15M", ATR_PERIOD)
                                    atr1h_abs = it.get("atr_abs_1H")  or compute_atr_abs(api, sym, "1H",  ATR_PERIOD)

                                    atr5_pct   = (float(atr5_abs)   / last_price * 100.0) if last_price > 0 else 0.0
                                    atr15_pct  = (float(atr15_abs)  / last_price * 100.0) if last_price > 0 else 0.0
                                    atr1h_pct  = (float(atr1h_abs)  / last_price * 100.0) if last_price > 0 else 0.0
                                    atr_sltp_pct = (float(atr_abs_for_sltp) / last_price * 100.0) if last_price > 0 else 0.0

                                    msg = (
                                        f"🔔 Открыта позиция {sym} {order_side}\n"
                                        f"Цена: {(entry_price or last_price):.8f}\n"
                                        f"Кол-во: {qty}\n"
                                        f"Плечо: x{LEVERAGE}\n"
                                        f"TP: {fmt_tp_sl(tp)}\n"
                                        f"SL: {fmt_tp_sl(sl)}\n"
                                        f"ATR(5M)={fmt2(atr5_pct)}% | ATR(15M)={fmt2(atr15_pct)}% | ATR(1H)={fmt2(atr1h_pct)}% • SL/TP ATR({ATR_TF_FOR_SLTP})={fmt2(atr_sltp_pct)}%\n"
                                        f"RSI ➜ 5M: {fmt2(r5)} | 15M: {fmt2(r15)} | 1H: {fmt2(r1h)}\n"
                                        f"Trend ➜ 5M:{it['trend_5M']} | 15M:{it['trend_15M']} | 1H:{it['trend_1H']} ({TREND_RULE_HUMAN})\n"
                                        f"Время: {now_iso()}"
                                    )
                                    if tg_trades:
                                        tg_trades.send_message(msg)
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
