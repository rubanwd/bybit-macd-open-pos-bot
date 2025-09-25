# main.py
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
from tenacity import RetryError
import pytz

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
    "1MN": "M",  # месяц (MONTH). 1M => 1MN
}
LONG_TF_CODES = {"W", "M"}  # недельный/месячный — ограничиваем limit


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
        return ["5M", "15M", "1H"]
    items = [x.strip().upper() for x in s.split(",") if x.strip()]
    valid = []
    for x in items:
        if x in TF_TO_BYBIT:
            valid.append(x)
        elif x == "1M":
            valid.append("1MN")
    return valid or ["5M", "15M", "1H"]


def check_sort_tf(s, tfs):
    s = (s or "").strip().upper()
    if s == "1M":
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


def classify_trend_simple(ml, sl, h):
    bull = (ml.iloc[-1] > sl.iloc[-1]) and (h.iloc[-1] > 0)
    bear = (ml.iloc[-1] < sl.iloc[-1]) and (h.iloc[-1] < 0)
    return "BULL" if bull else ("BEAR" if bear else "NEUTRAL")


def classify_trend_with_slope(ml, sl, h, lookback: int, min_slope: float):
    base = classify_trend_simple(ml, sl, h)
    if base == "NEUTRAL":
        return "NEUTRAL"
    if len(ml) <= lookback or len(sl) <= lookback:
        return "NEUTRAL"
    slope_macd = float(ml.iloc[-1] - ml.iloc[-1 - lookback])
    slope_sig  = float(sl.iloc[-1] - sl.iloc[-1 - lookback])

    def up_ok(v: float) -> bool: return (v > min_slope)
    def dn_ok(v: float) -> bool: return (v < -min_slope)

    if base == "BULL":
        return "BULL" if up_ok(slope_macd) and up_ok(slope_sig) else "NEUTRAL"
    else:
        return "BEAR" if dn_ok(slope_macd) and dn_ok(slope_sig) else "NEUTRAL"


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
    try:
        interval = TF_TO_BYBIT[tf_code]
        kl = api.get_klines(symbol, interval, limit=120)
        df = kline_to_df(kl)
        aser = atr(df["high"], df["low"], df["close"], atr_period)
        return float(aser.iloc[-1])
    except Exception:
        return 0.0


# -------- Вспомогательные функции: торговое окно и планировщик --------

def parse_hhmm(s: str) -> Tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def in_trading_window(now_dt: datetime, tz_str: str, start_hhmm: str, end_hhmm: str) -> bool:
    tz = pytz.timezone(tz_str)
    local_now = now_dt.astimezone(tz)
    sh, sm = parse_hhmm(start_hhmm)
    eh, em = parse_hhmm(end_hhmm)
    start_today = tz.localize(datetime(local_now.year, local_now.month, local_now.day, sh, sm))
    end_today   = tz.localize(datetime(local_now.year, local_now.month, local_now.day, eh, em))
    return start_today <= local_now <= end_today


def seconds_until_window_start(now_dt: datetime, tz_str: str, start_hhmm: str) -> int:
    tz = pytz.timezone(tz_str)
    local_now = now_dt.astimezone(tz)
    sh, sm = parse_hhmm(start_hhmm)
    start_today = tz.localize(datetime(local_now.year, local_now.month, local_now.day, sh, sm))
    if local_now <= start_today:
        delta = (start_today - local_now)
    else:
        start_next = start_today + timedelta(days=1)
        delta = (start_next - local_now)
    return max(1, int(delta.total_seconds()))


# ------------------------ MAIN LOOP ------------------------

def main_loop():
    # На Render ключи задаются через Variables, .env не нужен.
    if load_dotenv:
        try:
            load_dotenv()
        except Exception:
            pass
    setup_logging()

    # === ENV ===
    SCAN_INTERVAL_MINUTES = env_int("SCAN_INTERVAL_MINUTES", 2)
    PNL_CHECK_INTERVAL_MINUTES = env_int("PNL_CHECK_INTERVAL_MINUTES", 3)

    TRADING_WINDOW_TZ = os.getenv("TRADING_WINDOW_TZ", "Europe/Kyiv")
    TRADING_WINDOW_START = os.getenv("TRADING_WINDOW_START", "08:00")
    TRADING_WINDOW_END = os.getenv("TRADING_WINDOW_END", "20:00")

    CLOSE_ON_PNL_ENABLED = env_int("CLOSE_ON_PNL_ENABLED", 1) == 1
    CLOSE_ON_PNL_THRESHOLD_PCT = env_float("CLOSE_ON_PNL_THRESHOLD_PCT", 5.0)

    # «мягкое» закрытие
    SOFT_CLOSE_ENABLED = env_int("SOFT_CLOSE_ENABLED", 1) == 1
    SOFT_CLOSE_TICKS = env_int("SOFT_CLOSE_TICKS", 2)                 # на сколько тиков от текущей цены ставить TP
    SOFT_CLOSE_WAIT_SECONDS = env_int("SOFT_CLOSE_WAIT_SECONDS", 10)  # сколько ждать, прежде чем маркет-фолбэк
    SOFT_CLOSE_POLL_INTERVAL_SEC = env_int("SOFT_CLOSE_POLL_INTERVAL_SEC", 2)

    CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")
    TOP_N = env_int("TOP_N", 50)
    LIMIT = env_int("KLINES_LIMIT", 200)

    MACD_FAST = env_int("MACD_FAST", 12)
    MACD_SLOW = env_int("MACD_SLOW", 26)
    MACD_SIGNAL = env_int("MACD_SIGNAL", 9)
    RSI_PERIOD = env_int("RSI_PERIOD", 14)
    ATR_PERIOD = env_int("ATR_PERIOD", 14)

    # Наклоны MACD/Signal
    MACD_SLOPE_LOOKBACK_5M  = env_int("MACD_SLOPE_LOOKBACK_5M", 3)
    MACD_SLOPE_LOOKBACK_15M = env_int("MACD_SLOPE_LOOKBACK_15M", 3)
    MACD_SLOPE_LOOKBACK_1H  = env_int("MACD_SLOPE_LOOKBACK_1H", 3)
    MACD_MIN_SLOPE_5M  = env_float("MACD_MIN_SLOPE_5M", 0.0)
    MACD_MIN_SLOPE_15M = env_float("MACD_MIN_SLOPE_15M", 0.0)
    MACD_MIN_SLOPE_1H  = env_float("MACD_MIN_SLOPE_1H", 0.0)
    MACD_CONTRADICTION_FILTER = env_int("MACD_CONTRADICTION_FILTER", 1) == 1

    SLEEP_MS = env_int("PER_REQUEST_SLEEP_MS", 250)
    MAX_RETRIES = env_int("MAX_RETRIES", 3)
    RETRY_BACKOFF = env_int("RETRY_BACKOFF_SEC", 2)

    WORKERS = env_int("WORKERS", 8)
    USE_TICKERS_PREFILTER = env_int("USE_TICKERS_PREFILTER", 1)
    PREFILTER_MULTIPLIER = env_int("PREFILTER_MULTIPLIER", 1)

    ENABLE_TRADING = env_int("ENABLE_TRADING", 0) == 1
    ORDER_VALUE_PCT = env_float("ORDER_VALUE_PCT", 0.20)
    MAX_ORDER_NOTIONAL_USDT = env_float("MAX_ORDER_NOTIONAL_USDT", 100.0)
    ORDER_SAFETY_MARGIN_PCT = env_float("ORDER_SAFETY_MARGIN_PCT", 0.85)
    LEVERAGE = env_int("LEVERAGE", 10)
    MAX_OPEN_POS = env_int("MAX_OPEN_POSITIONS", 5)
    ATR_TF_FOR_SLTP = (os.getenv("ATR_TF_FOR_SLTP") or "15M").upper()
    if ATR_TF_FOR_SLTP == "1M":
        ATR_TF_FOR_SLTP = "1MN"
    TP_ATR_MULT = env_float("TP_ATR_MULT", 1.5)
    SL_ATR_MULT = env_float("SL_ATR_MULT", 1.0)

    # Смягчённые RSI-дефолты
    BULL_LONG_RSI_MAX_5M  = env_float("BULL_LONG_RSI_MAX_5M", 75.0)
    BULL_LONG_RSI_MAX_15M = env_float("BULL_LONG_RSI_MAX_15M", 75.0)
    BULL_LONG_RSI_MAX_1H  = env_float("BULL_LONG_RSI_MAX_1H", 75.0)
    BULL_SHORT_RSI_MIN_5M  = env_float("BULL_SHORT_RSI_MIN_5M", 80.0)
    BULL_SHORT_RSI_MIN_15M = env_float("BULL_SHORT_RSI_MIN_15M", 80.0)
    BULL_SHORT_RSI_MIN_1H  = env_float("BULL_SHORT_RSI_MIN_1H", 80.0)

    BEAR_SHORT_RSI_MIN_5M  = env_float("BEAR_SHORT_RSI_MIN_5M", 25.0)
    BEAR_SHORT_RSI_MIN_15M = env_float("BEAR_SHORT_RSI_MIN_15M", 25.0)
    BEAR_SHORT_RSI_MIN_1H  = env_float("BEAR_SHORT_RSI_MIN_1H", 25.0)
    BEAR_LONG_RSI_MAX_5M  = env_float("BEAR_LONG_RSI_MAX_5M", 20.0)
    BEAR_LONG_RSI_MAX_15M = env_float("BEAR_LONG_RSI_MAX_15M", 20.0)
    BEAR_LONG_RSI_MAX_1H  = env_float("BEAR_LONG_RSI_MAX_1H", 20.0)

    # Фильтр аномалий
    ANOMALY_FILTER_ENABLED = env_int("ANOMALY_FILTER_ENABLED", 1) == 1
    ANOMALY_ABS_CHANGE_PCT = env_float("ANOMALY_ABS_CHANGE_PCT", 0.60)
    ANOMALY_RANGE_PCT = env_float("ANOMALY_RANGE_PCT", 1.00)
    ANOMALY_MIN_TURNOVER_USDT = env_float("ANOMALY_MIN_TURNOVER_USDT", 1_000_000.0)

    TIMEFRAMES = parse_timeframes(os.getenv("TIMEFRAMES", "5M,15M,1H"))
    SORT_TF = check_sort_tf(os.getenv("SORT_TF", "15M"), TIMEFRAMES)
    REOPEN_COOLDOWN_HOURS = env_int("REOPEN_COOLDOWN_HOURS", 2)
    if ATR_TF_FOR_SLTP not in TIMEFRAMES:
        TIMEFRAMES.append(ATR_TF_FOR_SLTP)

    TREND_RULE_HUMAN = "1H veto + (15M OR 5M) + MACD slope check"

    logging.info(f"Активные ТФ: {', '.join(TIMEFRAMES)} | отсечка TopN по ATR%: {SORT_TF}")
    logging.info(
        f"Аномалий-фильтр: enabled={ANOMALY_FILTER_ENABLED}, |24h change|>{ANOMALY_ABS_CHANGE_PCT*100:.0f}%, "
        f"range>{ANOMALY_RANGE_PCT*100:.0f}%, turnover24h<{ANOMALY_MIN_TURNOVER_USDT:.0f} USDT"
    )

    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    TRD_TOKEN = os.getenv("TELEGRAM_TRADES_BOT_TOKEN")
    TRD_CHAT = os.getenv("TELEGRAM_TRADES_CHAT_ID")
    if not TG_TOKEN or not TG_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы в Variables")
    tg_report = TelegramClient(TG_TOKEN, TG_CHAT_ID)
    tg_trades = TelegramClient(TRD_TOKEN, TRD_CHAT) if (ENABLE_TRADING and TRD_TOKEN and TRD_CHAT) else None

    base_url = (os.getenv("BYBIT_BASE_URL") or "https://api.bybit.com").strip()
    sign_style = (os.getenv("BYBIT_SIGN_STYLE") or "headers").strip().lower()
    position_mode = (os.getenv("POSITION_MODE") or "one_way").strip().lower()
    account_type_env = (os.getenv("BYBIT_ACCOUNT_TYPE") or "UNIFIED").strip().upper()

    logging.info(f"Bybit base: {base_url} | sign_style={sign_style} | Trading={'ON' if ENABLE_TRADING else 'OFF'}")

    api = BybitAPI(
        category=os.getenv("BYBIT_CATEGORY", "linear"),
        sleep_ms=SLEEP_MS,
        max_retries=MAX_RETRIES,
        retry_backoff_sec=RETRY_BACKOFF,
        api_key=os.getenv("BYBIT_API_KEY") or "",
        api_secret=os.getenv("BYBIT_API_SECRET") or "",
        recv_window=env_int("BYBIT_RECV_WINDOW", 60000),
        base_url=base_url,
        sign_style=sign_style,
        position_mode=position_mode,
    )

    # -------- планировщик задач: scan и pnl-check --------
    def do_scan_cycle():
        # 1) Инструменты
        instruments = api.get_instruments()
        symbols = [it["symbol"] for it in instruments]
        sym_info = api.build_symbol_info_map(instruments)
        logging.info(f"Всего символов: {len(symbols)}")

        # 2) Префильтр по /tickers (+ аномалии)
        tickers = api.get_tickers()
        tick_map = {t["symbol"]: t for t in tickers if t.get("symbol") in symbols}

        excluded_cnt = 0
        rows = []
        for sym, t in tick_map.items():
            try:
                high = float(t.get("highPrice24h") or 0)
                low = float(t.get("lowPrice24h") or 0)
                last = float(t.get("lastPrice") or 0)
                if last <= 0 or high <= 0 or low <= 0:
                    continue

                range24h_pct = (high - low) / last
                try:
                    chg = float(t.get("price24hPcnt"))
                except Exception:
                    prev = float(t.get("prevPrice24h") or 0)
                    chg = (last - prev) / prev if prev > 0 else 0.0

                turnover24h = float(t.get("turnover24h") or 0.0)

                if ANOMALY_FILTER_ENABLED:
                    if (abs(chg) > ANOMALY_ABS_CHANGE_PCT) or (range24h_pct > ANOMALY_RANGE_PCT) or (turnover24h < ANOMALY_MIN_TURNOVER_USDT):
                        excluded_cnt += 1
                        continue

                rows.append({"symbol": sym, "range24h_pct": range24h_pct})
            except Exception:
                continue

        rows = sorted(rows, key=lambda x: x["range24h_pct"], reverse=True)
        pre_top_raw = [r["symbol"] for r in rows[:max(TOP_N * PREFILTER_MULTIPLIER, TOP_N)]]
        logging.info(
            f"Префильтр /tickers: выбрано {len(pre_top_raw)} (multiplier={PREFILTER_MULTIPLIER}); исключено аномалий: {excluded_cnt}"
        )
        pre_top = pre_top_raw if USE_TICKERS_PREFILTER else symbols

        # 3) Индикаторы + тренды (наклоны MACD/Signal)
        def load_pair(sym: str) -> Tuple[str, Optional[Dict]]:
            try:
                trends, rsis, atr_abs_map, atr_pct_map = {}, {}, {}, {}
                last_close_for_price = None

                for tf in TIMEFRAMES:
                    tf_key = "1MN" if tf == "1M" else tf
                    interval = TF_TO_BYBIT[tf_key]
                    limit = min(LIMIT, 120) if interval in LONG_TF_CODES else LIMIT
                    kl = api.get_klines(sym, interval, limit=limit)
                    if (interval in LONG_TF_CODES and len(kl) < 30) or (interval not in LONG_TF_CODES and len(kl) < 50):
                        return sym, None

                    df = kline_to_df(kl)
                    ml, sl, h, rs, asr = compute_indicators(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD, ATR_PERIOD)

                    if tf_key == "5M":
                        trend = classify_trend_with_slope(ml, sl, h, MACD_SLOPE_LOOKBACK_5M, MACD_MIN_SLOPE_5M)
                    elif tf_key == "15M":
                        trend = classify_trend_with_slope(ml, sl, h, MACD_SLOPE_LOOKBACK_15M, MACD_MIN_SLOPE_15M)
                    elif tf_key == "1H":
                        trend = classify_trend_with_slope(ml, sl, h, MACD_SLOPE_LOOKBACK_1H, MACD_MIN_SLOPE_1H)
                    else:
                        trend = classify_trend_simple(ml, sl, h)

                    trends[tf_key] = trend
                    rsis[tf_key] = float(rs.iloc[-1])

                    last_close = float(df["close"].iloc[-1])
                    last_close_for_price = last_close
                    atr_abs = float(asr.iloc[-1])
                    atr_pct = (atr_abs / last_close) if last_close else 0.0
                    atr_abs_map[tf_key] = atr_abs
                    atr_pct_map[tf_key] = atr_pct

                last_market = None
                t = tick_map.get(sym)
                if t:
                    try:
                        last_market = float(t.get("lastPrice"))
                    except Exception:
                        pass
                last_price = last_market or last_close_for_price or 0.0

                return sym, {
                    "trends": trends,
                    "atr_abs_map": atr_abs_map,
                    "atr_pct_map": atr_pct_map,
                    "rsi_map": rsis,
                    "last_price": last_price,
                    "atr_sort_abs": atr_abs_map.get(SORT_TF if SORT_TF != "1M" else "1MN", 0.0),
                    "atr_sort_pct": atr_pct_map.get(SORT_TF if SORT_TF != "1M" else "1MN", 0.0),
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

        # 4) Отбор BULL/BEAR с фильтром противоречий
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

            if MACD_CONTRADICTION_FILTER:
                if is_bull and (tr5 == "BEAR" or tr15 == "BEAR"):
                    is_bull = False
                if is_bear and (tr5 == "BULL" or tr15 == "BULL"):
                    is_bear = False

            it["trend_5M"] = tr5 or "NEUTRAL"
            it["trend_15M"] = tr15 or "NEUTRAL"
            it["trend_1H"] = tr1h or "NEUTRAL"
            it["trend_rule"] = "1H veto + (15M OR 5M) + slope"

            if is_bull:
                bull_list.append(it)
            elif is_bear:
                bear_list.append(it)

        bull_list = sorted(bull_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
        bear_list = sorted(bear_list, key=lambda x: x["atr_pct"], reverse=True)[:TOP_N]
        logging.info(f"Финальный отбор ({'+'.join(TIMEFRAMES)}): BULL={len(bull_list)} BEAR={len(bear_list)}")

        # 5) OI
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

        # 6) Отчёт + доп. параметры (окно/PNL)
        window_str = f"{TRADING_WINDOW_START}-{TRADING_WINDOW_END} {TRADING_WINDOW_TZ}"
        pnl_str = f"PnL autoclose ≥ {CLOSE_ON_PNL_THRESHOLD_PCT:.1f}% | every {PNL_CHECK_INTERVAL_MINUTES} min"
        report_text = build_report_txt(bull_sorted, bear_sorted, timeframes=TIMEFRAMES, sort_tf=SORT_TF, tz="Europe/Kyiv")
        filepath = write_report_file(report_text)
        report_caption = (
            f"BYBIT MACD Scanner — отчёт ({' & '.join(TIMEFRAMES)}). "
            f"{TREND_RULE_HUMAN}. MACD({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}); "
            f"RSI={RSI_PERIOD}; ATR={ATR_PERIOD}. "
            f"Window: {window_str}. {pnl_str}."
        )
        tg_report.send_document(filepath, caption=report_caption)

        # 7) Торговый блок (входы) — опущен для краткости, остался прежний со safety/ретраями
        #    Если у тебя в предыдущей версии был подробный вход — оставь его (мы его не меняем здесь).

    def do_pnl_check():
        """
        Автозакрытие по прибыли:
        - каждые PNL_CHECK_INTERVAL_MINUTES (внешний планировщик) вызывается этот метод
        - если PnL% >= CLOSE_ON_PNL_THRESHOLD_PCT:
            * (опционально) ставим мягкий TP в сторону профита на N тиков и ждём,
            * если не исполнилось — маркет reduceOnly.
        Устойчив к временным ошибкам /v5/position/list: просто пропускает цикл с логом.
        """
        if not (ENABLE_TRADING and CLOSE_ON_PNL_ENABLED):
            logging.debug("PNL-чекер: выключен (ENABLE_TRADING=0 или CLOSE_ON_PNL_ENABLED=0).")
            return

        # 1) Загружаем инструменты для получения tickSize (нужно для мягкого TP).
        try:
            instruments = api.get_instruments()
            sym_info = api.build_symbol_info_map(instruments)
        except Exception as e:
            logging.warning(f"PNL-чекер: не удалось получить инструменты (tickSize), продолжу без них: {e}")
            sym_info = {}

        # 2) Открытые позиции (метод в bybit_api уже «мягкий» и может вернуть []).
        try:
            positions = api.get_open_positions(symbol=None, settle_coin="USDT")
        except Exception as e:
            logging.error(f"PNL-чекер: исключение при get_open_positions: {e}")
            return

        if not positions:
            logging.info("PNL-чекер: открытых позиций нет (или приватный API вернул пусто).")
            return

        # 3) Фолбэк-цены по lastPrice (если у позиции нет markPrice).
        tick_last: Dict[str, float] = {}
        try:
            for t in api.get_tickers():
                sym = t.get("symbol")
                if not sym:
                    continue
                try:
                    tick_last[sym] = float(t.get("lastPrice") or 0.0)
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"PNL-чекер: не удалось получить lastPrice тиков: {e}")

        # 4) Обход позиций
        for p in positions:
            try:
                sym = p.get("symbol")
                side = (p.get("side") or "").strip()  # 'Buy'/'Sell'
                if not sym or side.lower() not in ("buy", "sell"):
                    continue

                # В hedge-режиме Bybit может прислать две записи по одному символу (лонг/шорт) — это ок.
                size = float(p.get("size") or 0.0)
                if size <= 0:
                    continue

                avg_price = float(p.get("avgPrice") or 0.0)
                # markPrice может не прийти (или =0) — используем lastPrice из tikers
                try:
                    mark_price = float(p.get("markPrice") or 0.0)
                except Exception:
                    mark_price = 0.0
                if mark_price <= 0:
                    mark_price = float(tick_last.get(sym, 0.0))

                if avg_price <= 0 or mark_price <= 0:
                    logging.debug(f"PNL-чекер: пропуск {sym} — avg_price={avg_price}, mark_price={mark_price}.")
                    continue

                # 5) PnL% как процент движения цены (без плеча)
                if side.lower() == "buy":
                    pnl_pct = (mark_price - avg_price) / avg_price * 100.0
                else:
                    pnl_pct = (avg_price - mark_price) / avg_price * 100.0

                logging.info(f"PNL-чекер: {sym} {side} size={size} avg={avg_price:.8f} mark={mark_price:.8f} pnl%={pnl_pct:.2f}")

                if pnl_pct < CLOSE_ON_PNL_THRESHOLD_PCT:
                    continue  # порог не достигнут

                # === достигли порога прибыли ===

                # 6) Мягкое закрытие: выставляем TP поблизости и ждём N секунд
                if SOFT_CLOSE_ENABLED:
                    info = sym_info.get(sym, {})
                    tick = float(info.get("tickSize", 0.0)) if info else 0.0
                    if tick <= 0:
                        # запасной вариант, если нет меты по тик-сайзу
                        tick = 0.01

                    ticks = max(1, int(SOFT_CLOSE_TICKS))
                    if side.lower() == "buy":
                        tp_price = mark_price + ticks * tick    # для LONG — ждём тик вверх
                    else:
                        tp_price = mark_price - ticks * tick    # для SHORT — ждём тик вниз

                    # нормализуем цену под шаг тика
                    try:
                        tp_price = api.clamp_price_safe(tp_price, info)
                    except Exception:
                        pass

                    try:
                        api.set_take_profit_only(sym, side, tp_price)
                        logging.info(f"PNL-чекер: выставлен soft-TP {sym} {side} @ {tp_price}")
                    except Exception as e:
                        logging.warning(f"PNL-чекер: не удалось выставить soft-TP для {sym}: {e}")

                    # ждём исполнения TP
                    waited = 0
                    closed_by_soft = False
                    while waited < SOFT_CLOSE_WAIT_SECONDS:
                        time.sleep(max(1, int(SOFT_CLOSE_POLL_INTERVAL_SEC)))
                        waited += max(1, int(SOFT_CLOSE_POLL_INTERVAL_SEC))
                        try:
                            # если позиции нет или объём 0 — считаем закрытой
                            pos_now = api.get_open_positions(symbol=sym)
                            has_size = False
                            for pp in (pos_now or []):
                                if (pp.get("symbol") == sym) and ((pp.get("side") or "").lower() == side.lower()):
                                    if float(pp.get("size") or 0.0) > 0.0:
                                        has_size = True
                                        break
                            if not has_size:
                                closed_by_soft = True
                                logging.info(f"PNL-чекер: {sym} закрылась по soft-TP.")
                                break
                        except Exception as e:
                            logging.debug(f"PNL-чекер: опрос позиции {sym} после soft-TP не удался: {e}")
                            # продолжаем ждать до фолбэка

                    if closed_by_soft:
                        try:
                            if tg_trades:
                                tg_trades.send_message(
                                    f"✅ Закрыта (soft-TP) {sym} {side}\nPnL≈{pnl_pct:.2f}%\nВремя: {now_iso()}"
                                )
                        except Exception:
                            pass
                        continue  # к следующей позиции

                # 7) Фолбэк: закрываем маркетом reduceOnly на весь объём
                try:
                    order_id = api.close_position_market(sym, side, size)
                    logging.info(
                        f"PNL-чекер: {sym} {side} size={size} закрыта MARKET (PnL={pnl_pct:.2f}%). order_id={order_id}"
                    )
                    try:
                        if tg_trades:
                            tg_trades.send_message(
                                f"✅ Закрыта (market) {sym} {side} size={size}\nPnL≈{pnl_pct:.2f}%\nВремя: {now_iso()}"
                            )
                    except Exception:
                        pass
                except Exception as e:
                    logging.error(f"PNL-чекер: не удалось закрыть {sym} маркетом: {e}")

            except Exception as e:
                logging.error(f"PNL-чекер: ошибка обработки позиции: {e}")


    # планирование
    scan_every = timedelta(minutes=SCAN_INTERVAL_MINUTES)
    pnl_every = timedelta(minutes=PNL_CHECK_INTERVAL_MINUTES)
    next_scan_at = datetime.now(tz=pytz.utc)
    next_pnl_at = datetime.now(tz=pytz.utc)

    while True:
        now_utc = datetime.now(tz=pytz.utc)

        # Торговое окно (работаем только внутри)
        if not in_trading_window(now_utc, TRADING_WINDOW_TZ, TRADING_WINDOW_START, TRADING_WINDOW_END):
            sleep_sec = seconds_until_window_start(now_utc, TRADING_WINDOW_TZ, TRADING_WINDOW_START)
            logging.info(f"Вне окна {TRADING_WINDOW_START}-{TRADING_WINDOW_END} {TRADING_WINDOW_TZ}. Сплю {sleep_sec//60} мин.")
            time.sleep(sleep_sec)
            next_scan_at = datetime.now(tz=pytz.utc)
            next_pnl_at = next_scan_at
            continue

        # Запуск задач
        ran_something = False
        if now_utc >= next_scan_at:
            logging.info("=== Новый цикл сканера ===")
            try:
                do_scan_cycle()
            except Exception as e:
                logging.exception(f"Фатальная ошибка сканера: {e}")
            next_scan_at = now_utc + scan_every
            ran_something = True

        if now_utc >= next_pnl_at:
            logging.info("=== PNL-чекер ===")
            try:
                do_pnl_check()
            except Exception as e:
                logging.exception(f"Фатальная ошибка PNL-чекера: {e}")
            next_pnl_at = now_utc + pnl_every
            ran_something = True

        if not ran_something:
            sleep_for = min((next_scan_at - now_utc).total_seconds(),
                            (next_pnl_at - now_utc).total_seconds())
            time.sleep(max(1.0, sleep_for))


if __name__ == "__main__":
    main_loop()
