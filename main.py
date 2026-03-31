"""
Deriv Synthetic Indices Bot – 3‑Timeframe Strategy (Enhanced)
- 15M: EMA50 trend + ADX ≥ 25
- 5M: EMA20 pullback (within 0.5× ATR) + rejection candle
- 1M: strong candle (body ≥55%) + volume spike (>1.2×) + close beyond 5M extreme
- Risk: martingale (2×, max 3 steps), stop after 5 consecutive losses, daily/section caps
- Paper mode enabled (DEMO)
"""

import asyncio
import logging
import random
import time
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "ZkOFWOlPtwnjqTS"
APP_ID = 1089

# Trading symbols
MARKETS = ["R_50", "R_75", "R_100"]

# Risk limits
COOLDOWN_SEC = 60                  # seconds after a trade
MAX_TRADES_PER_DAY = 30
MAX_CONSEC_LOSSES = 5              # stop after 5 consecutive losses
DAILY_PROFIT_TARGET = 10.0
DAILY_LOSS_LIMIT = -8.0
SECTION_PROFIT_TARGET = 3.0        # stop after +$3 in a section
SECTIONS_PER_DAY = 1
SECTION_LENGTH_SEC = 86400

# Telegram (crypto bot token)
TELEGRAM_TOKEN = "8697638086:AAG00D0RXUAqXFTjy8-4XO4Bka2kBamo-VA"
TELEGRAM_CHAT_ID = "7634818949"

# Strategy timeframes
TF_1M = 60
TF_5M = 300
TF_15M = 900
CANDLES_1M = 300
CANDLES_5M = 300
CANDLES_15M = 300

# Indicators
EMA_TREND = 50                     # 15M EMA50
EMA_PULLBACK = 20                  # 5M EMA20
ADX_PERIOD = 14
ADX_MIN = 25.0
ATR_PERIOD = 14
PULLBACK_ATR_MULT = 0.5            # tighter
RSI_PERIOD = 14
RSI_BUY_MIN = 45.0
RSI_SELL_MAX = 55.0

# Entry confirmation
MIN_BODY_RATIO = 0.55
VOLUME_MULT = 1.2

# Martingale
PAYOUT_TARGET = 1.0                # base payout ($)
MARTINGALE_MULT = 2.0
MARTINGALE_MAX_STEPS = 3
MAX_STAKE_ALLOWED = 16.0
MIN_PAYOUT = 0.35

# Anti rate‑limit
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20
STATUS_REFRESH_COOLDOWN_SEC = 10

# Session filter (optional – set to None for 24/7)
ALLOWED_SESSIONS_UTC = None        # or {"LATE_NY"} etc.

# Logging
TRADE_LOG_FILE = "trade_log.jsonl"
STATS_DAYS = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= INDICATOR HELPERS =========================
def ema(values, period):
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    k = 2.0 / (period + 1.0)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i-1] * (1 - k)
    return ema

def rsi(values, period=14):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < period + 2:
        return np.array([])
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_arr = np.full(n, np.nan, dtype=float)
    rsi_arr[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period+1, n):
        gain = gains[i-1]
        loss = losses[i-1]
        avg_gain = (avg_gain * (period-1) + gain) / period
        avg_loss = (avg_loss * (period-1) + loss) / period
        rs = avg_gain / (avg_loss + 1e-12)
        rsi_arr[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_arr

def atr(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        return np.array([])
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_close),
                               np.abs(lows - prev_close)))
    atr_arr = np.full(n, np.nan, dtype=float)
    atr_arr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, n):
        atr_arr[i] = (atr_arr[i-1] * (period-1) + tr[i]) / period
    return atr_arr

def adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period * 2 + 2:
        return np.array([])
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - prev_close),
                               np.abs(lows[1:] - prev_close)))
    tr_s = np.zeros_like(tr)
    plus_s = np.zeros_like(plus_dm)
    minus_s = np.zeros_like(minus_dm)
    tr_s[period-1] = np.sum(tr[:period])
    plus_s[period-1] = np.sum(plus_dm[:period])
    minus_s[period-1] = np.sum(minus_dm[:period])
    for i in range(period, len(tr)):
        tr_s[i] = tr_s[i-1] - (tr_s[i-1]/period) + tr[i]
        plus_s[i] = plus_s[i-1] - (plus_s[i-1]/period) + plus_dm[i]
        minus_s[i] = minus_s[i-1] - (minus_s[i-1]/period) + minus_dm[i]
    plus_di = 100.0 * (plus_s / (tr_s + 1e-12))
    minus_di = 100.0 * (minus_s / (tr_s + 1e-12))
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx_arr = np.full(n, np.nan, dtype=float)
    dx_full = np.full(n, np.nan, dtype=float)
    dx_full[1:] = dx
    start = period * 2
    adx_arr[start] = np.nanmean(dx_full[period:start+1])
    for i in range(start+1, n):
        adx_arr[i] = ((adx_arr[i-1] * (period-1)) + dx_full[i]) / period
    return adx_arr

def build_candles(data):
    out = []
    for x in data.get("candles", []):
        out.append({
            "t0": int(x["epoch"]),
            "o": float(x["open"]),
            "h": float(x["high"]),
            "l": float(x["low"]),
            "c": float(x["close"]),
            "v": float(x.get("volume", 0))
        })
    return out

def session_bucket(epoch_ts):
    dt = datetime.fromtimestamp(epoch_ts, ZoneInfo("UTC"))
    h = dt.hour
    if 0 <= h <= 6:
        return "ASIA"
    elif 7 <= h <= 11:
        return "LONDON"
    elif 12 <= h <= 15:
        return "OVERLAP"
    elif 16 <= h <= 20:
        return "NEWYORK"
    else:
        return "LATE_NY"

def fmt_time_hhmmss(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except:
        return "—"

def fmt_hhmm(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M")
    except:
        return "—"

def money2(x):
    import math
    return math.ceil(float(x) * 100.0) / 100.0

# ========================= BOT CORE =========================
class DerivMultiTFBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}
        self.watchdog_task = None

        self.active_trade_info = None
        self.active_market = None
        self.trade_start_time = 0.0
        self.active_trade_meta = None

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.max_loss_streak_today = 0
        self.hit_5_losses_today = False

        self.current_stake = 0.0
        self.martingale_step = 0
        self.martingale_halt = False

        # Section
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1
        self.section_pause_until = 0.0

        self.trade_lock = asyncio.Lock()
        self._pending_buy = False

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}
        self.last_candle_ts = {m: 0 for m in MARKETS}
        self._empty_candle_strikes = {m: 0 for m in MARKETS}
        self._next_poll_epoch = {m: 0.0 for m in MARKETS}
        self._rate_limit_strikes = {m: 0 for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0

        self.status_cooldown_until = 0.0

        self.trade_records = []
        self._load_trade_log()

    # ---------- stats ----------
    def _load_trade_log(self):
        if not os.path.exists(TRADE_LOG_FILE):
            return
        try:
            with open(TRADE_LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "t" in rec:
                        self.trade_records.append(rec)
            self._prune_trade_records()
        except Exception as e:
            logger.warning(f"Failed to load trade log: {e}")

    def _append_trade_log(self, rec):
        try:
            with open(TRADE_LOG_FILE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write trade log: {e}")

    def _prune_trade_records(self):
        cutoff = time.time() - (STATS_DAYS * 86400)
        self.trade_records = [r for r in self.trade_records if float(r.get("t", 0)) >= cutoff]

    def record_trade_result(self, symbol, open_epoch, profit):
        sess = session_bucket(open_epoch)
        win = 1 if profit > 0 else 0
        rec = {"t": open_epoch, "symbol": symbol, "session": sess, "win": win, "profit": profit}
        self.trade_records.append(rec)
        self._append_trade_log(rec)
        self._prune_trade_records()

    def stats_30d(self):
        self._prune_trade_records()
        by_market = {}
        by_session = {}
        for r in self.trade_records:
            sym = r.get("symbol", "—")
            sess = r.get("session", "—")
            win = r.get("win", 0)
            by_market.setdefault(sym, {"wins": 0, "losses": 0, "trades": 0})
            by_session.setdefault(sess, {"wins": 0, "losses": 0, "trades": 0})
            by_market[sym]["trades"] += 1
            by_session[sess]["trades"] += 1
            if win:
                by_market[sym]["wins"] += 1
                by_session[sess]["wins"] += 1
            else:
                by_market[sym]["losses"] += 1
                by_session[sess]["losses"] += 1
        return by_market, by_session

    # ---------- helpers ----------
    @staticmethod
    def _is_gatewayish_error(msg):
        m = (msg or "").lower()
        return any(k in m for k in ["gateway", "bad gateway", "502", "503", "504", "timeout", "timed out",
                                    "temporarily unavailable", "connection", "websocket", "not connected",
                                    "disconnect", "internal server error", "service unavailable"])

    @staticmethod
    def _is_rate_limit_error(msg):
        m = (msg or "").lower()
        return ("rate limit" in m) or ("reached the rate limit" in m) or ("too many requests" in m) or ("429" in m)

    async def safe_send_tg(self, text, retries=5):
        if not self.app:
            return
        for i in range(1, retries+1):
            try:
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, text)
                return
            except Exception as e:
                if self._is_gatewayish_error(str(e)):
                    await asyncio.sleep(0.8*i + random.random()*0.4)
                else:
                    await asyncio.sleep(0.4*i)
        logger.warning(f"Telegram send failed after retries")

    # ---------- session & limits ----------
    def _today_midnight_epoch(self):
        now = datetime.now(self.tz)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _get_section_index_for_epoch(self, epoch_ts):
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        return idx0 + 1

    def _next_section_start_epoch(self, epoch_ts):
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        next_start = midnight + (idx0 + 1) * SECTION_LENGTH_SEC
        if idx0 + 1 >= SECTIONS_PER_DAY:
            next_midnight = (datetime.fromtimestamp(midnight, self.tz) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            return next_midnight
        return float(next_start)

    def _sync_section_if_needed(self):
        now = time.time()
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            return
        new_idx = self._get_section_index_for_epoch(now)
        if new_idx != self.section_index:
            self.section_index = new_idx
            self.section_profit = 0.0
            self.section_pause_until = 0.0

    def _session_gate_ok(self):
        if ALLOWED_SESSIONS_UTC is None:
            return True, "OK"
        sess = session_bucket(time.time())
        if sess in ALLOWED_SESSIONS_UTC:
            return True, "OK"
        return False, f"Session blocked: {sess} (allowed: {', '.join(sorted(ALLOWED_SESSIONS_UTC))})"

    def _daily_reset_if_needed(self):
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            self.current_day = today
            self.trades_today = 0
            self.total_losses_today = 0
            self.consecutive_losses = 0
            self.total_profit_today = 0.0
            self.cooldown_until = 0.0
            self.pause_until = 0.0
            self.martingale_step = 0
            self.current_stake = 0.0
            self.martingale_halt = False

            self.section_profit = 0.0
            self.sections_won_today = 0
            self.section_index = self._get_section_index_for_epoch(time.time())
            self.section_pause_until = 0.0

            self.max_loss_streak_today = 0
            self.hit_5_losses_today = False

        self._sync_section_if_needed()

    def can_auto_trade(self):
        self._daily_reset_if_needed()

        ok_sess, msg_sess = self._session_gate_ok()
        if not ok_sess:
            return False, msg_sess

        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = (datetime.now(self.tz).replace(hour=0, minute=0, second=0) + timedelta(days=1)).timestamp()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= DAILY_LOSS_LIMIT:
            self.pause_until = (datetime.now(self.tz).replace(hour=0, minute=0, second=0) + timedelta(days=1)).timestamp()
            return False, f"Daily loss limit reached (-${DAILY_LOSS_LIMIT:.2f})"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, f"Stopped: {MAX_CONSEC_LOSSES} consecutive losses reached"

        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Stopped: daily trade limit ({MAX_TRADES_PER_DAY}) reached"

        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"

        if self.active_trade_info:
            return False, "Trade in progress"

        if self._pending_buy:
            return False, "Trade in progress (pending buy)"

        if not self.api:
            return False, "Not connected"

        return True, "OK"

    # ---------- Deriv connection ----------
    async def connect(self):
        try:
            if not self.active_token:
                return False
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def safe_reconnect(self):
        try:
            if self.api:
                await self.api.disconnect()
        except Exception:
            pass
        self.api = None
        return await self.connect()

    async def safe_deriv_call(self, fn_name, payload, retries=6):
        last_err = None
        for attempt in range(1, retries+1):
            try:
                if not self.api:
                    ok = await self.safe_reconnect()
                    if not ok:
                        raise RuntimeError("Reconnect failed")
                fn = getattr(self.api, fn_name)
                return await fn(payload)
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_gatewayish_error(msg):
                    await self.safe_reconnect()
                if self._is_rate_limit_error(msg):
                    await asyncio.sleep(min(20.0, 2.5*attempt + random.random()))
                else:
                    await asyncio.sleep(min(8.0, 0.6*attempt + random.random()*0.5))
        raise last_err

    async def safe_ticks_history(self, payload, retries=4):
        async with self._ticks_lock:
            now = time.time()
            gap = (self._last_ticks_ts + TICKS_GLOBAL_MIN_INTERVAL) - now
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_ticks_ts = time.time()
        return await self.safe_deriv_call("ticks_history", payload, retries=retries)

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except Exception:
            pass

    # ---------- fetch candles ----------
    async def fetch_candles(self, symbol, tf_sec, count):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "candles",
            "granularity": tf_sec,
        }
        data = await self.safe_ticks_history(payload, retries=4)
        return build_candles(data)

    # ---------- health watchdog ----------
    async def health_watchdog(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for sym in MARKETS:
                last_time = self.market_debug.get(sym, {}).get("time", 0)
                if last_time and now - last_time > 120:
                    logger.warning(f"Scanner for {sym} appears dead – restarting")
                    if sym in self.market_tasks and not self.market_tasks[sym].done():
                        self.market_tasks[sym].cancel()
                    self.market_tasks[sym] = asyncio.create_task(self.scan_market(sym))

    # ---------- scanner ----------
    async def scan_market(self, symbol):
        self._next_poll_epoch[symbol] = time.time() + random.random() * 0.5
        while self.is_scanning:
            try:
                now = time.time()
                nxt = self._next_poll_epoch.get(symbol, 0.0)
                if now < nxt:
                    await asyncio.sleep(min(1.0, nxt - now))
                    continue

                # Quick gate check to avoid work if can't trade
                ok_gate, gate = self.can_auto_trade()
                if not ok_gate:
                    self.market_debug[symbol] = {"time": now, "gate": gate, "signal": None, "why": [gate]}
                    self._next_poll_epoch[symbol] = now + 5
                    continue

                # Fetch candles for all three timeframes
                c15 = await self.fetch_candles(symbol, TF_15M, CANDLES_15M)
                await asyncio.sleep(0.1)
                c5 = await self.fetch_candles(symbol, TF_5M, CANDLES_5M)
                await asyncio.sleep(0.1)
                c1 = await self.fetch_candles(symbol, TF_1M, CANDLES_1M)

                # Check candle counts and health
                if len(c15) < 60 or len(c5) < 60 or len(c1) < 30:
                    self._empty_candle_strikes[symbol] += 1
                    if self._empty_candle_strikes[symbol] > 5:
                        logger.warning(f"[{symbol}] too many empty/insufficient candles – reconnecting")
                        await self.safe_reconnect()
                        self._empty_candle_strikes[symbol] = 0
                        self._next_poll_epoch[symbol] = time.time() + 5
                    else:
                        self._next_poll_epoch[symbol] = time.time() + 10
                    continue
                else:
                    self._empty_candle_strikes[symbol] = 0

                # Check candle timestamp advancement (health)
                if c1:
                    latest_ts = c1[-1]["t0"]
                    if latest_ts > self.last_candle_ts.get(symbol, 0):
                        self.last_candle_ts[symbol] = latest_ts
                    else:
                        if time.time() - self.last_candle_ts.get(symbol, time.time()) > 300:
                            logger.warning(f"[{symbol}] No new candle for 5 min – reconnecting")
                            await self.safe_reconnect()
                            self._next_poll_epoch[symbol] = time.time() + 5
                            continue

                confirm = c1[-2]
                confirm_t0 = confirm["t0"]
                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    continue
                self.last_processed_closed_t0[symbol] = confirm_t0

                # ----- 15M analysis -----
                closes_15 = [c["c"] for c in c15]
                highs_15 = [c["h"] for c in c15]
                lows_15 = [c["l"] for c in c15]

                ema50_15 = ema(closes_15, EMA_TREND)[-2]
                price_15 = closes_15[-2]
                trend_up = price_15 > ema50_15
                trend_down = price_15 < ema50_15

                # ADX filter
                adx_vals = adx(highs_15, lows_15, closes_15, ADX_PERIOD)
                if len(adx_vals) < 2 or np.isnan(adx_vals[-2]):
                    self._next_poll_epoch[symbol] = time.time() + 5
                    continue
                adx_val = adx_vals[-2]
                if adx_val < ADX_MIN:
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate, "signal": None,
                                                 "why": [f"ADX too low ({adx_val:.1f} < {ADX_MIN})"], "adx": adx_val}
                    self._next_poll_epoch[symbol] = time.time() + 10
                    continue

                # ----- 5M pullback -----
                closes_5 = [c["c"] for c in c5]
                highs_5 = [c["h"] for c in c5]
                lows_5 = [c["l"] for c in c5]

                ema20_5 = ema(closes_5, EMA_PULLBACK)[-2]
                atr_5 = atr(highs_5, lows_5, closes_5, ATR_PERIOD)[-2]
                price_5 = closes_5[-2]
                rsi_5 = rsi(closes_5, RSI_PERIOD)[-2]

                pullback_zone = atr_5 * PULLBACK_ATR_MULT
                near_ema20 = abs(price_5 - ema20_5) <= pullback_zone
                pullback_buy_ok = near_ema20 and rsi_5 >= RSI_BUY_MIN
                pullback_sell_ok = near_ema20 and rsi_5 <= RSI_SELL_MAX

                # Rejection candle on 5M (optional, we use wick + close away)
                pb_candle_5m = c5[-2]
                pb_high = pb_candle_5m["h"]
                pb_low = pb_candle_5m["l"]
                pb_open = pb_candle_5m["o"]
                pb_close = pb_candle_5m["c"]
                candle_range = pb_high - pb_low
                lower_wick = min(pb_open, pb_close) - pb_low
                upper_wick = pb_high - max(pb_open, pb_close)
                rejection_long = (lower_wick / candle_range >= 0.35) and (pb_close > pb_low + (atr_5 * 0.3))
                rejection_short = (upper_wick / candle_range >= 0.35) and (pb_close < pb_high - (atr_5 * 0.3))

                # ----- 1M confirmation -----
                conf_candle = c1[-2]
                conf_vol = conf_candle.get("v", 0)
                # Body ratio
                conf_body = abs(conf_candle["c"] - conf_candle["o"])
                conf_range = conf_candle["h"] - conf_candle["l"]
                body_ratio = conf_body / conf_range if conf_range > 0 else 0
                strong_candle = body_ratio >= MIN_BODY_RATIO
                # Volume spike
                volumes = [c.get("v", 0) for c in c1[-21:-1]]
                avg_vol = sum(volumes) / len(volumes) if volumes else 0
                vol_ok = conf_vol > avg_vol * VOLUME_MULT
                # Extra confirmation: close beyond 5M extreme
                extra_confirm_buy = conf_candle["c"] > pb_high
                extra_confirm_sell = conf_candle["c"] < pb_low

                # Combine signals
                call_ready = (trend_up and pullback_buy_ok and rejection_long and strong_candle and vol_ok and extra_confirm_buy)
                put_ready = (trend_down and pullback_sell_ok and rejection_short and strong_candle and vol_ok and extra_confirm_sell)

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                # Debug info
                trend_label = "UPTREND" if trend_up else "DOWNTREND" if trend_down else "SIDEWAYS"
                ema_label = f"15M EMA{EMA_TREND}: {ema50_15:.2f}"
                pullback_label = f"5M near EMA20: {'✅' if near_ema20 else '❌'} | RSI: {rsi_5:.1f}"
                confirm_label = f"1M body:{body_ratio:.2f} vol:{conf_vol:.0f}/{avg_vol:.0f} | break5M: {'✅' if extra_confirm_buy or extra_confirm_sell else '❌'}"

                why = []
                if not ok_gate:
                    why.append(f"Gate: {gate}")
                if signal:
                    why.append(f"READY: {signal}")
                else:
                    why.append(f"ADX:{adx_val:.1f}, trend:{trend_label}, nearEMA:{near_ema20}, rej:{rejection_long or rejection_short}, strong:{strong_candle}, vol:{vol_ok}, break:{extra_confirm_buy or extra_confirm_sell}")

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,
                    "trend_label": trend_label,
                    "ema_label": ema_label,
                    "trend_strength": "STRONG" if adx_val >= ADX_MIN else "WEAK",
                    "pullback_label": pullback_label,
                    "confirm_close_label": confirm_label,
                    "block_label": "OK" if signal else "—",
                    "rsi_now": rsi_5,
                    "body_ratio": body_ratio,
                    "adx_now": adx_val,
                    "why": why[:3]
                }

                if call_ready:
                    await self.execute_trade("CALL", symbol, source="AUTO", rsi_now=rsi_5, ema50_slope=0.0)
                elif put_ready:
                    await self.execute_trade("PUT", symbol, source="AUTO", rsi_now=rsi_5, ema50_slope=0.0)

                # Schedule next poll after the next 1M candle close
                next_minute = int(time.time() // 60 * 60) + 60
                self._next_poll_epoch[symbol] = next_minute + 0.5

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")
                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol] = self._rate_limit_strikes.get(symbol, 0) + 1
                    backoff = min(180, RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[symbol])
                    self._next_poll_epoch[symbol] = time.time() + backoff
                else:
                    self._next_poll_epoch[symbol] = time.time() + 10
                    await asyncio.sleep(2 if not self._is_gatewayish_error(msg) else 5)
            await asyncio.sleep(0.05)

    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        asyncio.create_task(self.health_watchdog())
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time() - self.trade_start_time) > (DURATION_MIN * 60 + 90):
                    self.active_trade_info = None
                    self.active_trade_meta = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    # ========================= TRADE EXECUTION =========================
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL", rsi_now=0.0, ema50_slope=0.0):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _ = self.can_auto_trade()
            if not ok:
                return
            if self._pending_buy:
                return

            self._pending_buy = True
            try:
                import math
                payout = float(PAYOUT_TARGET) * (MARTINGALE_MULT ** self.martingale_step)
                payout = money2(payout)
                payout = max(0.01, payout)
                if not math.isfinite(payout):
                    payout = 0.01
                payout = max(MIN_PAYOUT, payout)
                payout = money2(payout)

                proposal_req = {
                    "proposal": 1,
                    "amount": payout,
                    "basis": "payout",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": DURATION_MIN,
                    "duration_unit": "m",
                    "symbol": symbol,
                }

                prop = await self.safe_deriv_call("proposal", proposal_req, retries=6)
                if "error" in prop:
                    err = prop["error"].get("message", "Proposal error")
                    await self.safe_send_tg(f"❌ Proposal Error:\n{err}")
                    return

                p = prop["proposal"]
                proposal_id = p["id"]
                ask_price = float(p.get("ask_price", 0.0))
                if ask_price <= 0:
                    await self.safe_send_tg("❌ Proposal returned invalid ask_price.")
                    return

                if ask_price > MAX_STAKE_ALLOWED:
                    await self.safe_send_tg(f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}")
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": ask_price}, retries=1)
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                    return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                self.active_trade_meta = {"symbol": symbol, "side": side, "open_epoch": self.trade_start_time, "source": source}

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (f"🚀 {side} TRADE OPENED\n"
                       f"🛒 Market: {safe_symbol}\n"
                       f"⏱ Expiry: {DURATION_MIN}m\n"
                       f"🎁 Payout: ${payout:.2f}\n"
                       f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                       f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                       f"🕓 Session (UTC): {session_bucket(self.trade_start_time)}\n"
                       f"🤖 Source: {source}\n"
                       f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}")
                await self.safe_send_tg(msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source, side, rsi_now, ema50_slope))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")
            finally:
                self._pending_buy = False

    async def check_result(self, cid, source, side, rsi_now, ema50_slope):
        await asyncio.sleep(DURATION_MIN * 60 + 5)
        try:
            res = await self.safe_deriv_call("proposal_open_contract", {"proposal_open_contract": 1, "contract_id": cid}, retries=6)
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO" and self.active_trade_meta:
                sym = self.active_trade_meta.get("symbol", "—")
                open_epoch = float(self.active_trade_meta.get("open_epoch", time.time()))
                self.record_trade_result(sym, open_epoch, profit)

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                # Section profit cap
                if self.section_profit >= SECTION_PROFIT_TARGET:
                    self.sections_won_today += 1
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    if self.consecutive_losses > self.max_loss_streak_today:
                        self.max_loss_streak_today = self.consecutive_losses
                    if self.consecutive_losses >= 5 and not self.hit_5_losses_today:
                        self.hit_5_losses_today = True
                        await self.safe_send_tg("⚠️ ALERT: You have hit 5 losses in a row today (at least once).")

                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False   # stop scanning for the day
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                # Daily profit target
                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps" if self.martingale_halt else ""
            section_note = f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}" if time.time() < self.section_pause_until else ""

            next_payout = money2(PAYOUT_TARGET * (MARTINGALE_MULT ** self.martingale_step))

            msg = (f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                   f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
                   f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                   f"📌 Max streak today: {self.max_loss_streak_today} | Hit 5-loss today: {'YES' if self.hit_5_losses_today else 'NO'}\n"
                   f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                   f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                   f"💰 Balance: {self.balance}"
                   f"{pause_note}{section_note}{halt_note}")
            await self.safe_send_tg(msg)

        finally:
            self.active_trade_info = None
            self.active_trade_meta = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= TELEGRAM UI =========================
bot = DerivMultiTFBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧩 SECTION", callback_data="NEXT_SECTION")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
    ])

def format_market_detail(sym, d):
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"
    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    signal = d.get("signal") or "—"
    trend_label = d.get("trend_label", "—")
    pullback_label = d.get("pullback_label", "—")
    confirm_label = d.get("confirm_close_label", "—")
    rsi = d.get("rsi_now", "—")
    adx = d.get("adx_now", "—")
    body = d.get("body_ratio", "—")
    why = " ".join(d.get("why", []))
    return (f"📍 {sym.replace('_',' ')} ({age}s)\n"
            f"Gate: {gate}\n"
            f"ADX: {adx:.1f} | RSI(5M): {rsi:.1f} | Body(1M): {body:.2f}\n"
            f"Trend: {trend_label}\n{pullback_label}\n{confirm_label}\n"
            f"Signal: {signal}\nWhy: {why[:80]}")

async def _safe_answer(q, text=None, show_alert=False):
    try:
        await q.answer(text=text, show_alert=show_alert)
    except Exception as e:
        logger.warning(f"Callback answer ignored: {e}")

async def _safe_edit(q, text, markup=None):
    try:
        await q.edit_message_text(text, reply_markup=markup)
    except Exception as e:
        logger.warning(f"Edit failed: {e}")

async def btn_handler(update, context):
    q = update.callback_query
    await _safe_answer(q)
    await _safe_edit(q, "⏳ Working...", reply_markup=main_keyboard())

    if q.data == "SET_DEMO":
        bot.active_token, bot.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot.connect()
        await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())
    elif q.data == "SET_REAL":
        bot.active_token, bot.account_type = REAL_TOKEN, "LIVE"
        ok = await bot.connect()
        await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())
    elif q.data == "START_SCAN":
        if not bot.api:
            await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
            return
        bot.is_scanning = True
        bot.scanner_task = asyncio.create_task(bot.background_scanner())
        await _safe_edit(q, "🔍 SCANNER ACTIVE\n✅ Press STATUS to monitor.", reply_markup=main_keyboard())
    elif q.data == "STOP_SCAN":
        bot.is_scanning = False
        if bot.scanner_task and not bot.scanner_task.done():
            bot.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())
    elif q.data == "NEXT_SECTION":
        bot._daily_reset_if_needed()
        now = time.time()
        nxt = bot._next_section_start_epoch(now)
        if nxt <= now + 1:
            nxt = now + 1
        forced_idx = bot._get_section_index_for_epoch(nxt + 1)
        bot.section_index = forced_idx
        bot.section_profit = 0.0
        bot.section_pause_until = 0.0
        await _safe_edit(q, f"🧩 Moved to Section {bot.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        test_symbol = MARKETS[0] if MARKETS else "R_50"
        asyncio.create_task(bot.execute_trade("CALL", test_symbol, "Manual Test", source="MANUAL"))
        await _safe_edit(q, f"🧪 Test trade triggered (CALL {test_symbol.replace('_',' ')}).", reply_markup=main_keyboard())
    elif q.data == "STATUS":
        now = time.time()
        if now < bot.status_cooldown_until:
            left = int(bot.status_cooldown_until - now)
            await _safe_edit(q, f"⏳ Refresh cooldown: {left}s", reply_markup=main_keyboard())
            return
        bot.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC

        await bot.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        ok_gate, gate = bot.can_auto_trade()

        trade_status = "No Active Trade"
        if bot.active_trade_info and bot.api:
            try:
                res = await bot.safe_deriv_call("proposal_open_contract", {"proposal_open_contract": 1, "contract_id": bot.active_trade_info}, retries=4)
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, DURATION_MIN*60 - int(time.time() - bot.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt = str(bot.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = "⏸ Paused until midnight\n" if time.time() < bot.pause_until else ""
        section_line = f"🧩 Section paused until {fmt_hhmm(bot.section_pause_until)}\n" if time.time() < bot.section_pause_until else ""

        by_mkt, by_sess = bot.stats_30d()
        def stats_block(title, items):
            if not items:
                return f"{title}: No trades"
            lines = [f"{title} (last {STATS_DAYS}d):"]
            for k, v in sorted(items.items(), key=lambda x: x[1]["trades"], reverse=True):
                total = v["trades"]
                wins = v["wins"]
                wr = (wins/total*100) if total else 0
                lines.append(f"- {k.replace('_',' ')}: {wr:.1f}% ({wins}/{total})")
            return "\n".join(lines)

        stats_txt = stats_block("Markets", by_mkt) + "\n" + stats_block("Sessions(UTC)", by_sess)

        header = (f"🕒 {now_time}\n🤖 Bot: {'ACTIVE' if bot.is_scanning else 'OFFLINE'} ({bot.account_type})\n"
                  f"{pause_line}{section_line}"
                  f"🧩 Section: {bot.section_index}/{SECTIONS_PER_DAY} | PnL: {bot.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
                  f"🎲 Martingale step: {bot.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                  f"🎯 Daily: +{bot.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                  f"📊 Trades: {bot.trades_today}/{MAX_TRADES_PER_DAY} | Losses: {bot.total_losses_today}\n"
                  f"📉 Streak: {bot.consecutive_losses}/{MAX_CONSEC_LOSSES} | Max today: {bot.max_loss_streak_today}\n"
                  f"🚦 Gate: {gate}\n💰 Balance: {bot.balance}\n"
                  f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
                  f"{stats_txt}\n\n📡 LIVE SCAN\n\n")

        details = "\n\n".join(format_market_detail(sym, bot.market_debug.get(sym, {})) for sym in MARKETS)
        await _safe_edit(q, header + details, reply_markup=main_keyboard())

async def start_cmd(update, context):
    await update.message.reply_text(
        "💎 Deriv 3‑Timeframe Bot (ADX, Strong Entry, Stop after 5 losses)\n"
        f"⏱ Expiry: {DURATION_MIN}m\n"
        f"✅ Daily target +${DAILY_PROFIT_TARGET:.0f} | Section +${SECTION_PROFIT_TARGET:.2f}\n"
        f"✅ Stops after {MAX_CONSEC_LOSSES} losses in a row\n"
        f"✅ Martingale: {MARTINGALE_MULT}× up to {MARTINGALE_MAX_STEPS} steps\n"
        f"✅ Trade with DEMO first!",
        reply_markup=main_keyboard()
    )

# ========================= MAIN =========================
if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
