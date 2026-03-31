# ⚠️ SECURITY NOTE:
# Do NOT post your Deriv / Telegram tokens publicly.
# Paste them only on your local machine.

import asyncio
import logging
import random
import time
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

# Use a single synthetic index for now (you can add more)
MARKETS = ["R_75", "R_100"]   # Volatility 75 and 100

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 60
MAX_CONSEC_LOSSES = 10

# Telegram token from crypto bot
TELEGRAM_TOKEN = "8697638086:AAG00D0RXUAqXFTjy8-4XO4Bka2kBamo-VA"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
DURATION_MIN = 2  # 2-minute expiry

# Timeframes (in seconds)
TF_15M = 15 * 60
TF_5M = 5 * 60
TF_1M = 60

CANDLES_15M = 100   # enough for EMA50 and ADX
CANDLES_5M = 100
CANDLES_1M = 100

# EMA periods
EMA_TREND_PERIOD = 50     # 15M EMA50
EMA_PULLBACK_PERIOD = 20  # 5M EMA20

# Pullback zone multiplier (ATR)
PULLBACK_ATR_MULT = 0.75

# Rejection candle threshold: wick must be at least this fraction of total range
REJECTION_WICK_MIN = 0.3
REJECTION_CLOSE_MOVE = 0.2   # close must be this fraction of range away from wick

# Entry confirmation
MIN_BODY_RATIO = 0.50
VOLUME_MULT = 1.2
RSI_PERIOD = 7       # short period for quick momentum

# ADX filter (optional)
ADX_PERIOD = 14
ADX_MIN = 25.0

# ========================= RISK MANAGEMENT =========================
DAILY_PROFIT_TARGET = 2.0
SECTION_PROFIT_TARGET = 1
MAX_STAKE_ALLOWED = 10.00
MARTINGALE_MULT = 1.8
MARTINGALE_MAX_STEPS = 4
MARTINGALE_MAX_STAKE = 16.0

# ========================= SECTIONS =========================
SECTIONS_PER_DAY = 4
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)

# ========================= ANTI RATE-LIMIT =========================
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20
STATUS_REFRESH_COOLDOWN_SEC = 10

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
    # Simplified ADX; returns the last value or None
    # (full implementation from original code can be reused, but we'll keep it simple for now)
    # For brevity, we'll just compute using the existing function; but to avoid duplicating, we can use a simpler version.
    # Since the original code had a full ADX, we'll use that.
    # We'll import the existing function from the original code? But we're rewriting. Let's include a minimal ADX.
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
    """Convert ticks_history candles to list of dicts."""
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

def money2(x: float) -> float:
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

        self.active_trade_info = None
        self.active_market = None
        self.trade_start_time = 0.0

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.current_stake = 0.0
        self.martingale_step = 0
        self.martingale_halt = False

        # sections
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1
        self.section_pause_until = 0.0

        self.trade_lock = asyncio.Lock()

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_ts = {m: {TF_1M: 0, TF_5M: 0, TF_15M: 0} for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        # anti rate-limit state
        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {m: 0.0 for m in MARKETS}
        self._rate_limit_strikes = {m: 0 for m in MARKETS}

        self.status_cooldown_until = 0.0

    # ---------- helpers ----------
    @staticmethod
    def _is_gatewayish_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(k in m for k in [
            "gateway", "bad gateway", "502", "503", "504",
            "timeout", "timed out", "temporarily unavailable",
            "connection", "websocket", "not connected", "disconnect",
            "internal server error", "service unavailable"
        ])

    @staticmethod
    def _is_rate_limit_error(msg: str) -> bool:
        m = (msg or "").lower()
        return ("rate limit" in m) or ("reached the rate limit" in m) or ("too many requests" in m) or ("429" in m)

    async def safe_send_tg(self, text: str, retries: int = 5):
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

    # ---------- Sections ----------
    def _today_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def _get_section_index_for_epoch(self, epoch_ts: float) -> int:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        return int(idx0 + 1)

    def _next_section_start_epoch(self, epoch_ts: float) -> float:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        next_start = midnight + (idx0 + 1) * SECTION_LENGTH_SEC
        if idx0 + 1 >= SECTIONS_PER_DAY:
            next_midnight = (datetime.fromtimestamp(midnight, self.tz) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return next_midnight.timestamp()
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

    # ---------- Deriv connection ----------
    async def connect(self) -> bool:
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

    async def safe_reconnect(self) -> bool:
        try:
            if self.api:
                await self.api.disconnect()
        except Exception:
            pass
        self.api = None
        return await self.connect()

    async def safe_deriv_call(self, fn_name: str, payload: dict, retries: int = 6):
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

    async def safe_ticks_history(self, payload: dict, retries: int = 4):
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

    # ---------- Daily reset ----------
    def _next_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight.timestamp()

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

        self._sync_section_if_needed()

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= -2.0:
            self.pause_until = self._next_midnight_epoch()
            return False, "Stopped: Daily loss limit (-$2.00) reached"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: max loss streak reached"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if self.active_trade_info:
            return False, "Trade in progress"
        if not self.api:
            return False, "Not connected"
        return True, "OK"

    # ---------- Fetch candles for a timeframe ----------
    async def fetch_candles(self, symbol: str, tf_seconds: int, count: int):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "candles",
            "granularity": tf_seconds,
        }
        data = await self.safe_ticks_history(payload, retries=4)
        return build_candles(data)

    # ---------- Core signal builder ----------
    def evaluate_signal(self, symbol, candles_15m, candles_5m, candles_1m):
        """Returns 'CALL', 'PUT', or None with debug info."""
        dbg = {}

        # 1. 15M trend: EMA50
        closes_15 = [c["c"] for c in candles_15m]
        if len(closes_15) < EMA_TREND_PERIOD + 5:
            return None, "Not enough 15M data", dbg
        ema50_15 = ema(closes_15, EMA_TREND_PERIOD)[-1]
        price_15 = closes_15[-1]
        trend_up = price_15 > ema50_15
        trend_down = price_15 < ema50_15
        dbg["trend_15m"] = "UP" if trend_up else "DOWN" if trend_down else "SIDE"
        dbg["ema50_15"] = round(ema50_15, 5)

        # ADX on 15M (optional, but adds strength)
        if len(candles_15m) > ADX_PERIOD*2:
            highs_15 = [c["h"] for c in candles_15m]
            lows_15 = [c["l"] for c in candles_15m]
            adx_vals = adx(highs_15, lows_15, closes_15, ADX_PERIOD)
            if len(adx_vals) > 0 and not np.isnan(adx_vals[-1]):
                adx_val = adx_vals[-1]
                dbg["adx"] = round(adx_val, 2)
                if adx_val < ADX_MIN:
                    return None, f"ADX too low ({adx_val:.1f} < {ADX_MIN})", dbg
            else:
                dbg["adx"] = "N/A"

        # 2. 5M pullback to EMA20
        closes_5 = [c["c"] for c in candles_5m]
        if len(closes_5) < EMA_PULLBACK_PERIOD + 5:
            return None, "Not enough 5M data", dbg
        ema20_5 = ema(closes_5, EMA_PULLBACK_PERIOD)[-1]
        # ATR for volatility zone
        highs_5 = [c["h"] for c in candles_5m]
        lows_5 = [c["l"] for c in candles_5m]
        atr_vals = atr(highs_5, lows_5, closes_5, 14)
        if len(atr_vals) < 2 or np.isnan(atr_vals[-1]):
            return None, "ATR not ready", dbg
        atr_5 = atr_vals[-1]
        pullback_zone = atr_5 * PULLBACK_ATR_MULT

        # Last closed 5M candle (index -2 because last is forming)
        if len(candles_5m) < 3:
            return None, "Not enough 5M candles", dbg
        pb_candle = candles_5m[-2]
        pb_low = pb_candle["l"]
        pb_high = pb_candle["h"]
        pb_close = pb_candle["c"]
        pb_open = pb_candle["o"]

        # Check if price touched the EMA20 zone
        touches_zone = abs(pb_low - ema20_5) <= pullback_zone or abs(pb_high - ema20_5) <= pullback_zone
        dbg["touches_zone"] = touches_zone

        # Rejection detection
        # For long: lower wick touches zone and close is above the wick by at least REJECTION_CLOSE_MOVE fraction of candle range
        candle_range = pb_high - pb_low
        if candle_range == 0:
            return None, "Zero range candle", dbg
        lower_wick = min(pb_open, pb_close) - pb_low
        upper_wick = pb_high - max(pb_open, pb_close)
        rejection_long = (lower_wick / candle_range >= REJECTION_WICK_MIN and
                          (pb_close - pb_low) / candle_range >= REJECTION_CLOSE_MOVE)
        rejection_short = (upper_wick / candle_range >= REJECTION_WICK_MIN and
                           (pb_high - pb_close) / candle_range >= REJECTION_CLOSE_MOVE)
        dbg["rejection_long"] = rejection_long
        dbg["rejection_short"] = rejection_short

        # 3. 1M confirmation
        if len(candles_1m) < 20:
            return None, "Not enough 1M data", dbg
        conf_candle = candles_1m[-2]   # last closed 1M candle
        conf_open = conf_candle["o"]
        conf_close = conf_candle["c"]
        conf_high = conf_candle["h"]
        conf_low = conf_candle["l"]
        conf_vol = conf_candle.get("v", 0)
        conf_body = abs(conf_close - conf_open)
        conf_range = conf_high - conf_low
        if conf_range == 0:
            return None, "Zero range candle", dbg
        body_ratio = conf_body / conf_range
        strong_candle = body_ratio >= MIN_BODY_RATIO
        dbg["body_ratio"] = round(body_ratio, 2)
        dbg["strong_candle"] = strong_candle

        # Volume spike
        volumes = [c.get("v", 0) for c in candles_1m[-21:-1]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 0
        vol_ok = conf_vol > avg_vol * VOLUME_MULT
        dbg["vol_ok"] = vol_ok

        # RSI on 1M closes (short period)
        closes_1 = [c["c"] for c in candles_1m]
        rsi_vals = rsi(closes_1, RSI_PERIOD)
        if len(rsi_vals) < 3 or np.isnan(rsi_vals[-2]):
            return None, "RSI not ready", dbg
        rsi_val = rsi_vals[-2]
        dbg["rsi"] = round(rsi_val, 2)

        # 4. Combine signals
        # CALL (long) conditions:
        # - 15M uptrend
        # - 5M price touched EMA20 zone
        # - 5M rejection long candle
        # - 1M strong bullish candle, volume spike, RSI > 50
        if (trend_up and touches_zone and rejection_long and
            strong_candle and conf_close > conf_open and vol_ok and rsi_val > 50):
            return "CALL", "All conditions met", dbg

        # PUT (short) conditions:
        # - 15M downtrend
        # - 5M price touched EMA20 zone
        # - 5M rejection short candle
        # - 1M strong bearish candle, volume spike, RSI < 50
        if (trend_down and touches_zone and rejection_short and
            strong_candle and conf_close < conf_open and vol_ok and rsi_val < 50):
            return "PUT", "All conditions met", dbg

        # No signal
        reason = f"15M trend: {'up' if trend_up else 'down' if trend_down else 'side'}, "
        reason += f"touches zone: {touches_zone}, "
        reason += f"rejection: {'long' if rejection_long else 'short' if rejection_short else 'none'}, "
        reason += f"1M: {'strong' if strong_candle else 'weak'}, vol_ok: {vol_ok}, RSI: {rsi_val:.1f}"
        return None, reason, dbg

    # ---------- Scanner loop ----------
    async def scan_market(self, symbol: str):
        self._next_poll_epoch[symbol] = time.time() + random.random() * 0.5
        while self.is_scanning:
            try:
                now = time.time()
                nxt = self._next_poll_epoch.get(symbol, 0.0)
                if now < nxt:
                    await asyncio.sleep(min(1.0, nxt - now))
                    continue

                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                ok_gate, gate = self.can_auto_trade()
                if not ok_gate:
                    self.market_debug[symbol] = {"time": now, "gate": gate, "signal": None, "why": [gate]}
                    self._next_poll_epoch[symbol] = now + 5
                    continue

                # Fetch candles for all three timeframes
                c15 = await self.fetch_candles(symbol, TF_15M, CANDLES_15M)
                await asyncio.sleep(0.1)  # small delay between requests
                c5 = await self.fetch_candles(symbol, TF_5M, CANDLES_5M)
                await asyncio.sleep(0.1)
                c1 = await self.fetch_candles(symbol, TF_1M, CANDLES_1M)

                if not c15 or not c5 or not c1:
                    self.market_debug[symbol] = {"time": now, "gate": gate, "signal": None, "why": ["Missing candles"]}
                    self._next_poll_epoch[symbol] = now + 10
                    continue

                # Evaluate signal
                signal, reason, dbg = self.evaluate_signal(symbol, c15, c5, c1)
                dbg["time"] = now
                dbg["gate"] = gate
                dbg["why"] = reason
                dbg["signal"] = signal
                self.market_debug[symbol] = dbg

                # If signal, execute trade
                if signal:
                    await self.execute_trade(signal, symbol, source="AUTO", rsi_now=dbg.get("rsi", 50))

                # Next poll: wait until next 1M candle close (approximately)
                # We'll schedule at the start of the next minute
                next_minute = int(time.time() // 60 * 60) + 60
                self._next_poll_epoch[symbol] = next_minute + 0.5

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error ({symbol}): {e}")
                self.market_debug[symbol] = {"time": time.time(), "error": str(e), "signal": None}
                await asyncio.sleep(5)

    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time() - self.trade_start_time > (DURATION_MIN * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    # ========================= TRADE EXECUTION (unchanged from original) =========================
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL",
                            rsi_now: float = 0.0, ema50_slope: float = 0.0):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                import math
                payout = float(DAILY_PROFIT_TARGET) * (MARTINGALE_MULT ** self.martingale_step)
                payout = money2(payout)
                payout = max(0.01, payout)
                if not math.isfinite(payout):
                    payout = 0.01
                payout = max(0.35, payout)
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
                    await self.safe_send_tg(
                        f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}"
                    )
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": ask_price}, retries=6)
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                    return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🤖 Source: {source}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}"
                )
                await self.safe_send_tg(msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source, side, rsi_now, ema50_slope))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")

    async def check_result(self, cid: int, source: str, side: str, rsi_now: float, ema50_slope: float):
        await asyncio.sleep(DURATION_MIN * 60 + 5)
        try:
            res = await self.safe_deriv_call(
                "proposal_open_contract",
                {"proposal_open_contract": 1, "contract_id": cid},
                retries=6,
            )
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                if self.section_profit >= SECTION_PROFIT_TARGET:
                    self.sections_won_today += 1
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False
                    if self.consecutive_losses >= 3:
                        self.section_pause_until = self._next_section_start_epoch(time.time())
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps" if self.martingale_halt else ""
            section_note = (
                f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}"
                if time.time() < self.section_pause_until
                else ""
            )

            next_payout = money2(DAILY_PROFIT_TARGET * (MARTINGALE_MULT ** self.martingale_step))

            await self.safe_send_tg(
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} | Sections won: {self.sections_won_today}\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                    f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                    f"💰 Balance: {self.balance}"
                    f"{pause_note}{section_note}{halt_note}"
                )
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivMultiTFBot()

def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
                InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN"),
            ],
            [
                InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
                InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS"),
            ],
            [InlineKeyboardButton("🧩 SECTION", callback_data="NEXT_SECTION")],
            [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
            [
                InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
                InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL"),
            ],
        ]
    )

def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"
    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    signal = d.get("signal") or "—"
    why = d.get("why", "")
    adx = d.get("adx", "—")
    rsi = d.get("rsi", "—")
    body_ratio = d.get("body_ratio", "—")
    touches_zone = d.get("touches_zone", False)
    rejection_long = d.get("rejection_long", False)
    rejection_short = d.get("rejection_short", False)
    strong_candle = d.get("strong_candle", False)
    vol_ok = d.get("vol_ok", False)

    lines = [
        f"📍 {sym.replace('_',' ')} ({age}s)",
        f"Gate: {gate}",
        f"ADX: {adx} | RSI: {rsi}",
        f"Body: {body_ratio} | Vol spike: {'✅' if vol_ok else '❌'}",
        f"Touches zone: {'✅' if touches_zone else '❌'}",
        f"Rejection: {'LONG' if rejection_long else 'SHORT' if rejection_short else '—'}",
        f"Strong candle: {'✅' if strong_candle else '❌'}",
        f"Signal: {signal}",
        f"{why[:100]}"
    ]
    return "\n".join(lines)

async def _safe_answer(q, text: str | None = None, show_alert: bool = False):
    try:
        await q.answer(text=text, show_alert=show_alert)
    except Exception as e:
        logger.warning(f"Callback answer ignored: {e}")

async def _safe_edit(q, text: str, reply_markup=None):
    try:
        await q.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.warning(f"Edit failed: {e}")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await _safe_answer(q)
    await _safe_edit(q, "⏳ Working...", reply_markup=main_keyboard())

    if q.data == "SET_DEMO":
        bot_logic.active_token, bot_logic.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())

    elif q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())
        await _safe_edit(q, "🔍 SCANNER ACTIVE\n✅ Press STATUS to monitor.", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done():
            bot_logic.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "NEXT_SECTION":
        bot_logic._daily_reset_if_needed()
        now = time.time()
        nxt = bot_logic._next_section_start_epoch(now)
        if nxt <= now + 1:
            nxt = now + 1

        forced_idx = bot_logic._get_section_index_for_epoch(nxt + 1)
        bot_logic.section_index = forced_idx
        bot_logic.section_profit = 0.0
        bot_logic.section_pause_until = 0.0

        await _safe_edit(
            q,
            f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.",
            reply_markup=main_keyboard(),
        )

    elif q.data == "TEST_BUY":
        asyncio.create_task(bot_logic.execute_trade("CALL", "R_75", "Manual Test", source="MANUAL"))
        await _safe_edit(q, "🧪 Test trade triggered (CALL R_75).", reply_markup=main_keyboard())

    elif q.data == "STATUS":
        now = time.time()
        if now < bot_logic.status_cooldown_until:
            left = int(bot_logic.status_cooldown_until - now)
            await _safe_edit(q, f"⏳ Refresh cooldown: {left}s\n\nPress again after cooldown.", reply_markup=main_keyboard())
            return
        bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC

        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.safe_deriv_call(
                    "proposal_open_contract",
                    {"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info},
                    retries=4,
                )
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = "⏸ Paused until 12:00am WAT\n" if time.time() < bot_logic.pause_until else ""
        section_line = f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n" if time.time() < bot_logic.section_pause_until else ""

        next_payout = money2(DAILY_PROFIT_TARGET * (MARTINGALE_MULT ** bot_logic.martingale_step))

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}{section_line}"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | Section PnL: {bot_logic.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f} | Sections won: {bot_logic.sections_won_today}\n"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (FULL)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await _safe_edit(q, header + details, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Multi-Timeframe Bot\n"
        f"🕯 Timeframes: 15M (EMA50) → 5M (pullback) → 1M (entry)\n"
        f"⏱ Expiry: {DURATION_MIN}m\n"
        "✅ Strong trend + pullback + volume + RSI confirmation\n"
        "✅ Anti-rate-limit enabled\n",
        reply_markup=main_keyboard(),
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
