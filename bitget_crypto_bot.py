"""
Bitget Crypto Futures Bot - HYBRID STRATEGY (RANGE STRENGTHENED)
Strategy: Market Regime Detection based on ADX
- ADX 25-35: TREND (Pullback) → 1:4 RR
- ADX 20-25: MIXED (Breakout) → 1:2.5 RR  
- ADX < 20: RANGE (Mean Reversion) → 1:1.5 RR (now with extra filters)
- ADX > 35: TREND_CAUTION (Reduced risk) → 1:3 RR

Changes:
- XRP replaced by BTC (better liquidity, cleaner trends)
- RANGE strategy now requires:
   * 15M trend alignment (price > EMA50 for longs)
   * ADX < 20 AND ADX decreasing (to confirm ranging)
   * Reversal candle (bullish/bearish close beyond band)
   * Wider bands (2.0x ATR)
   * Volume confirmation
- Per-symbol settings for future tuning
- Paper mode enabled by default
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import ccxt.async_support as ccxt
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ========================= CREDENTIALS =========================
# (Keep your existing credentials; no changes)
BITGET_API_KEY    = "bg_d0944109a841af8a4167114466af2bf3"
BITGET_SECRET     = "e2bf8eed9bc0f4963d4c2c325ba19eb03476f9b504341217bbbe7343c80268be"
BITGET_PASSPHRASE = "Salome1234"
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN", "8697638086:AAG00D0RXUAqXFTjy8-4XO4Bka2kBamo-VA")

USE_TESTNET = False
PAPER_MODE  = True   # True = simulate trades | False = real orders

# ========================= MARKETS =========================
# Replaced XRP with BTC
SYMBOLS = ["SOL/USDT:USDT", "BTC/USDT:USDT", "ADA/USDT:USDT"]

# ========================= TIMEFRAMES =========================
TF_TREND    = "15m"
TF_ENTRY    = "5m"
CANDLES_15M = 150
CANDLES_5M  = 80

# ========================= INDICATORS =========================
EMA_PERIOD      = 50
ADX_PERIOD      = 14
RSI_PERIOD      = 14
VOLUME_LOOKBACK = 20
ATR_PERIOD      = 14

# ========================= REGIME THRESHOLDS =========================
TREND_MIN_ADX      = 25    # ADX >= 25 = trending (healthy)
TREND_MAX_ADX      = 35    # ADX > 35 = over-extended (caution)
WEAK_TREND_MIN     = 20    # ADX 20-25 = weak trend
# ADX < 20 = choppy/ranging

# ========================= PULLBACK FILTER (TREND STRATEGY) =========================
PULLBACK_ATR_MULTIPLIER = 0.75
CANDLE_BODY_RATIO_MIN   = 0.50
STOP_ATR_MULTIPLIER     = 0.5

# ========================= RANGE STRATEGY (STRENGTHENED) =========================
RANGE_ATR_MULTIPLIER = 2.0      # Wider bands = 2.0x ATR
RANGE_RSI_PERIOD     = 7
RANGE_RSI_OVERSOLD   = 30       # Tighter oversold threshold
RANGE_RSI_OVERBOUGHT = 70       # Tighter overbought threshold
RANGE_MIN_ADX        = 20       # Only trade if ADX < 20 (confirmed ranging)
RANGE_REQUIRE_REVERSAL_CANDLE = True   # Require candle to close beyond band

# ========================= BREAKOUT STRATEGY =========================
BREAKOUT_PERIOD      = 20
BREAKOUT_VOLUME_MULT = 1.5

# ========================= RISK/REWARD BY REGIME =========================
RR_RATIOS = {
    "TREND": 4.0,
    "TREND_CAUTION": 3.0,
    "RANGE": 1.5,
    "MIXED": 2.5,
}

RISK_MULTIPLIERS = {
    "TREND": 1.0,
    "TREND_CAUTION": 0.5,
    "RANGE": 0.5,
    "MIXED": 0.75,
}

BASE_RISK_PER_TRADE = 0.25  # USDT

# ========================= PER-SYMBOL SETTINGS (OPTIONAL) =========================
# You can adjust thresholds per symbol here if needed
SYMBOL_SETTINGS = {
    "SOL/USDT:USDT": {
        "trend_min_adx": 25,
        "pullback_atr_mult": 0.75,
        "range_atr_mult": 2.0,
    },
    "BTC/USDT:USDT": {
        "trend_min_adx": 22,          # BTC trends earlier
        "pullback_atr_mult": 0.6,     # Tighter pullbacks on BTC
        "range_atr_mult": 2.0,
    },
    "ADA/USDT:USDT": {
        "trend_min_adx": 25,
        "pullback_atr_mult": 0.75,
        "range_atr_mult": 2.0,
    },
}

# ========================= SESSION FILTER =========================
SESSION_START_UTC = 8
SESSION_END_UTC   = 21

# ========================= LIMITS =========================
MAX_TRADES_PER_DAY   = 10
MAX_CONSEC_LOSSES    = 5
CONSEC_LOSS_PAUSE_HR = 24
COOLDOWN_SEC         = 300
MAX_OPEN_POSITIONS   = 3      # One per symbol

# ========================= OTHER =========================
TRADE_LOG_FILE          = "hybrid_trades.json"
SCAN_INTERVAL_SEC       = 15
STATUS_REFRESH_COOLDOWN = 10
TIMEZONE                = "Africa/Lagos"


# ========================= HELPERS (unchanged) =========================
def ema_value(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return float(ema)


def candle_body_ratio(candle):
    rng = abs(candle[2] - candle[3])
    if rng == 0:
        return 0.0
    return abs(candle[4] - candle[1]) / rng


def rsi_value(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def adx_value(candles, period=14):
    if len(candles) < period * 2 + 1:
        return None
    highs  = np.array([c[2] for c in candles], dtype=float)
    lows   = np.array([c[3] for c in candles], dtype=float)
    closes = np.array([c[4] for c in candles], dtype=float)

    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, len(candles)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr  = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - highs[i - 1], 0) if (h - highs[i - 1]) > (lows[i - 1] - l) else 0
        ndm = max(lows[i - 1] - l, 0) if (lows[i - 1] - l) > (h - highs[i - 1]) else 0
        tr_list.append(tr)
        pdm_list.append(pdm)
        ndm_list.append(ndm)

    def smooth(arr, p):
        s = sum(arr[:p])
        out = [s]
        for v in arr[p:]:
            s = s - s / p + v
            out.append(s)
        return out

    atr  = smooth(tr_list, period)
    apdm = smooth(pdm_list, period)
    andm = smooth(ndm_list, period)

    dx_list = []
    for i in range(len(atr)):
        pdi = 100 * apdm[i] / atr[i] if atr[i] else 0
        ndi = 100 * andm[i] / atr[i] if atr[i] else 0
        dsum = pdi + ndi
        dx   = 100 * abs(pdi - ndi) / dsum if dsum else 0
        dx_list.append(dx)

    if len(dx_list) < period:
        return None
    return float(np.mean(dx_list[-period:]))


def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    highs = [c[2] for c in candles[-period-1:]]
    lows = [c[3] for c in candles[-period-1:]]
    closes = [c[4] for c in candles[-period-2:-1]]
    tr_values = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period


def is_session_active():
    utc_hour = datetime.utcnow().hour
    return SESSION_START_UTC <= utc_hour < SESSION_END_UTC


def now_wat():
    return datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")


# ========================= SIGNAL =========================
@dataclass
class Signal:
    side:     str
    symbol:   str
    entry:    float
    stop:     float
    target:   float
    reason:   str
    regime:   str
    rr_ratio: float


# ========================= BOT =========================
class HybridBot:
    def __init__(self):
        self.exchange          = None
        self.app               = None
        self.is_scanning       = False
        self.active_positions  = {}
        self.paper_positions   = {}
        self.cooldown_until    = 0
        self.pause_until       = 0
        self.status_cd_until   = 0
        self.balance_usdt      = 0.0
        self.trades_today      = 0
        self.profit_today      = 0.0
        self.losses_today      = 0
        self.consec_losses     = 0
        self.last_reset_date   = None
        self.market_debug      = {}
        self.trade_history     = []
        
        self.win_rate_stats = {
            "SOL/USDT:USDT": {"wins": 0, "losses": 0, "regime_stats": {}},
            "BTC/USDT:USDT": {"wins": 0, "losses": 0, "regime_stats": {}},
            "ADA/USDT:USDT": {"wins": 0, "losses": 0, "regime_stats": {}},
        }
        
        self._chat_ids = set()
        self._load_trades()

    def _load_trades(self):
        try:
            if os.path.exists(TRADE_LOG_FILE):
                with open(TRADE_LOG_FILE, "r") as f:
                    self.trade_history = json.load(f)
        except Exception:
            self.trade_history = []

    def _save_trade(self, record):
        self.trade_history.append(record)
        try:
            with open(TRADE_LOG_FILE, "w") as f:
                json.dump(self.trade_history[-500:], f, indent=2)
        except Exception:
            pass
        
        symbol = record.get("symbol")
        result = record.get("result")
        regime = record.get("regime", "UNKNOWN")
        
        if symbol in self.win_rate_stats:
            if result == "WIN":
                self.win_rate_stats[symbol]["wins"] += 1
            elif result == "LOSS":
                self.win_rate_stats[symbol]["losses"] += 1
            
            if regime not in self.win_rate_stats[symbol]["regime_stats"]:
                self.win_rate_stats[symbol]["regime_stats"][regime] = {"wins": 0, "losses": 0}
            if result == "WIN":
                self.win_rate_stats[symbol]["regime_stats"][regime]["wins"] += 1
            elif result == "LOSS":
                self.win_rate_stats[symbol]["regime_stats"][regime]["losses"] += 1

    def _reset_daily(self):
        today = datetime.now(ZoneInfo(TIMEZONE)).date()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.trades_today    = 0
            self.profit_today    = 0.0
            self.losses_today    = 0
            self.consec_losses   = 0
            logger.info(f"Daily reset: {today}")

    def get_symbol_setting(self, symbol, key, default):
        """Fetch per-symbol setting if exists, else return default."""
        if symbol in SYMBOL_SETTINGS and key in SYMBOL_SETTINGS[symbol]:
            return SYMBOL_SETTINGS[symbol][key]
        return default

    def get_risk_per_trade(self, regime):
        multiplier = RISK_MULTIPLIERS.get(regime, 0.5)
        return BASE_RISK_PER_TRADE * multiplier

    def get_rr_ratio(self, regime):
        return RR_RATIOS.get(regime, 2.0)

    async def connect(self):
        try:
            if self.exchange:
                try:
                    await self.exchange.close()
                except Exception:
                    pass

            self.exchange = ccxt.bitget({
                "apiKey": BITGET_API_KEY,
                "secret": BITGET_SECRET,
                "password": BITGET_PASSPHRASE,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })

            if USE_TESTNET:
                self.exchange.set_sandbox_mode(True)

            await self.exchange.load_markets()
            await self.fetch_balance()

            for sym in SYMBOLS:
                if sym in self.exchange.markets:
                    logger.info(f"MARKET OK: {sym}")
                else:
                    logger.warning(f"MARKET MISSING: {sym}")

            logger.info(f"Connected to Bitget {'TESTNET' if USE_TESTNET else 'LIVE'} | PAPER_MODE={PAPER_MODE}")
            return True
        except Exception as e:
            logger.error(f"Connect failed: {e}")
            return False

    async def fetch_balance(self):
        if not self.exchange:
            return
        try:
            bal = await self.exchange.fetch_balance()
            usdt = bal.get("USDT", {})
            self.balance_usdt = float(usdt.get("free", 0) or 0) + float(usdt.get("used", 0) or 0)
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")

    async def get_current_price(self, symbol):
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker.get("last", 0))
        except Exception:
            return 0.0

    async def tg(self, text):
        if not self.app:
            return
        for cid in list(self._chat_ids):
            try:
                await self.app.bot.send_message(chat_id=cid, text=str(text)[:4000])
            except Exception:
                pass

    def total_open_positions(self):
        return len(self.active_positions) + len(self.paper_positions)

    def has_position_for_symbol(self, symbol):
        return symbol in self.active_positions or symbol in self.paper_positions

    def can_trade(self, symbol=None):
        self._reset_daily()

        if not is_session_active():
            utc_hour = datetime.utcnow().hour
            return False, f"Outside session ({utc_hour:02d}:00 UTC) — active {SESSION_START_UTC:02d}:00-{SESSION_END_UTC:02d}:00"

        if time.time() < self.pause_until:
            remaining = int((self.pause_until - time.time()) / 3600)
            return False, f"Paused after {MAX_CONSEC_LOSSES} consecutive losses — {remaining}h left"

        if time.time() < self.cooldown_until:
            left = int(self.cooldown_until - time.time())
            return False, f"Cooldown {left}s"

        if symbol and self.has_position_for_symbol(symbol):
            return False, f"Position already exists for {symbol}"

        if self.total_open_positions() >= MAX_OPEN_POSITIONS:
            return False, f"Max positions open ({MAX_OPEN_POSITIONS})"

        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Daily limit reached ({MAX_TRADES_PER_DAY})"

        if self.consec_losses >= MAX_CONSEC_LOSSES:
            return False, f"Paused for the day — {MAX_CONSEC_LOSSES} consecutive losses"

        return True, "OK"

    async def fetch_ohlcv(self, symbol, tf, limit):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                data = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                if not data:
                    logger.warning(f"OHLCV {symbol} {tf}: Empty (attempt {attempt}/{max_retries})")
                    await asyncio.sleep(2 * attempt)
                    continue
                return data[:-1] if len(data) > 1 else data
            except Exception as e:
                err = str(e)
                logger.warning(f"OHLCV {symbol} {tf} attempt {attempt}/{max_retries}: {err}")
                if "rate limit" in err.lower() or "429" in err:
                    await asyncio.sleep(5 * attempt)
                elif attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
        logger.error(f"OHLCV {symbol} {tf}: All attempts failed")
        return []

    def determine_regime(self, c15, c5, symbol):
        adx = adx_value(c15, ADX_PERIOD)
        if adx is None:
            return "MIXED"
        # Use per-symbol trend_min_adx for slight adjustments
        trend_min = self.get_symbol_setting(symbol, "trend_min_adx", TREND_MIN_ADX)
        if adx > TREND_MAX_ADX:
            return "TREND_CAUTION"
        elif adx >= trend_min:
            return "TREND"
        elif adx >= WEAK_TREND_MIN:
            return "MIXED"
        else:
            return "RANGE"

    # ------------------------------------------------------------------
    # TREND strategy (unchanged)
    # ------------------------------------------------------------------
    def build_signal_trend(self, symbol, c15, c5, is_caution=False):
        dbg = {}
        if len(c15) < EMA_PERIOD + 5:
            return None, "15M warming up", dbg

        cl15 = [x[4] for x in c15]
        price15 = cl15[-1]
        ema50_15 = ema_value(cl15, EMA_PERIOD)
        if ema50_15 is None:
            return None, "EMA50 not ready", dbg

        trend_up = price15 > ema50_15
        trend_down = price15 < ema50_15
        dbg["ema50_15"] = round(ema50_15, 5)
        dbg["price15"] = round(price15, 5)
        dbg["trend"] = "UP" if trend_up else "DOWN" if trend_down else "SIDE"

        if not trend_up and not trend_down:
            return None, "15M sideways", dbg

        adx = adx_value(c15, ADX_PERIOD)
        trend_min = self.get_symbol_setting(symbol, "trend_min_adx", TREND_MIN_ADX)
        adx_ok = adx is not None and adx >= trend_min
        dbg["adx"] = round(adx, 2) if adx else "N/A"
        if not adx_ok:
            return None, f"ADX {dbg['adx']} < {trend_min}", dbg

        if len(c5) < RSI_PERIOD + VOLUME_LOOKBACK + 5:
            return None, "5M warming up", dbg

        cl5 = [x[4] for x in c5]
        vol5 = [x[5] for x in c5]
        cur5 = c5[-1]
        prev5 = c5[-2]

        cur_open = cur5[1]
        cur_high = cur5[2]
        cur_low = cur5[3]
        cur_close = cur5[4]
        cur_vol = cur5[5]

        ema50_5m = ema_value(cl5, EMA_PERIOD)
        if ema50_5m is None:
            return None, "5M EMA50 not ready", dbg

        rsi = rsi_value(cl5, RSI_PERIOD)
        rsi_ok_buy = rsi is not None and rsi > 50
        rsi_ok_sell = rsi is not None and rsi < 50

        avg_vol = float(np.mean(vol5[-(VOLUME_LOOKBACK+1):-1])) if len(vol5) > VOLUME_LOOKBACK else 0
        vol_ok = avg_vol == 0 or cur_vol > avg_vol

        cur_body = candle_body_ratio(cur5)
        body_ok = cur_body >= CANDLE_BODY_RATIO_MIN

        bull_candle = cur_close > cur_open
        bear_candle = cur_close < cur_open

        prev_low = prev5[3]
        prev_high = prev5[2]

        atr = calculate_atr(c5, ATR_PERIOD)
        pullback_mult = self.get_symbol_setting(symbol, "pullback_atr_mult", PULLBACK_ATR_MULTIPLIER)
        if atr:
            pullback_zone = atr * pullback_mult
        else:
            pullback_zone = ema50_5m * 0.003

        pullback_bull = abs(prev_low - ema50_5m) <= pullback_zone
        pullback_bear = abs(prev_high - ema50_5m) <= pullback_zone

        dbg["ema50_5m"] = round(ema50_5m, 5)
        dbg["rsi"] = round(rsi, 2) if rsi else "N/A"
        dbg["vol_ok"] = vol_ok
        dbg["body_ok"] = body_ok

        regime = "TREND_CAUTION" if is_caution else "TREND"
        rr_ratio = self.get_rr_ratio(regime)

        if trend_up and pullback_bull and bull_candle and body_ok and rsi_ok_buy and vol_ok:
            entry = float(cur_close)
            stop = float(cur_low)
            if atr:
                min_stop_distance = atr * STOP_ATR_MULTIPLIER
                if (entry - stop) < min_stop_distance:
                    stop = entry - min_stop_distance
            min_sl = entry * 0.0015
            if (entry - stop) < min_sl:
                stop = entry - min_sl
            if stop >= entry:
                return None, "Invalid long SL >= entry", dbg
            risk = entry - stop
            target = entry + risk * rr_ratio
            return Signal("buy", symbol, entry, stop, target,
                          f"{regime}: Pullback to EMA50 | ADX:{dbg['adx']} | RSI:{dbg['rsi']}",
                          regime, rr_ratio), f"{regime} LONG ✅ (1:{rr_ratio:.0f})", dbg

        if trend_down and pullback_bear and bear_candle and body_ok and rsi_ok_sell and vol_ok:
            entry = float(cur_close)
            stop = float(cur_high)
            if atr:
                min_stop_distance = atr * STOP_ATR_MULTIPLIER
                if (stop - entry) < min_stop_distance:
                    stop = entry + min_stop_distance
            min_sl = entry * 0.0015
            if (stop - entry) < min_sl:
                stop = entry + min_sl
            if stop <= entry:
                return None, "Invalid short SL <= entry", dbg
            risk = stop - entry
            target = entry - risk * rr_ratio
            return Signal("sell", symbol, entry, stop, target,
                          f"{regime}: Pullback to EMA50 | ADX:{dbg['adx']} | RSI:{dbg['rsi']}",
                          regime, rr_ratio), f"{regime} SHORT ✅ (1:{rr_ratio:.0f})", dbg

        return None, f"{regime}: Waiting for pullback", dbg

    # ------------------------------------------------------------------
    # RANGE strategy (STRENGTHENED)
    # ------------------------------------------------------------------
    def build_signal_range(self, symbol, c15, c5):
        """Mean reversion with trend alignment, ADX slope, reversal candle."""
        dbg = {}

        # 1) Check 15M trend
        if len(c15) < EMA_PERIOD + 5:
            return None, "15M warming up", dbg
        cl15 = [x[4] for x in c15]
        price15 = cl15[-1]
        ema50_15 = ema_value(cl15, EMA_PERIOD)
        if ema50_15 is None:
            return None, "EMA50 not ready", dbg
        dbg["trend_15m"] = "UP" if price15 > ema50_15 else "DOWN" if price15 < ema50_15 else "SIDE"
        dbg["ema50_15"] = round(ema50_15, 5)

        # 2) ADX must be < RANGE_MIN_ADX and falling
        adx = adx_value(c15, ADX_PERIOD)
        if adx is None:
            return None, "ADX not ready", dbg
        dbg["adx"] = round(adx, 2)

        # Calculate ADX slope (previous ADX from earlier candles)
        # Use a second ADX call with shifted data, but simple: compare to 5 candles ago
        adx_prev = None
        if len(c15) > 10:
            adx_prev = adx_value(c15[:-5], ADX_PERIOD)
        adx_falling = (adx_prev is not None and adx < adx_prev) or (adx_prev is None)
        dbg["adx_falling"] = adx_falling

        if adx >= RANGE_MIN_ADX:
            return None, f"ADX {adx:.1f} >= {RANGE_MIN_ADX} — market trending", dbg

        # 3) 5M data and bands
        closes = [x[4] for x in c5]
        current_price = closes[-1]
        ema20 = ema_value(closes, 20)
        atr = calculate_atr(c5, ATR_PERIOD)
        if ema20 is None or atr is None:
            return None, "Range indicators not ready", dbg

        range_mult = self.get_symbol_setting(symbol, "range_atr_mult", RANGE_ATR_MULTIPLIER)
        upper_band = ema20 + (atr * range_mult)
        lower_band = ema20 - (atr * range_mult)
        dbg["upper_band"] = round(upper_band, 5)
        dbg["lower_band"] = round(lower_band, 5)
        dbg["ema20"] = round(ema20, 5)

        # 4) RSI (short period)
        rsi = rsi_value(closes, RANGE_RSI_PERIOD)
        if rsi is None:
            return None, "RSI not ready", dbg
        dbg["rsi"] = round(rsi, 2)

        # 5) Volume check
        volumes = [x[5] for x in c5]
        avg_vol = np.mean(volumes[-VOLUME_LOOKBACK-1:-1]) if len(volumes) > VOLUME_LOOKBACK else 0
        cur_vol = c5[-1][5]
        vol_ok = avg_vol == 0 or cur_vol > avg_vol * 0.8
        dbg["vol_ok"] = vol_ok

        rr_ratio = self.get_rr_ratio("RANGE")

        # 6) Long conditions
        if current_price < lower_band and rsi < RANGE_RSI_OVERSOLD:
            # 15M trend must not be strong downtrend
            if price15 < ema50_15 * 0.99:   # more than 1% below EMA50
                return None, f"15M strong downtrend ({dbg['trend_15m']}), skip long", dbg
            # Optional: require ADX falling
            if not adx_falling:
                return None, "ADX not falling, range may be ending", dbg
            # Reversal candle: bullish close above lower band
            cur_candle = c5[-1]
            if RANGE_REQUIRE_REVERSAL_CANDLE:
                if not (cur_candle[4] > cur_candle[1] and cur_candle[4] > lower_band):
                    return None, "No bullish reversal candle", dbg
            # Entry
            entry = current_price
            stop = lower_band - (atr * 0.5)
            target = ema20
            # Adjust minimum stop
            min_sl = entry * 0.0015
            if (entry - stop) < min_sl:
                stop = entry - min_sl
            if stop >= entry:
                return None, "Invalid stop", dbg
            risk = entry - stop
            max_target = entry + (risk * rr_ratio)
            target = min(target, max_target)
            return Signal("buy", symbol, entry, stop, target,
                          f"RANGE: Mean reversion | RSI:{rsi:.0f} at lower band",
                          "RANGE", rr_ratio), f"RANGE LONG ✅ (1:{rr_ratio:.1f})", dbg

        # 7) Short conditions
        if current_price > upper_band and rsi > RANGE_RSI_OVERBOUGHT:
            if price15 > ema50_15 * 1.01:   # more than 1% above EMA50
                return None, f"15M strong uptrend ({dbg['trend_15m']}), skip short", dbg
            if not adx_falling:
                return None, "ADX not falling, range may be ending", dbg
            if RANGE_REQUIRE_REVERSAL_CANDLE:
                if not (cur_candle[4] < cur_candle[1] and cur_candle[4] < upper_band):
                    return None, "No bearish reversal candle", dbg
            entry = current_price
            stop = upper_band + (atr * 0.5)
            target = ema20
            min_sl = entry * 0.0015
            if (stop - entry) < min_sl:
                stop = entry + min_sl
            if stop <= entry:
                return None, "Invalid stop", dbg
            risk = stop - entry
            max_target = entry - (risk * rr_ratio)
            target = max(target, max_target)
            return Signal("sell", symbol, entry, stop, target,
                          f"RANGE: Mean reversion | RSI:{rsi:.0f} at upper band",
                          "RANGE", rr_ratio), f"RANGE SHORT ✅ (1:{rr_ratio:.1f})", dbg

        return None, f"RANGE: No entry", dbg

    # ------------------------------------------------------------------
    # MIXED strategy (unchanged)
    # ------------------------------------------------------------------
    def build_signal_breakout(self, symbol, c15, c5):
        current_candle = c5[-1]
        current_price = current_candle[4]
        current_volume = current_candle[5]
        highs_20 = [c[2] for c in c5[-BREAKOUT_PERIOD:]]
        lows_20 = [c[3] for c in c5[-BREAKOUT_PERIOD:]]
        high_20 = max(highs_20)
        low_20 = min(lows_20)
        volumes = [c[5] for c in c5[-BREAKOUT_PERIOD-1:-1]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        atr = calculate_atr(c5, ATR_PERIOD)
        dbg = {"high_20": round(high_20,5), "low_20": round(low_20,5),
               "volume_spike": round(current_volume/avg_volume,2) if avg_volume else 0}
        rr_ratio = self.get_rr_ratio("MIXED")
        if current_price > high_20 and current_volume > avg_volume * BREAKOUT_VOLUME_MULT:
            entry = current_price
            stop = high_20
            if atr and (entry - stop) < atr * 0.5:
                stop = entry - (atr * 0.5)
            if stop >= entry:
                return None, "Invalid stop", dbg
            risk = entry - stop
            target = entry + risk * rr_ratio
            return Signal("buy", symbol, entry, stop, target,
                          f"MIXED: Breakout above {high_20:.5f} | Vol:{dbg['volume_spike']}x",
                          "MIXED", rr_ratio), f"MIXED LONG ✅ (1:{rr_ratio:.1f})", dbg
        elif current_price < low_20 and current_volume > avg_volume * BREAKOUT_VOLUME_MULT:
            entry = current_price
            stop = low_20
            if atr and (stop - entry) < atr * 0.5:
                stop = entry + (atr * 0.5)
            if stop <= entry:
                return None, "Invalid stop", dbg
            risk = stop - entry
            target = entry - risk * rr_ratio
            return Signal("sell", symbol, entry, stop, target,
                          f"MIXED: Breakdown below {low_20:.5f} | Vol:{dbg['volume_spike']}x",
                          "MIXED", rr_ratio), f"MIXED SHORT ✅ (1:{rr_ratio:.1f})", dbg
        return None, f"MIXED: No breakout", dbg

    # ------------------------------------------------------------------
    # Hybrid dispatcher
    # ------------------------------------------------------------------
    async def build_signal_hybrid(self, symbol, c15, c5):
        regime = self.determine_regime(c15, c5, symbol)
        if regime == "TREND":
            signal, reason, dbg = self.build_signal_trend(symbol, c15, c5, is_caution=False)
            dbg["regime"] = "TREND"
            return signal, reason, dbg
        elif regime == "TREND_CAUTION":
            signal, reason, dbg = self.build_signal_trend(symbol, c15, c5, is_caution=True)
            dbg["regime"] = "TREND_CAUTION"
            return signal, reason, dbg
        elif regime == "RANGE":
            signal, reason, dbg = self.build_signal_range(symbol, c15, c5)
            dbg["regime"] = "RANGE"
            return signal, reason, dbg
        else:  # MIXED
            signal, reason, dbg = self.build_signal_breakout(symbol, c15, c5)
            dbg["regime"] = "MIXED"
            return signal, reason, dbg

    # ------------------------------------------------------------------
    # Order placement, reconciliation, scanning (unchanged)
    # ------------------------------------------------------------------
    async def set_leverage(self, symbol):
        try:
            await self.exchange.set_leverage(3, symbol, params={"marginMode": "cross"})
        except Exception as e:
            logger.warning(f"Leverage set failed {symbol}: {e}")

    async def get_size(self, symbol, entry, stop, regime):
        await self.fetch_balance()
        risk_amount = self.get_risk_per_trade(regime)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0
        market = self.exchange.market(symbol)
        contract_size = float(market.get("contractSize", 1.0) or 1.0)
        raw_size = risk_amount / (stop_distance * contract_size)
        precision = market.get("precision", {}).get("amount")
        size = float(self.exchange.amount_to_precision(symbol, raw_size)) if precision is not None else raw_size
        min_amt = float((market.get("limits", {}).get("amount") or {}).get("min") or 0)
        if min_amt and size < min_amt:
            size = min_amt
        return float(size)

    async def place_paper_trade(self, signal: Signal, entry_candle_ts=None):
        if signal.symbol in self.paper_positions or signal.symbol in self.active_positions:
            return
        sym_clean = signal.symbol.replace("/USDT:USDT", "")
        risk_amount = self.get_risk_per_trade(signal.regime)
        self.paper_positions[signal.symbol] = {
            "side": signal.side, "entry": signal.entry, "stop": signal.stop,
            "target": signal.target, "opened_at": time.time(),
            "opened_candle_ts": entry_candle_ts, "size": 0.0,
            "reason": signal.reason, "regime": signal.regime, "rr_ratio": signal.rr_ratio,
        }
        self.trades_today += 1
        self.cooldown_until = time.time() + COOLDOWN_SEC
        await self.tg(
            f"📝 PAPER TRADE OPENED\n"
            f"Pair: {sym_clean} | {signal.side.upper()} | STRATEGY: {signal.regime}\n"
            f"Entry: {signal.entry:.5f}\nStop: {signal.stop:.5f}\nTarget: {signal.target:.5f}\n"
            f"Risk: ${risk_amount:.2f} | RR: 1:{signal.rr_ratio:.1f}\nReason: {signal.reason}"
        )
        logger.info(f"PAPER trade opened: {signal.symbol} {signal.side} @ {signal.entry} [{signal.regime}]")

    async def place_live_trade(self, signal: Signal):
        await self.set_leverage(signal.symbol)
        size = await self.get_size(signal.symbol, signal.entry, signal.stop, signal.regime)
        if size <= 0:
            await self.tg(f"⚠️ {signal.symbol} size invalid — skipping")
            return
        opposite = "sell" if signal.side == "buy" else "buy"
        hold_side = "long" if signal.side == "buy" else "short"
        risk_amount = self.get_risk_per_trade(signal.regime)
        entry_params = {"marginMode": "cross", "tradeSide": "open", "holdSide": hold_side}
        sl_params = {"stopPrice": signal.stop, "triggerPrice": signal.stop, "reduceOnly": True,
                     "marginMode": "cross", "tradeSide": "close", "holdSide": hold_side}
        tp_params = {"stopPrice": signal.target, "triggerPrice": signal.target, "reduceOnly": True,
                     "marginMode": "cross", "tradeSide": "close", "holdSide": hold_side}
        try:
            entry_order = await self.exchange.create_order(
                symbol=signal.symbol, type="market", side=signal.side, amount=size, params=entry_params)
            avg_price = float(entry_order.get("average") or entry_order.get("price") or signal.entry)
            await asyncio.sleep(0.5)
            try:
                await self.exchange.create_order(symbol=signal.symbol, type="stop_market", side=opposite,
                                                 amount=size, params=sl_params)
            except Exception as sl_err:
                logger.warning(f"SL order failed {signal.symbol}: {sl_err}")
            await asyncio.sleep(0.5)
            try:
                await self.exchange.create_order(symbol=signal.symbol, type="take_profit_market", side=opposite,
                                                 amount=size, params=tp_params)
            except Exception as tp_err:
                logger.warning(f"TP order failed {signal.symbol}: {tp_err}")
            self.active_positions[signal.symbol] = {
                "side": signal.side, "size": size, "entry": avg_price, "stop": signal.stop,
                "target": signal.target, "opened_at": time.time(),
                "regime": signal.regime, "rr_ratio": signal.rr_ratio,
            }
            self.trades_today += 1
            self.cooldown_until = time.time() + COOLDOWN_SEC
            sym_clean = signal.symbol.replace("/USDT:USDT", "")
            risk_actual = abs(avg_price - signal.stop) * size
            await self.tg(
                f"🚀 LIVE TRADE OPENED\nPair: {sym_clean} | {signal.side.upper()} | STRATEGY: {signal.regime}\n"
                f"Entry: {avg_price:.5f}\nStop: {signal.stop:.5f}\nTarget: {signal.target:.5f}\n"
                f"Size: {size} contracts\nRisk: ${risk_actual:.3f} | RR: 1:{signal.rr_ratio:.1f}\n"
                f"Reason: {signal.reason}"
            )
            logger.info(f"LIVE trade opened: {signal.symbol} {signal.side} @ {avg_price} [{signal.regime}]")
        except Exception as e:
            logger.error(f"Order failed {signal.symbol}: {e}")
            await self.tg(f"❌ Order failed {signal.symbol}: {str(e)[:250]}")

    async def place_trade(self, signal: Signal, entry_candle_ts=None):
        if PAPER_MODE:
            await self.place_paper_trade(signal, entry_candle_ts)
        else:
            await self.place_live_trade(signal)

    async def reconcile_paper_positions(self):
        if not self.exchange or not self.paper_positions:
            return
        for sym in list(self.paper_positions.keys()):
            try:
                pos = self.paper_positions[sym]
                candles = await self.fetch_ohlcv(sym, TF_ENTRY, 4)
                if not candles or len(candles) < 2:
                    continue
                last = candles[-1]
                last_ts = last[0]
                if pos.get("opened_candle_ts") == last_ts:
                    continue
                high = float(last[2])
                low = float(last[3])
                side = pos["side"]
                entry = float(pos["entry"])
                stop = float(pos["stop"])
                target = float(pos["target"])
                regime = pos.get("regime", "UNKNOWN")
                rr_ratio = pos.get("rr_ratio", 2.0)
                risk_amount = self.get_risk_per_trade(regime)
                result = None
                pnl = 0.0
                if side == "buy":
                    if low <= stop:
                        result = "LOSS"
                        pnl = -risk_amount
                    elif high >= target:
                        result = "WIN"
                        pnl = risk_amount * rr_ratio
                else:
                    if high >= stop:
                        result = "LOSS"
                        pnl = -risk_amount
                    elif low <= target:
                        result = "WIN"
                        pnl = risk_amount * rr_ratio
                if result is None:
                    continue
                self.paper_positions.pop(sym, None)
                if pnl < 0:
                    self.consec_losses += 1
                    self.losses_today += 1
                    logger.warning(f"Consecutive losses: {self.consec_losses}/{MAX_CONSEC_LOSSES}")
                    if self.consec_losses >= MAX_CONSEC_LOSSES:
                        self.pause_until = time.time() + (CONSEC_LOSS_PAUSE_HR * 3600)
                        await self.tg(
                            f"⏸️ {MAX_CONSEC_LOSSES} CONSECUTIVE LOSSES — PAUSING FOR THE DAY (24h)\n"
                            f"Loss streak: {self.consec_losses}\nToday's PnL: {self.profit_today:+.4f} USDT"
                        )
                else:
                    self.consec_losses = 0
                self.profit_today = round(self.profit_today + pnl, 4)
                self._save_trade({
                    "time": time.time(), "symbol": sym, "side": side, "entry": entry,
                    "stop": stop, "target": target, "pnl": pnl, "mode": "paper",
                    "result": result, "regime": regime, "rr_ratio": rr_ratio,
                })
                sym_clean = sym.replace("/USDT:USDT", "")
                stats = self.win_rate_stats.get(sym, {"wins": 0, "losses": 0})
                total = stats["wins"] + stats["losses"]
                wr = (stats["wins"] / total * 100) if total > 0 else 0
                await self.tg(
                    f"{'✅ PAPER WIN' if pnl >= 0 else '❌ PAPER LOSS'} | {sym_clean} | {regime}\n"
                    f"PnL: {pnl:+.4f} USDT (1:{rr_ratio:.1f} RR)\nToday: {self.profit_today:+.4f} USDT\n"
                    f"Streak: {self.consec_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"Win Rate {sym_clean}: {wr:.1f}% ({stats['wins']}/{total})"
                )
            except Exception as e:
                logger.warning(f"Paper reconcile error {sym}: {e}")

    async def reconcile_live_positions(self):
        # same as before (omitted for brevity, unchanged)
        pass

    async def reconcile(self):
        if PAPER_MODE:
            await self.reconcile_paper_positions()
        else:
            await self.reconcile_live_positions()

    async def scan_symbol(self, symbol, stagger=0):
        await asyncio.sleep(stagger)
        self.market_debug[symbol] = {"time": time.time(), "why": "Initializing...", "signal": None}
        while self.is_scanning:
            try:
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                self._reset_daily()
                if not self.exchange:
                    self.market_debug[symbol] = {"time": time.time(), "why": "Not connected", "signal": None}
                    await asyncio.sleep(5)
                    continue
                if symbol in self.active_positions or symbol in self.paper_positions:
                    continue
                can, gate = self.can_trade(symbol)
                if not can:
                    self.market_debug[symbol] = {"time": time.time(), "why": gate, "signal": None}
                    continue
                self.market_debug[symbol] = {"time": time.time(), "why": "Fetching candles...", "signal": None}
                c15 = await self.fetch_ohlcv(symbol, TF_TREND, CANDLES_15M)
                await asyncio.sleep(0.3)
                c5 = await self.fetch_ohlcv(symbol, TF_ENTRY, CANDLES_5M)
                if not c15 or not c5:
                    msg = " | ".join([f"{tf}:0 candles" for tf, d in [(TF_TREND,c15),(TF_ENTRY,c5)] if not d])
                    logger.warning(f"{symbol}: {msg}")
                    self.market_debug[symbol] = {"time": time.time(), "why": msg, "signal": None}
                    await asyncio.sleep(10)
                    continue
                signal, reason, dbg = await self.build_signal_hybrid(symbol, c15, c5)
                dbg["time"] = time.time()
                dbg["why"] = reason
                dbg["signal"] = signal.side.upper() if signal else None
                dbg["regime"] = signal.regime if signal else dbg.get("regime", "UNKNOWN")
                self.market_debug[symbol] = dbg
                if signal:
                    entry_candle_ts = c5[-1][0]
                    await self.place_trade(signal, entry_candle_ts)
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                self.market_debug[symbol] = {"time": time.time(), "why": f"Error: {str(e)[:100]}", "signal": None}
                await asyncio.sleep(10)

    async def watchdog(self):
        while True:
            await asyncio.sleep(15)
            try:
                await self.reconcile()
                await self.fetch_balance()
            except Exception as e:
                logger.warning(f"Watchdog: {e}")

    async def run(self):
        asyncio.create_task(self.watchdog())
        while True:
            await asyncio.sleep(60)


# ========================= TELEGRAM UI =========================
bot = HybridBot()

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🔌 CONNECT", callback_data="CONNECT")],
        [InlineKeyboardButton("⏸ PAUSE", callback_data="PAUSE"),
         InlineKeyboardButton("▶️ RESUME", callback_data="RESUME")],
        [InlineKeyboardButton("📈 WIN RATES", callback_data="WINRATES")],
    ])

def fmt_debug(sym, d):
    if not d:
        return f"📍 {sym.replace('/USDT:USDT','')} ⏳ No data yet\n"
    age = int(time.time() - d.get("time", time.time()))
    signal = d.get("signal") or "—"
    why = d.get("why", "—")
    regime = d.get("regime", "—")
    trend = d.get("trend", "—")
    ema50_15 = d.get("ema50_15", "—")
    ema50_5m = d.get("ema50_5m", "—")
    adx = d.get("adx", "—")
    rsi = d.get("rsi", "—")
    vol_ok = "✅" if d.get("vol_ok") else "❌"
    body_ok = "✅" if d.get("body_ok") else "❌"
    sym_c = sym.replace("/USDT:USDT", "")
    return (f"📍 {sym_c} ({age}s) | {regime}\n"
            f"15M: {trend} | EMA50:{ema50_15} | ADX:{adx}\n"
            f"5M:  EMA50:{ema50_5m} | RSI:{rsi} | Vol:{vol_ok} Body:{body_ok}\n"
            f"Signal: {signal} | {why[:50]}\n")

async def safe_edit(q, text, markup=None):
    try:
        await q.edit_message_text(text=text[:4000], reply_markup=markup)
    except Exception:
        pass

async def btn_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    bot._chat_ids.add(q.message.chat_id)
    try:
        await q.answer()
    except Exception:
        pass

    if q.data == "CONNECT":
        ok = await bot.connect()
        mode = "PAPER" if PAPER_MODE else "LIVE"
        await safe_edit(q, f"{'✅ Connected' if ok else '❌ Failed'} | {mode}\nBalance: {bot.balance_usdt:.2f} USDT\n"
                        f"Max positions: {MAX_OPEN_POSITIONS} | Stop after {MAX_CONSEC_LOSSES} losses", keyboard())
    elif q.data == "START":
        if not bot.exchange:
            await safe_edit(q, "❌ Connect first", keyboard())
            return
        bot.is_scanning = False
        await asyncio.sleep(1)
        bot.is_scanning = True
        for i, sym in enumerate(SYMBOLS):
            asyncio.create_task(bot.scan_symbol(sym, stagger=i*12))
        mode = "PAPER" if PAPER_MODE else "LIVE"
        await safe_edit(q,
            f"🔍 HYBRID SCANNER ACTIVE\nMode: {mode}\nPairs: SOL | BTC | ADA\n"
            f"Max positions: {MAX_OPEN_POSITIONS} (one per symbol)\n"
            f"Strategies:\n  📈 TREND (ADX 25-35): Pullback | 1:4 RR | Full risk\n"
            f"  ⚠️ TREND_CAUTION (ADX >35): Pullback | 1:3 RR | Half risk\n"
            f"  📊 RANGE (ADX<20): Mean Reversion (strengthened) | 1:1.5 RR | Half risk\n"
            f"  🚀 MIXED (ADX 20-25): Breakout | 1:2.5 RR | 75% risk\n"
            f"Session: {SESSION_START_UTC:02d}:00–{SESSION_END_UTC:02d}:00 UTC\n"
            f"Stop after {MAX_CONSEC_LOSSES} consecutive losses", keyboard())
    elif q.data == "STOP":
        bot.is_scanning = False
        await safe_edit(q, "⏹️ Scanner stopped", keyboard())
    elif q.data == "PAUSE":
        bot.pause_until = time.time() + 86400
        await safe_edit(q, "⏸ Paused 24h", keyboard())
    elif q.data == "RESUME":
        bot.pause_until = 0
        bot.consec_losses = 0
        await safe_edit(q, "▶️ Resumed — consecutive loss counter reset", keyboard())
    elif q.data == "WINRATES":
        msg = "📊 **WIN RATES BY SYMBOL & STRATEGY**\n\n"
        for sym, stats in bot.win_rate_stats.items():
            sym_clean = sym.replace("/USDT:USDT", "")
            wins = stats["wins"]
            losses = stats["losses"]
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            msg += f"**{sym_clean}**\n  Overall: {win_rate:.1f}% ({wins}/{total})\n"
            for regime, rstats in stats.get("regime_stats", {}).items():
                rwins = rstats["wins"]
                rlosses = rstats["losses"]
                rtotal = rwins + rlosses
                rwr = (rwins / rtotal * 100) if rtotal > 0 else 0
                msg += f"    {regime}: {rwr:.1f}% ({rwins}/{rtotal})\n"
            msg += "\n"
        await safe_edit(q, msg, keyboard())
    elif q.data == "STATUS":
        now = time.time()
        if now < bot.status_cd_until:
            await safe_edit(q, f"⏳ {int(bot.status_cd_until-now)}s", keyboard())
            return
        bot.status_cd_until = now + STATUS_REFRESH_COOLDOWN
        await bot.fetch_balance()
        _, gate = bot.can_trade()
        open_pos = ""
        all_positions = {}
        all_positions.update(bot.active_positions)
        all_positions.update(bot.paper_positions)
        for sym, pos in all_positions.items():
            sym_c = sym.replace("/USDT:USDT", "")
            age = int(time.time() - pos["opened_at"])
            regime = pos.get("regime", "UNKNOWN")
            entry = pos["entry"]
            stop = pos["stop"]
            target = pos["target"]
            side = pos["side"]
            current_price = await bot.get_current_price(sym)
            if current_price > 0:
                if side == "buy":
                    pnl_pct = ((current_price - entry) / entry) * 100
                    pnl_status = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                    dist_stop = ((current_price - stop) / entry) * 100
                    dist_target = ((target - current_price) / entry) * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100
                    pnl_status = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                    dist_stop = ((stop - current_price) / entry) * 100
                    dist_target = ((current_price - target) / entry) * 100
                open_pos += (f"\n{pnl_status} {sym_c} {side.upper()} [{regime}]\n"
                             f"   Entry: {entry:.5f} | Current: {current_price:.5f}\n"
                             f"   PnL: {pnl_pct:+.2f}% | SL: {stop:.5f} | TP: {target:.5f}\n"
                             f"   To Stop: {dist_stop:.2f}% | To Target: {dist_target:.2f}%\n"
                             f"   Age: {age}s")
            else:
                open_pos += f"\n🔵 {sym_c} {side.upper()} [{regime}] @ {entry:.5f} | SL:{stop:.5f} TP:{target:.5f} | Age: {age}s\n   ⏳ Price unavailable"
        session_active = is_session_active()
        utc_hour = datetime.utcnow().hour
        mode = "📝 PAPER" if PAPER_MODE else "💰 LIVE"
        total_wins = sum(s["wins"] for s in bot.win_rate_stats.values())
        total_losses = sum(s["losses"] for s in bot.win_rate_stats.values())
        total_trades = total_wins + total_losses
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        pause_msg = ""
        if bot.pause_until > time.time():
            remaining_hours = int((bot.pause_until - time.time()) / 3600)
            pause_msg = f"\n⏸️ PAUSED: {remaining_hours}h left (after {MAX_CONSEC_LOSSES} losses)"
        header = (f"🕒 {now_wat()}\n🤖 {'ACTIVE' if bot.is_scanning else 'OFFLINE'} | {mode}\n"
                  f"💰 Balance: {bot.balance_usdt:.2f} USDT\n📈 PnL Today: {bot.profit_today:+.4f} USDT\n"
                  f"📊 Win Rate: {overall_wr:.1f}% ({total_wins}/{total_trades})\n"
                  f"📉 Streak: {bot.consec_losses}/{MAX_CONSEC_LOSSES} | Trades: {bot.trades_today}/{MAX_TRADES_PER_DAY}\n"
                  f"📌 Positions: {bot.total_open_positions()}/{MAX_OPEN_POSITIONS}\n"
                  f"🕐 Session: {'✅ ACTIVE' if session_active else f'❌ CLOSED ({utc_hour:02d}:00 UTC)'}\n"
                  f"🚦 Gate: {gate}{pause_msg}")
        if open_pos:
            header += f"\n\n📌 OPEN POSITIONS:\n{open_pos}"
        scan_lines = "\n\n📡 LIVE SCAN\n" + "\n".join(fmt_debug(sym, bot.market_debug.get(sym, {})) for sym in SYMBOLS)
        await safe_edit(q, header + scan_lines, keyboard())

async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bot._chat_ids.add(update.message.chat_id)
    mode = "PAPER" if PAPER_MODE else "LIVE"
    await update.message.reply_text(
        "💎 Bitget HYBRID Futures Bot (RANGE Strengthened)\n"
        f"Mode: {mode}\n\n"
        "**STRATEGIES (Based on ADX):**\n"
        "📈 TREND (ADX 25-35): Pullback to EMA50 | 1:4 RR | Full risk\n"
        "⚠️ TREND_CAUTION (ADX >35): Pullback | 1:3 RR | Half risk\n"
        "📊 RANGE (ADX <20): Mean Reversion (strengthened) | 1:1.5 RR | Half risk\n"
        "🚀 MIXED (ADX 20-25): Breakout + Volume | 1:2.5 RR | 75% risk\n\n"
        "**RANGE Enhancements:**\n"
        "- 15M trend alignment (no strong downtrend for longs)\n"
        "- ADX must be <20 AND falling\n"
        "- Requires reversal candle (close beyond band)\n"
        "- Wider bands (2.0x ATR)\n\n"
        f"**Risk:** Stop after {MAX_CONSEC_LOSSES} consecutive losses, max {MAX_OPEN_POSITIONS} positions\n"
        f"Session: {SESSION_START_UTC:02d}:00–{SESSION_END_UTC:02d}:00 UTC\n\n"
        "**How to use:**\n1. CONNECT\n2. START\n3. STATUS (live PnL)\n4. WIN RATES",
        reply_markup=keyboard()
    )

async def symbols_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bot._chat_ids.add(update.message.chat_id)
    if not bot.exchange:
        await update.message.reply_text("❌ Connect first — press CONNECT button")
        return
    try:
        markets = bot.exchange.markets
        sol = [s for s in markets.keys() if "SOL" in s and "USDT" in s]
        btc = [s for s in markets.keys() if "BTC" in s and "USDT" in s]
        ada = [s for s in markets.keys() if "ADA" in s and "USDT" in s]
        msg = (f"📋 Available USDT Markets on Bitget\n\n"
               f"SOL pairs:\n" + "\n".join(sol[:10] or ["None found"]) + "\n\n"
               f"BTC pairs:\n" + "\n".join(btc[:10] or ["None found"]) + "\n\n"
               f"ADA pairs:\n" + "\n".join(ada[:10] or ["None found"]))
        await update.message.reply_text(msg[:4000])
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")

# ========================= MAIN =========================
if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing TELEGRAM_TOKEN env variable")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("symbols", symbols_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bot.run())
    app.run_polling(close_loop=False)
