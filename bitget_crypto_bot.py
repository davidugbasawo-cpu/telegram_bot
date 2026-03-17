"""
Bitget Crypto Futures Bot
Strategy: 15M Trend + 5M Breakout with Chop Filter
  15M → trend direction (EMA 200 only)
  5M  → breakout entry (previous candle high/low break + body strength + range filter)
Pairs: SOL/USDT:USDT, DOGE/USDT:USDT, XRP/USDT:USDT perpetual futures

Changes from previous version:
- Simplified to EMA 200 only (removed EMA50, RSI, MACD etc.)
- New chop filter: EMA distance + trend clarity + 15M candle strength
- New entry: 5M breakout of previous candle high/low
- Body >= 50% of candle range required
- 5M sideways filter: breakout candle range >= avg of last 10 candles
- Session filter: only trade during active sessions (UTC)
- Precise EMA distance rule: price must be >= 0.5% from EMA200
- Minimum SL distance: >= 0.15% of entry price
- Daily trade limit: 6
- Consecutive loss pause: 3 losses = 2 hour pause
- Fixed risk: $0.25 per trade
- RR: 1:3
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import ccxt.async_support as ccxt
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ========================= CREDENTIALS =========================
BITGET_API_KEY    = "bg_d0944109a841af8a4167114466af2bf3"
BITGET_SECRET     = "e2bf8eed9bc0f4963d4c2c325ba19eb03476f9b504341217bbbe7343c80268be"
BITGET_PASSPHRASE = "Salome1234"
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_TOKEN", "8697638086:AAG00D0RXUAqXFTjy8-4XO4Bka2kBamo-VA")

USE_TESTNET = os.getenv("BITGET_TESTNET", "true").lower() == "true"

# ========================= MARKETS =========================
SYMBOLS = ["SOL/USDT:USDT", "DOGE/USDT:USDT", "XRP/USDT:USDT"]

# ========================= TIMEFRAMES =========================
TF_TREND   = "15m"
TF_ENTRY   = "5m"
CANDLES_15M = 250
CANDLES_5M  = 60

# ========================= INDICATOR =========================
EMA_PERIOD = 200   # Only one EMA — keep it simple

# ========================= CHOP FILTER =========================
EMA_DISTANCE_MIN_PCT  = 0.005   # Price must be >= 0.5% from EMA200 on 15M
CANDLE_BODY_RATIO_MIN = 0.50    # 15M last candle body must be >= 50% of range
SLOPE_LOOKBACK        = 5       # Candles to measure EMA200 slope direction

# ========================= ENTRY FILTER =========================
BREAKOUT_BODY_MIN     = 0.50    # 5M breakout candle body >= 50% of range
SIDEWAYS_LOOKBACK     = 10      # Candles to calculate avg range for sideways filter

# ========================= SESSION FILTER =========================
# Only trade during active market hours (UTC)
# London/NY overlap: 12:00-16:00 UTC
# New York session:  13:00-21:00 UTC
SESSION_START_UTC = 12   # 12:00 UTC
SESSION_END_UTC   = 21   # 21:00 UTC

# ========================= RISK =========================
RISK_PER_TRADE    = 0.25   # Fixed $0.25 per trade
LEVERAGE          = 3
RR_RATIO          = 3.0    # 1:3 risk/reward
MIN_SL_PCT        = 0.0015 # Minimum SL distance = 0.15% of entry

# ========================= LIMITS =========================
MAX_TRADES_PER_DAY   = 6
MAX_CONSEC_LOSSES    = 3
CONSEC_LOSS_PAUSE_HR = 2      # Pause 2 hours after 3 consecutive losses
COOLDOWN_SEC         = 300    # 5 min between trades
MAX_OPEN_POSITIONS   = 1

# ========================= OTHER =========================
TRADE_LOG_FILE          = "bitget_trades.json"
SCAN_INTERVAL_SEC       = 15
STATUS_REFRESH_COOLDOWN = 10
TIMEZONE                = "Africa/Lagos"


# ========================= HELPERS =========================
def ema_value(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return float(ema)


def ema_series(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    out = [None] * (period - 1) + [ema]
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
        out.append(float(ema))
    return out


def candle_body_ratio(candle):
    """Body as ratio of total candle range"""
    rng = abs(candle[2] - candle[3])  # high - low
    if rng == 0:
        return 0.0
    return abs(candle[4] - candle[1]) / rng  # |close - open| / range


def is_session_active():
    """Check if current UTC hour is within active trading session"""
    utc_hour = datetime.utcnow().hour
    return SESSION_START_UTC <= utc_hour < SESSION_END_UTC


def now_wat():
    return datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")


# ========================= SIGNAL =========================
@dataclass
class Signal:
    side:    str
    symbol:  str
    entry:   float
    stop:    float
    target:  float
    reason:  str


# ========================= BOT =========================
class BitgetBot:
    def __init__(self):
        self.exchange          = None
        self.app               = None
        self.is_scanning       = False
        self.active_positions  = {}
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
        self._chat_ids         = set()
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

    def _reset_daily(self):
        today = datetime.now(ZoneInfo(TIMEZONE)).date()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.trades_today    = 0
            self.profit_today    = 0.0
            self.losses_today    = 0
            self.consec_losses   = 0

    async def connect(self):
        try:
            if self.exchange:
                try:
                    await self.exchange.close()
                except Exception:
                    pass
            self.exchange = ccxt.bitget({
                "apiKey":   BITGET_API_KEY,
                "secret":   BITGET_SECRET,
                "password": BITGET_PASSPHRASE,
                "enableRateLimit": True,
                "options":  {"defaultType": "swap"},
            })
            if USE_TESTNET:
                self.exchange.set_sandbox_mode(True)
            await self.exchange.load_markets()
            await self.fetch_balance()
            logger.info(f"Connected to Bitget {'TESTNET' if USE_TESTNET else 'LIVE'}")
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

    async def tg(self, text):
        if not self.app:
            return
        for cid in list(self._chat_ids):
            try:
                await self.app.bot.send_message(chat_id=cid, text=str(text)[:4000])
            except Exception:
                pass

    def can_trade(self):
        self._reset_daily()

        # Session filter
        if not is_session_active():
            utc_hour = datetime.utcnow().hour
            return False, f"Outside session ({utc_hour:02d}:00 UTC) — active {SESSION_START_UTC:02d}:00-{SESSION_END_UTC:02d}:00"

        if time.time() < self.pause_until:
            remaining = int((self.pause_until - time.time()) / 60)
            return False, f"Paused after {MAX_CONSEC_LOSSES} losses — {remaining}min left"

        if time.time() < self.cooldown_until:
            left = int(self.cooldown_until - time.time())
            return False, f"Cooldown {left}s"

        if len(self.active_positions) >= MAX_OPEN_POSITIONS:
            return False, "Max positions open"

        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Daily limit reached ({MAX_TRADES_PER_DAY})"

        if self.consec_losses >= MAX_CONSEC_LOSSES:
            return False, f"Paused — {MAX_CONSEC_LOSSES} consecutive losses"

        return True, "OK"

    async def fetch_ohlcv(self, symbol, tf, limit):
        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            # Drop last (open/unconfirmed) candle
            return data[:-1] if len(data) > 1 else data
        except Exception as e:
            logger.warning(f"OHLCV {symbol} {tf}: {e}")
            return []

    def build_signal(self, symbol, c15, c5):
        """
        Returns (Signal | None, reason_str, debug_dict)

        Logic:
        1. 15M EMA200 direction → trend
        2. Chop filter: EMA distance + slope + last 15M candle body
        3. 5M: breakout of previous candle high (BUY) or low (SELL)
        4. Breakout candle body >= 50%
        5. Breakout candle range >= avg of last 10 candles (sideways filter)
        6. SL below/above previous 5M candle
        7. Min SL distance check
        """
        dbg = {}

        # ── 15M TREND ──────────────────────────────────────
        if len(c15) < EMA_PERIOD + SLOPE_LOOKBACK + 5:
            return None, "15M warming up", dbg

        cl15 = [x[4] for x in c15]
        last_15 = c15[-1]
        price15 = cl15[-1]

        ema200_series = ema_series(cl15, EMA_PERIOD)
        if not ema200_series or ema200_series[-1] is None:
            return None, "EMA200 not ready", dbg

        ema200 = ema200_series[-1]

        # EMA200 slope — compare current vs N candles ago
        old_ema = None
        for val in reversed(ema200_series[-(SLOPE_LOOKBACK+2):-1]):
            if val is not None:
                old_ema = val
                break
        ema_slope = (ema200 - old_ema) if old_ema else 0.0

        trend_up   = price15 > ema200 and ema_slope > 0
        trend_down = price15 < ema200 and ema_slope < 0

        dbg["ema200_15"]  = round(ema200, 5)
        dbg["price15"]    = round(price15, 5)
        dbg["ema_slope"]  = round(ema_slope, 6)
        dbg["trend"]      = "UP" if trend_up else "DOWN" if trend_down else "SIDE"

        if not trend_up and not trend_down:
            return None, f"15M sideways | Price:{price15:.4f} EMA200:{ema200:.4f}", dbg

        # ── CHOP FILTER ────────────────────────────────────
        # 1. EMA distance — price must be >= 0.5% from EMA200
        ema_dist_pct = abs(price15 - ema200) / ema200
        dist_ok = ema_dist_pct >= EMA_DISTANCE_MIN_PCT

        # 2. EMA slope must not be flat (already embedded in trend_up/down)
        slope_ok = ema_slope != 0.0

        # 3. Last 15M candle must have body >= 50% of range (no doji)
        last_15_body = candle_body_ratio(last_15)
        candle_ok = last_15_body >= CANDLE_BODY_RATIO_MIN

        dbg["ema_dist_pct"]  = round(ema_dist_pct * 100, 3)
        dbg["dist_ok"]       = dist_ok
        dbg["candle_ok"]     = candle_ok
        dbg["last_15_body"]  = round(last_15_body, 2)

        if not dist_ok:
            return None, f"Price too close to EMA200 ({ema_dist_pct*100:.2f}% < 0.5%)", dbg
        if not candle_ok:
            return None, f"Weak 15M candle — body {last_15_body:.2f} < 0.50 (doji/indecision)", dbg

        # ── 5M ENTRY ───────────────────────────────────────
        if len(c5) < SIDEWAYS_LOOKBACK + 3:
            return None, "5M warming up", dbg

        prev5 = c5[-2]   # Previous confirmed candle
        cur5  = c5[-1]   # Current forming candle (last confirmed)

        prev_high = prev5[2]
        prev_low  = prev5[3]
        cur_open  = cur5[1]
        cur_close = cur5[4]
        cur_high  = cur5[2]
        cur_low   = cur5[3]

        # Breakout check
        bull_breakout = cur_close > prev_high and cur_open <= prev_high
        bear_breakout = cur_close < prev_low  and cur_open >= prev_low

        # Breakout candle body ratio
        cur_body = candle_body_ratio(cur5)
        body_ok  = cur_body >= BREAKOUT_BODY_MIN

        # Sideways filter — current candle range vs average of last N candles
        recent_ranges = [abs(c[2] - c[3]) for c in c5[-(SIDEWAYS_LOOKBACK+2):-2]]
        avg_range     = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
        cur_range     = abs(cur_high - cur_low)
        range_ok      = avg_range == 0 or cur_range >= avg_range

        dbg["prev_high"]    = round(prev_high, 5)
        dbg["prev_low"]     = round(prev_low, 5)
        dbg["bull_break"]   = bull_breakout
        dbg["bear_break"]   = bear_breakout
        dbg["cur_body"]     = round(cur_body, 2)
        dbg["body_ok"]      = body_ok
        dbg["range_ok"]     = range_ok
        dbg["cur_range"]    = round(cur_range, 5)
        dbg["avg_range"]    = round(avg_range, 5)

        # ── LONG SETUP ─────────────────────────────────────
        if trend_up and bull_breakout and body_ok and range_ok:
            entry = float(cur_close)
            stop  = float(prev_low)

            # Minimum SL distance check
            min_sl = entry * MIN_SL_PCT
            if (entry - stop) < min_sl:
                return None, f"SL too tight ({entry-stop:.5f} < min {min_sl:.5f})", dbg

            if stop >= entry:
                return None, "Invalid long SL >= entry", dbg

            risk   = entry - stop
            target = entry + risk * RR_RATIO

            return Signal(
                "buy", symbol, entry, stop, target,
                f"15M UP + 5M breakout above {prev_high:.4f} | Body:{cur_body:.2f}"
            ), "LONG ✅", dbg

        # ── SHORT SETUP ────────────────────────────────────
        if trend_down and bear_breakout and body_ok and range_ok:
            entry = float(cur_close)
            stop  = float(prev_high)

            # Minimum SL distance check
            min_sl = entry * MIN_SL_PCT
            if (stop - entry) < min_sl:
                return None, f"SL too tight ({stop-entry:.5f} < min {min_sl:.5f})", dbg

            if stop <= entry:
                return None, "Invalid short SL <= entry", dbg

            risk   = stop - entry
            target = entry - risk * RR_RATIO

            return Signal(
                "sell", symbol, entry, stop, target,
                f"15M DOWN + 5M breakout below {prev_low:.4f} | Body:{cur_body:.2f}"
            ), "SHORT ✅", dbg

        # ── NO SIGNAL — detailed reason ────────────────────
        if trend_up and not bull_breakout:
            reason = f"Waiting 5M breakout above {prev_high:.4f} (cur:{cur_close:.4f})"
        elif trend_down and not bear_breakout:
            reason = f"Waiting 5M breakout below {prev_low:.4f} (cur:{cur_close:.4f})"
        elif not body_ok:
            reason = f"Weak breakout candle body {cur_body:.2f} < {BREAKOUT_BODY_MIN}"
        elif not range_ok:
            reason = f"Candle too small — range {cur_range:.5f} < avg {avg_range:.5f}"
        else:
            reason = "No valid setup"

        return None, reason, dbg

    async def set_leverage(self, symbol):
        try:
            await self.exchange.set_leverage(LEVERAGE, symbol)
        except Exception as e:
            logger.warning(f"Leverage set failed {symbol}: {e}")

    async def get_size(self, symbol, entry, stop):
        """Calculate position size based on fixed $0.25 risk"""
        await self.fetch_balance()
        risk_amount   = RISK_PER_TRADE   # Fixed $0.25
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0
        market        = self.exchange.market(symbol)
        contract_size = float(market.get("contractSize", 1.0) or 1.0)
        raw_size      = risk_amount / (stop_distance * contract_size)
        precision     = market.get("precision", {}).get("amount")
        size          = float(self.exchange.amount_to_precision(symbol, raw_size)) if precision else raw_size
        min_amt       = float((market.get("limits", {}).get("amount") or {}).get("min") or 0)
        if min_amt and size < min_amt:
            size = min_amt
        return float(size)

    async def place_trade(self, signal: Signal):
        await self.set_leverage(signal.symbol)
        size = await self.get_size(signal.symbol, signal.entry, signal.stop)
        if size <= 0:
            await self.tg(f"⚠️ {signal.symbol} size invalid — skipping")
            return

        opposite = "sell" if signal.side == "buy" else "buy"
        try:
            # Market entry
            entry_order = await self.exchange.create_order(
                symbol=signal.symbol, type="market",
                side=signal.side, amount=size,
                params={"marginMode": "cross"}
            )
            avg_price = float(entry_order.get("average") or entry_order.get("price") or signal.entry)

            # Stop loss
            await self.exchange.create_order(
                symbol=signal.symbol, type="stop_market",
                side=opposite, amount=size,
                params={
                    "stopPrice": signal.stop,
                    "triggerPrice": signal.stop,
                    "reduceOnly": True,
                    "marginMode": "cross",
                }
            )

            # Take profit
            await self.exchange.create_order(
                symbol=signal.symbol, type="take_profit_market",
                side=opposite, amount=size,
                params={
                    "stopPrice": signal.target,
                    "triggerPrice": signal.target,
                    "reduceOnly": True,
                    "marginMode": "cross",
                }
            )

            self.active_positions[signal.symbol] = {
                "side":      signal.side,
                "size":      size,
                "entry":     avg_price,
                "stop":      signal.stop,
                "target":    signal.target,
                "opened_at": time.time(),
            }
            self.trades_today   += 1
            self.cooldown_until  = time.time() + COOLDOWN_SEC

            sym_clean = signal.symbol.replace("/USDT:USDT", "")
            risk_actual = abs(avg_price - signal.stop) * size
            await self.tg(
                f"🚀 {sym_clean} {signal.side.upper()}\n"
                f"Entry:  {avg_price:.5f}\n"
                f"Stop:   {signal.stop:.5f}\n"
                f"Target: {signal.target:.5f}\n"
                f"Size:   {size} contracts\n"
                f"Risk:   ${risk_actual:.3f}\n"
                f"RR:     1:{RR_RATIO}\n"
                f"Reason: {signal.reason}"
            )
            logger.info(f"Trade opened: {signal.symbol} {signal.side} @ {avg_price}")

        except Exception as e:
            logger.error(f"Order failed {signal.symbol}: {e}")
            await self.tg(f"❌ Order failed {signal.symbol}: {str(e)[:200]}")

    async def reconcile(self):
        """Check if positions closed and update PnL"""
        if not self.exchange or not self.active_positions:
            return
        try:
            positions = await self.exchange.fetch_positions(SYMBOLS)
            live = {p["symbol"] for p in positions if float(p.get("contracts") or 0) > 0}

            for sym in list(self.active_positions.keys()):
                if sym not in live:
                    pos = self.active_positions.pop(sym)
                    pnl = 0.0
                    try:
                        history = await self.exchange.fetch_closed_orders(sym, limit=5)
                        if history:
                            pnl = sum(float(o.get("profit", 0) or 0) for o in history[-2:])
                    except Exception:
                        pass

                    if pnl < 0:
                        self.consec_losses += 1
                        self.losses_today  += 1
                        # Trigger pause if consecutive losses hit limit
                        if self.consec_losses >= MAX_CONSEC_LOSSES:
                            self.pause_until = time.time() + (CONSEC_LOSS_PAUSE_HR * 3600)
                            await self.tg(
                                f"⏸ {MAX_CONSEC_LOSSES} consecutive losses — "
                                f"pausing {CONSEC_LOSS_PAUSE_HR}h to protect balance"
                            )
                    else:
                        self.consec_losses = 0

                    self.profit_today = round(self.profit_today + pnl, 4)

                    self._save_trade({
                        "time":   time.time(),
                        "symbol": sym,
                        "side":   pos["side"],
                        "entry":  pos["entry"],
                        "stop":   pos["stop"],
                        "target": pos["target"],
                        "pnl":    pnl,
                    })

                    sym_clean = sym.replace("/USDT:USDT", "")
                    await self.tg(
                        f"{'✅ WIN' if pnl >= 0 else '❌ LOSS'} {sym_clean}\n"
                        f"PnL: {pnl:+.4f} USDT\n"
                        f"Today: {self.profit_today:+.4f} USDT | "
                        f"Streak: {self.consec_losses}/{MAX_CONSEC_LOSSES}"
                    )

        except Exception as e:
            logger.warning(f"Reconcile error: {e}")

    async def scan_symbol(self, symbol, stagger=0):
        await asyncio.sleep(stagger)
        self.market_debug[symbol] = {"time": time.time(), "why": "Initializing...", "signal": None}

        while self.is_scanning:
            try:
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                self._reset_daily()

                if not self.exchange:
                    self.market_debug[symbol] = {
                        "time": time.time(), "why": "Not connected", "signal": None}
                    await asyncio.sleep(5)
                    continue

                if symbol in self.active_positions:
                    continue

                can, gate = self.can_trade()
                if not can:
                    self.market_debug[symbol] = {
                        "time": time.time(), "why": gate, "signal": None}
                    continue

                self.market_debug[symbol] = {
                    "time": time.time(), "why": "Fetching candles...", "signal": None}

                # Sequential fetch to avoid rate limits
                c15 = await self.fetch_ohlcv(symbol, TF_TREND, CANDLES_15M)
                await asyncio.sleep(0.3)
                c5  = await self.fetch_ohlcv(symbol, TF_ENTRY,  CANDLES_5M)

                if not c15 or not c5:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "why": f"Candle fetch failed — 15M:{len(c15)} 5M:{len(c5)}. Check API.",
                        "signal": None}
                    continue

                signal, reason, dbg = self.build_signal(symbol, c15, c5)
                dbg["time"]   = time.time()
                dbg["why"]    = reason
                dbg["signal"] = signal.side.upper() if signal else None
                self.market_debug[symbol] = dbg

                if signal:
                    await self.place_trade(signal)

            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                self.market_debug[symbol] = {
                    "time": time.time(), "why": f"Error: {str(e)[:100]}", "signal": None}
                await asyncio.sleep(10)

    async def watchdog(self):
        while True:
            await asyncio.sleep(30)
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
bot = BitgetBot()


def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START",   callback_data="START"),
         InlineKeyboardButton("⏹️ STOP",    callback_data="STOP")],
        [InlineKeyboardButton("📊 STATUS",  callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🔌 CONNECT", callback_data="CONNECT")],
        [InlineKeyboardButton("⏸ PAUSE",   callback_data="PAUSE"),
         InlineKeyboardButton("▶️ RESUME",  callback_data="RESUME")],
        [InlineKeyboardButton("🧪 TESTNET", callback_data="TESTNET"),
         InlineKeyboardButton("💰 LIVE",    callback_data="LIVE")],
    ])


def fmt_debug(sym, d):
    if not d:
        return f"📍 {sym.replace('/USDT:USDT','')} ⏳ No data yet\n"
    age      = int(time.time() - d.get("time", time.time()))
    signal   = d.get("signal") or "—"
    why      = d.get("why", "—")
    trend    = d.get("trend", "—")
    ema200   = d.get("ema200_15", "—")
    slope    = d.get("ema_slope", 0)
    dist_pct = d.get("ema_dist_pct", "—")
    dist_ok  = "✅" if d.get("dist_ok") else "❌"
    candle_ok= "✅" if d.get("candle_ok") else "❌"
    bull_b   = "✅" if d.get("bull_break") else "⏳"
    bear_b   = "✅" if d.get("bear_break") else "⏳"
    body_ok  = "✅" if d.get("body_ok") else "❌"
    range_ok = "✅" if d.get("range_ok") else "❌"
    sym_c    = sym.replace("/USDT:USDT", "")
    return (
        f"📍 {sym_c} ({age}s)\n"
        f"15M: {trend} | EMA200:{ema200} Slope:{slope:.5f}\n"
        f"Dist: {dist_pct}% {dist_ok} | 15M Candle:{candle_ok}\n"
        f"5M:  Break Bull:{bull_b} Bear:{bear_b}\n"
        f"5M:  Body:{body_ok} | Range:{range_ok}\n"
        f"Signal: {signal} | {why}\n"
    )


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
        await safe_edit(q,
            f"{'✅ Connected' if ok else '❌ Failed'} | "
            f"{'TESTNET' if USE_TESTNET else 'LIVE'} | "
            f"Balance: {bot.balance_usdt:.2f} USDT",
            keyboard())

    elif q.data == "TESTNET":
        os.environ["BITGET_TESTNET"] = "true"
        await safe_edit(q, "🧪 Switched to TESTNET — reconnect to apply", keyboard())

    elif q.data == "LIVE":
        os.environ["BITGET_TESTNET"] = "false"
        await safe_edit(q, "💰 Switched to LIVE — reconnect to apply", keyboard())

    elif q.data == "START":
        if not bot.exchange:
            await safe_edit(q, "❌ Connect first", keyboard())
            return
        bot.is_scanning = True
        for i, sym in enumerate(SYMBOLS):
            asyncio.create_task(bot.scan_symbol(sym, stagger=i * 5))
        await safe_edit(q,
            f"🔍 Scanner active\n"
            f"Pairs: SOL DOGE XRP\n"
            f"Strategy: 15M EMA200 + 5M Breakout\n"
            f"Session: {SESSION_START_UTC:02d}:00–{SESSION_END_UTC:02d}:00 UTC\n"
            f"RR: 1:{RR_RATIO} | Leverage: {LEVERAGE}x\n"
            f"Risk/trade: ${RISK_PER_TRADE:.2f} fixed",
            keyboard())

    elif q.data == "STOP":
        bot.is_scanning = False
        await safe_edit(q, "⏹️ Scanner stopped", keyboard())

    elif q.data == "PAUSE":
        bot.pause_until = time.time() + 86400
        await safe_edit(q, "⏸ Paused 24h", keyboard())

    elif q.data == "RESUME":
        bot.pause_until   = 0
        bot.consec_losses = 0
        await safe_edit(q, "▶️ Resumed — consecutive loss counter reset", keyboard())

    elif q.data == "STATUS":
        now = time.time()
        if now < bot.status_cd_until:
            await safe_edit(q, f"⏳ {int(bot.status_cd_until-now)}s", keyboard())
            return
        bot.status_cd_until = now + STATUS_REFRESH_COOLDOWN
        await bot.fetch_balance()
        _, gate = bot.can_trade()

        open_pos = ""
        for sym, pos in bot.active_positions.items():
            sym_c = sym.replace("/USDT:USDT", "")
            age   = int(time.time() - pos["opened_at"])
            open_pos += (
                f"\n🔵 {sym_c} {pos['side'].upper()} @ {pos['entry']:.5f} "
                f"| SL:{pos['stop']:.5f} TP:{pos['target']:.5f} "
                f"| ({age}s)"
            )

        session_active = is_session_active()
        utc_hour = datetime.utcnow().hour

        header = (
            f"🕒 {now_wat()}\n"
            f"🤖 {'ACTIVE' if bot.is_scanning else 'OFFLINE'} | "
            f"{'🧪 TESTNET' if USE_TESTNET else '💰 LIVE'}\n"
            f"💰 Balance: {bot.balance_usdt:.2f} USDT\n"
            f"📈 PnL Today: {bot.profit_today:+.4f} USDT\n"
            f"🎯 RR 1:{RR_RATIO} | Risk ${RISK_PER_TRADE:.2f} | {LEVERAGE}x\n"
            f"📉 Streak: {bot.consec_losses}/{MAX_CONSEC_LOSSES} | "
            f"Trades: {bot.trades_today}/{MAX_TRADES_PER_DAY}\n"
            f"🕐 Session: {'✅ ACTIVE' if session_active else f'❌ CLOSED ({utc_hour:02d}:00 UTC)'}\n"
            f"🚦 Gate: {gate}"
        )

        if open_pos:
            header += f"\n\n📌 OPEN POSITIONS:{open_pos}"

        scan_lines = "\n\n📡 LIVE SCAN\n" + "\n".join(
            fmt_debug(sym, bot.market_debug.get(sym, {})) for sym in SYMBOLS)

        await safe_edit(q, header + scan_lines, keyboard())


async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    bot._chat_ids.add(update.message.chat_id)
    await update.message.reply_text(
        "💎 Bitget Crypto Futures Bot\n"
        "Strategy: 15M EMA200 Trend + 5M Breakout\n"
        "Pairs: SOL DOGE XRP perpetuals\n"
        f"RR: 1:{RR_RATIO} | Leverage: {LEVERAGE}x\n"
        f"Risk/trade: ${RISK_PER_TRADE:.2f} fixed\n"
        f"Session: {SESSION_START_UTC:02d}:00–{SESSION_END_UTC:02d}:00 UTC\n"
        "1. Press CONNECT\n"
        "2. Press START\n"
        "3. Press STATUS to monitor\n"
        "4. Send /symbols to check available pairs",
        reply_markup=keyboard()
    )


async def symbols_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List all SOL, DOGE, XRP USDT perpetual markets available on Bitget"""
    bot._chat_ids.add(update.message.chat_id)
    if not bot.exchange:
        await update.message.reply_text("❌ Connect first — press CONNECT button")
        return
    try:
        markets = bot.exchange.markets
        sol  = [s for s in markets.keys() if "SOL"  in s and "USDT" in s]
        doge = [s for s in markets.keys() if "DOGE" in s and "USDT" in s]
        xrp  = [s for s in markets.keys() if "XRP"  in s and "USDT" in s]
        msg = (
            f"📋 Available USDT Markets on Bitget\n\n"
            f"SOL pairs:\n" + "\n".join(sol[:10] or ["None found"]) + "\n\n"
            f"DOGE pairs:\n" + "\n".join(doge[:10] or ["None found"]) + "\n\n"
            f"XRP pairs:\n" + "\n".join(xrp[:10] or ["None found"])
        )
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
