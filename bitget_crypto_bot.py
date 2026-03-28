"""
Bitget Crypto Futures Bot – Enhanced EMA Cross (1m + 5m, 1:2 RR, Trailing Stop)
- 5m trend filter, recent cross detection, continuation candle
- Adaptive ATR, EMA gap, volume spike
- Stop: swing low/ATR, break-even after 1R, trailing stop
- Paper mode enabled
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

USE_TESTNET = False
PAPER_MODE  = True

# ========================= MARKETS =========================
SYMBOLS = ["SOL/USDT:USDT", "BTC/USDT:USDT", "ADA/USDT:USDT"]

# ========================= TIMEFRAMES =========================
TF_1M = "1m"
TF_5M = "5m"
CANDLES_1M = 200
CANDLES_5M = 100

# ========================= INDICATORS =========================
EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14
VOLUME_LOOKBACK = 20

# ========================= RECENT CROSS =========================
RECENT_CROSS_LOOKBACK = 3
MIN_EMA_GAP_PCT = 0.04

# ========================= ENTRY FILTERS =========================
CANDLE_BODY_RATIO_MIN = 0.50
VOLUME_MULT = 1.2
STOP_ATR_MULT = 0.7

# ========================= ADAPTIVE ATR =========================
ATR_AVG_LOOKBACK = 20
ATR_MIN_RATIO = 0.9

# ========================= RISK/REWARD =========================
RR_RATIO = 2.0
BASE_RISK_PER_TRADE = 0.25   # change to 1.0 for live if you wish

# ========================= BREAK-EVEN & TRAILING =========================
BREAK_EVEN_AFTER_1R = True
TRAIL_ACTIVATION_PCT = 0.35
TRAIL_DISTANCE_PCT = 0.20

# ========================= SESSION =========================
SESSION_START_UTC = 8
SESSION_END_UTC   = 21

# ========================= LIMITS =========================
MAX_TRADES_PER_DAY   = 20
MAX_CONSEC_LOSSES    = 5
CONSEC_LOSS_PAUSE_HR = 24
COOLDOWN_SEC         = 60
MAX_OPEN_POSITIONS   = 3

# ========================= OTHER =========================
TRADE_LOG_FILE          = "enhanced_trades.json"
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

def candle_body_ratio(candle):
    rng = abs(candle[2] - candle[3])
    if rng == 0:
        return 0.0
    return abs(candle[4] - candle[1]) / rng

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    highs = [c[2] for c in candles[-period-1:]]
    lows = [c[3] for c in candles[-period-1:]]
    closes = [c[4] for c in candles[-period-2:-1]]
    tr_values = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period

def is_session_active():
    utc_hour = datetime.utcnow().hour
    return SESSION_START_UTC <= utc_hour < SESSION_END_UTC

def now_wat():
    return datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class Signal:
    side:     str
    symbol:   str
    entry:    float
    stop:     float
    target:   float
    reason:   str
    rr_ratio: float

# ========================= BOT =========================
class EnhancedBot:
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
            "SOL/USDT:USDT": {"wins": 0, "losses": 0},
            "BTC/USDT:USDT": {"wins": 0, "losses": 0},
            "ADA/USDT:USDT": {"wins": 0, "losses": 0},
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
        if symbol in self.win_rate_stats:
            if result == "WIN":
                self.win_rate_stats[symbol]["wins"] += 1
            elif result == "LOSS":
                self.win_rate_stats[symbol]["losses"] += 1

    def _reset_daily(self):
        today = datetime.now(ZoneInfo(TIMEZONE)).date()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.trades_today    = 0
            self.profit_today    = 0.0
            self.losses_today    = 0
            self.consec_losses   = 0
            logger.info(f"Daily reset: {today}")

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

    # ------------------------------------------------------------------
    # ENHANCED SIGNAL BUILDER (5m + 1m, recent cross, continuation)
    # ------------------------------------------------------------------
    def build_signal(self, symbol, candles_1m, candles_5m):
        dbg = {}
        if len(candles_1m) < EMA_SLOW + RECENT_CROSS_LOOKBACK + 5:
            return None, "Not enough 1m data", dbg
        if len(candles_5m) < EMA_SLOW + 5:
            return None, "Not enough 5m data", dbg

        # ----- 5m trend filter -----
        closes_5m = [c[4] for c in candles_5m]
        ema9_5m = ema_value(closes_5m, EMA_FAST)
        ema21_5m = ema_value(closes_5m, EMA_SLOW)
        if ema9_5m is None or ema21_5m is None:
            return None, "5m EMAs not ready", dbg
        trend_up_5m = ema9_5m > ema21_5m
        trend_down_5m = ema9_5m < ema21_5m

        # ----- 1m data -----
        closes = [c[4] for c in candles_1m]
        volumes = [c[5] for c in candles_1m]
        current_price = closes[-1]

        # EMAs
        ema9 = ema_value(closes, EMA_FAST)
        ema21 = ema_value(closes, EMA_SLOW)
        if ema9 is None or ema21 is None:
            return None, "1m EMAs not ready", dbg

        # EMA gap filter
        ema_gap_pct = abs(ema9 - ema21) / current_price * 100
        if ema_gap_pct < MIN_EMA_GAP_PCT:
            return None, f"EMA gap too small ({ema_gap_pct:.2f}% < {MIN_EMA_GAP_PCT}%)", dbg

        # Recent cross detection
        cross_up_recent = False
        cross_down_recent = False
        for i in range(1, RECENT_CROSS_LOOKBACK + 1):
            if len(closes) < i + 2:
                continue
            ema9_prev = ema_value(closes[:-i], EMA_FAST)
            ema21_prev = ema_value(closes[:-i], EMA_SLOW)
            if ema9_prev is None or ema21_prev is None:
                continue
            if ema9_prev <= ema21_prev and ema9 > ema21:
                cross_up_recent = True
            if ema9_prev >= ema21_prev and ema9 < ema21:
                cross_down_recent = True

        # Continuation candle (last closed candle)
        conf_candle = candles_1m[-1]
        conf_body_ratio = candle_body_ratio(conf_candle)
        conf_vol = conf_candle[5]
        avg_vol = sum(volumes[-VOLUME_LOOKBACK-1:-1]) / VOLUME_LOOKBACK if len(volumes) > VOLUME_LOOKBACK else 0
        vol_ok = conf_vol > avg_vol * VOLUME_MULT
        conf_bull = conf_candle[4] > conf_candle[1]
        conf_bear = conf_candle[4] < conf_candle[1]
        conf_above_ema9 = conf_candle[4] > ema9
        conf_below_ema9 = conf_candle[4] < ema9

        # ATR & adaptive volatility
        atr = calculate_atr(candles_1m, ATR_PERIOD)
        if atr is None:
            return None, "ATR not ready", dbg

        if len(candles_1m) >= ATR_AVG_LOOKBACK + 2:
            atr_vals = []
            for i in range(ATR_AVG_LOOKBACK):
                sub_candles = candles_1m[-(ATR_AVG_LOOKBACK+2+i):-i-1] if i > 0 else candles_1m[-ATR_AVG_LOOKBACK-2:-1]
                sub_atr = calculate_atr(sub_candles, ATR_PERIOD)
                if sub_atr is not None:
                    atr_vals.append(sub_atr)
            avg_atr = sum(atr_vals) / len(atr_vals) if atr_vals else atr
            if atr < avg_atr * ATR_MIN_RATIO:
                return None, f"Low volatility (ATR {atr:.5f} < {ATR_MIN_RATIO*100:.0f}% of avg {avg_atr:.5f})", dbg

        # Recent swing levels (last 5 candles)
        recent_lows = [c[3] for c in candles_1m[-6:-1]]
        recent_highs = [c[2] for c in candles_1m[-6:-1]]
        swing_low = min(recent_lows)
        swing_high = max(recent_highs)

        # ----- Buy signal -----
        if (trend_up_5m and cross_up_recent and conf_bull and conf_above_ema9 and
            conf_body_ratio >= CANDLE_BODY_RATIO_MIN and vol_ok):
            entry = conf_candle[4]
            # Stop: min of swing low and ATR-based level
            stop_candidate = swing_low - (atr * STOP_ATR_MULT)
            stop = min(swing_low, stop_candidate)
            # Minimum distance protection
            if entry - stop < entry * 0.0015:
                stop = entry - (entry * 0.0015)
            if stop >= entry:
                return None, "Invalid stop", dbg
            risk = entry - stop
            target = entry + risk * RR_RATIO
            return Signal(
                "buy", symbol, entry, stop, target,
                f"5m up | recent cross | strong cont | vol spike",
                RR_RATIO
            ), f"LONG ✅ (1:{RR_RATIO:.0f})", dbg

        # ----- Sell signal -----
        if (trend_down_5m and cross_down_recent and conf_bear and conf_below_ema9 and
            conf_body_ratio >= CANDLE_BODY_RATIO_MIN and vol_ok):
            entry = conf_candle[4]
            stop_candidate = swing_high + (atr * STOP_ATR_MULT)
            stop = max(swing_high, stop_candidate)
            if stop - entry < entry * 0.0015:
                stop = entry + (entry * 0.0015)
            if stop <= entry:
                return None, "Invalid stop", dbg
            risk = stop - entry
            target = entry - risk * RR_RATIO
            return Signal(
                "sell", symbol, entry, stop, target,
                f"5m down | recent cross | strong cont | vol spike",
                RR_RATIO
            ), f"SHORT ✅ (1:{RR_RATIO:.0f})", dbg

        # No trade
        return None, f"Waiting (5m: {'up' if trend_up_5m else 'down' if trend_down_5m else 'side'}, cross_up:{cross_up_recent}, cross_down:{cross_down_recent})", dbg

    async def set_leverage(self, symbol):
        try:
            await self.exchange.set_leverage(3, symbol, params={"marginMode": "cross"})
        except Exception as e:
            logger.warning(f"Leverage set failed {symbol}: {e}")

    async def get_size(self, symbol, entry, stop):
        await self.fetch_balance()
        risk_amount = BASE_RISK_PER_TRADE
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
        self.paper_positions[signal.symbol] = {
            "side": signal.side, "entry": signal.entry, "stop": signal.stop,
            "target": signal.target, "opened_at": time.time(),
            "opened_candle_ts": entry_candle_ts, "size": 0.0,
            "reason": signal.reason, "rr_ratio": signal.rr_ratio,
            "highest_price": signal.entry,   # for trailing
            "lowest_price": signal.entry,
            "breakeven_activated": False,
        }
        self.trades_today += 1
        self.cooldown_until = time.time() + COOLDOWN_SEC
        await self.tg(
            f"📝 PAPER TRADE OPENED\n"
            f"Pair: {sym_clean} | {signal.side.upper()}\n"
            f"Entry: {signal.entry:.5f}\nStop: {signal.stop:.5f}\nTarget: {signal.target:.5f}\n"
            f"Risk: ${BASE_RISK_PER_TRADE:.2f} | RR: 1:{signal.rr_ratio:.1f}\nReason: {signal.reason}"
        )
        logger.info(f"PAPER trade opened: {signal.symbol} {signal.side} @ {signal.entry}")

    async def place_live_trade(self, signal: Signal):
        await self.set_leverage(signal.symbol)
        size = await self.get_size(signal.symbol, signal.entry, signal.stop)
        if size <= 0:
            await self.tg(f"⚠️ {signal.symbol} size invalid — skipping")
            return
        opposite = "sell" if signal.side == "buy" else "buy"
        hold_side = "long" if signal.side == "buy" else "short"
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
                "rr_ratio": signal.rr_ratio,
                "highest_price": avg_price,
                "lowest_price": avg_price,
                "breakeven_activated": False,
            }
            self.trades_today += 1
            self.cooldown_until = time.time() + COOLDOWN_SEC
            sym_clean = signal.symbol.replace("/USDT:USDT", "")
            risk_actual = abs(avg_price - signal.stop) * size
            await self.tg(
                f"🚀 LIVE TRADE OPENED\nPair: {sym_clean} | {signal.side.upper()}\n"
                f"Entry: {avg_price:.5f}\nStop: {signal.stop:.5f}\nTarget: {signal.target:.5f}\n"
                f"Size: {size} contracts\nRisk: ${risk_actual:.3f} | RR: 1:{signal.rr_ratio:.1f}\n"
                f"Reason: {signal.reason}"
            )
            logger.info(f"LIVE trade opened: {signal.symbol} {signal.side} @ {avg_price}")
        except Exception as e:
            logger.error(f"Order failed {signal.symbol}: {e}")
            await self.tg(f"❌ Order failed {signal.symbol}: {str(e)[:250]}")

    async def place_trade(self, signal: Signal, entry_candle_ts=None):
        if PAPER_MODE:
            await self.place_paper_trade(signal, entry_candle_ts)
        else:
            await self.place_live_trade(signal)

    # ------------------------------------------------------------------
    # RECONCILIATION (break-even + trailing stop)
    # ------------------------------------------------------------------
    async def reconcile_paper_positions(self):
        if not self.exchange or not self.paper_positions:
            return
        for sym in list(self.paper_positions.keys()):
            try:
                pos = self.paper_positions[sym]
                candles = await self.fetch_ohlcv(sym, TF_1M, 4)
                if not candles or len(candles) < 2:
                    continue
                last = candles[-1]
                last_ts = last[0]
                if pos.get("opened_candle_ts") == last_ts:
                    continue
                high = float(last[2])
                low = float(last[3])
                current_price = (high + low) / 2

                side = pos["side"]
                entry = pos["entry"]
                stop = pos["stop"]
                target = pos["target"]
                rr_ratio = pos.get("rr_ratio", RR_RATIO)
                risk_amount = BASE_RISK_PER_TRADE

                # Update highest/lowest
                if side == "buy":
                    highest = pos.get("highest_price", entry)
                    if current_price > highest:
                        highest = current_price
                        pos["highest_price"] = highest
                    profit_pct = (highest - entry) / entry * 100

                    # Break-even after 1R
                    if BREAK_EVEN_AFTER_1R and not pos.get("breakeven_activated", False):
                        risk_pct = (entry - stop) / entry * 100
                        if profit_pct >= risk_pct:
                            # Move stop to entry
                            new_stop = entry
                            pos["stop"] = new_stop
                            pos["breakeven_activated"] = True
                            await self.tg(f"🔒 {sym.replace('/USDT:USDT','')} BE activated → stop moved to entry")
                    # Trailing stop
                    if profit_pct >= TRAIL_ACTIVATION_PCT:
                        trail_stop = highest * (1 - TRAIL_DISTANCE_PCT / 100)
                        if trail_stop > stop:
                            pos["stop"] = trail_stop

                else:  # short
                    lowest = pos.get("lowest_price", entry)
                    if current_price < lowest:
                        lowest = current_price
                        pos["lowest_price"] = lowest
                    profit_pct = (entry - lowest) / entry * 100

                    if BREAK_EVEN_AFTER_1R and not pos.get("breakeven_activated", False):
                        risk_pct = (stop - entry) / entry * 100
                        if profit_pct >= risk_pct:
                            new_stop = entry
                            pos["stop"] = new_stop
                            pos["breakeven_activated"] = True
                            await self.tg(f"🔒 {sym.replace('/USDT:USDT','')} BE activated → stop moved to entry")
                    if profit_pct >= TRAIL_ACTIVATION_PCT:
                        trail_stop = lowest * (1 + TRAIL_DISTANCE_PCT / 100)
                        if trail_stop < stop:
                            pos["stop"] = trail_stop

                # Use updated stop for checking
                new_stop = pos["stop"]
                # Check stop/target
                result = None
                pnl = 0.0
                if side == "buy":
                    if low <= new_stop:
                        result = "LOSS"
                        pnl = -risk_amount
                    elif high >= target:
                        result = "WIN"
                        pnl = risk_amount * rr_ratio
                else:
                    if high >= new_stop:
                        result = "LOSS"
                        pnl = -risk_amount
                    elif low <= target:
                        result = "WIN"
                        pnl = risk_amount * rr_ratio

                if result is None:
                    # Update position in dict if stop changed
                    if new_stop != stop:
                        self.paper_positions[sym] = pos
                    continue

                # Trade closed
                self.paper_positions.pop(sym, None)
                self._close_trade(sym, pnl, result, rr_ratio, side, entry, stop, target)

            except Exception as e:
                logger.warning(f"Paper reconcile error {sym}: {e}")

    def _close_trade(self, sym, pnl, result, rr_ratio, side, entry, stop, target):
        if pnl < 0:
            self.consec_losses += 1
            self.losses_today += 1
            logger.warning(f"Consecutive losses: {self.consec_losses}/{MAX_CONSEC_LOSSES}")
            if self.consec_losses >= MAX_CONSEC_LOSSES:
                self.pause_until = time.time() + (CONSEC_LOSS_PAUSE_HR * 3600)
                asyncio.create_task(self.tg(
                    f"⏸️ {MAX_CONSEC_LOSSES} CONSECUTIVE LOSSES — PAUSING FOR THE DAY (24h)\n"
                    f"Loss streak: {self.consec_losses}\nToday's PnL: {self.profit_today:+.4f} USDT"
                ))
        else:
            self.consec_losses = 0

        self.profit_today = round(self.profit_today + pnl, 4)
        self._save_trade({
            "time": time.time(),
            "symbol": sym,
            "side": side,
            "entry": entry,
            "stop": stop,
            "target": target,
            "pnl": pnl,
            "mode": "paper",
            "result": result,
            "rr_ratio": rr_ratio,
        })
        sym_clean = sym.replace("/USDT:USDT", "")
        stats = self.win_rate_stats.get(sym, {"wins": 0, "losses": 0})
        total = stats["wins"] + stats["losses"]
        wr = (stats["wins"] / total * 100) if total > 0 else 0
        asyncio.create_task(self.tg(
            f"{'✅ PAPER WIN' if pnl >= 0 else '❌ PAPER LOSS'} | {sym_clean}\n"
            f"PnL: {pnl:+.4f} USDT (1:{rr_ratio:.1f} RR)\n"
            f"Today: {self.profit_today:+.4f} USDT\n"
            f"Streak: {self.consec_losses}/{MAX_CONSEC_LOSSES}\n"
            f"Win Rate {sym_clean}: {wr:.1f}% ({stats['wins']}/{total})"
        ))

    async def reconcile_live_positions(self):
        # For live, similar logic would be needed but we keep placeholder.
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

                # Fetch 1m and 5m candles
                c1 = await self.fetch_ohlcv(symbol, TF_1M, CANDLES_1M)
                await asyncio.sleep(0.2)
                c5 = await self.fetch_ohlcv(symbol, TF_5M, CANDLES_5M)
                if not c1 or not c5:
                    self.market_debug[symbol] = {"time": time.time(), "why": "No candles", "signal": None}
                    await asyncio.sleep(10)
                    continue

                signal, reason, dbg = self.build_signal(symbol, c1, c5)
                dbg["time"] = time.time()
                dbg["why"] = reason
                dbg["signal"] = signal.side.upper() if signal else None
                self.market_debug[symbol] = dbg

                if signal:
                    entry_candle_ts = c1[-1][0]
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
bot = EnhancedBot()

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
    sym_c = sym.replace("/USDT:USDT", "")
    return (f"📍 {sym_c} ({age}s)\n"
            f"Signal: {signal} | {why[:80]}\n")

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
            f"🔍 ENHANCED EMA CROSS SCANNER (1m+5m, 1:2 RR)\n"
            f"Mode: {mode}\nPairs: SOL | BTC | ADA\n"
            f"Strategy:\n"
            f"  📈 5m trend filter\n"
            f"  ⚡ Recent cross + continuation candle\n"
            f"  🔒 Break-even after 1R, trailing stop\n"
            f"  🌊 Adaptive ATR, EMA gap, volume spike\n"
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
        msg = "📊 **WIN RATES BY SYMBOL**\n\n"
        for sym, stats in bot.win_rate_stats.items():
            sym_clean = sym.replace("/USDT:USDT", "")
            wins = stats["wins"]
            losses = stats["losses"]
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            msg += f"**{sym_clean}**\n  Overall: {win_rate:.1f}% ({wins}/{total})\n\n"
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
                open_pos += (f"\n{pnl_status} {sym_c} {side.upper()}\n"
                             f"   Entry: {entry:.5f} | Current: {current_price:.5f}\n"
                             f"   PnL: {pnl_pct:+.2f}% | SL: {stop:.5f} | TP: {target:.5f}\n"
                             f"   To Stop: {dist_stop:.2f}% | To Target: {dist_target:.2f}%\n"
                             f"   Age: {age//3600}h{(age%3600)//60}m")
            else:
                open_pos += f"\n🔵 {sym_c} {side.upper()} @ {entry:.5f} | SL:{stop:.5f} TP:{target:.5f} | Age: {age//3600}h\n   ⏳ Price unavailable"
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
        "💎 Enhanced EMA Cross Bot (1m+5m, 1:2 RR, Trailing Stop)\n"
        f"Mode: {mode}\n\n"
        "**STRATEGY:**\n"
        "📈 5m trend filter (EMA9/21)\n"
        "⚡ Recent 1m EMA cross + continuation candle\n"
        "🔒 Break-even after 1R, then trailing stop\n"
        "🌊 Adaptive ATR, EMA gap, volume spike\n\n"
        f"**RISK:** Stop after {MAX_CONSEC_LOSSES} losses, max {MAX_OPEN_POSITIONS} positions\n"
        f"Session: {SESSION_START_UTC:02d}:00–{SESSION_END_UTC:02d}:00 UTC\n\n"
        "**USE:** CONNECT → START → STATUS",
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
