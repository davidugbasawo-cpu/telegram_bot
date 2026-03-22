"""
Bitget Crypto Futures Bot
Strategy: Binary Trend Pullback (15M Trend + 5M Pullback Entry)
  15M → trend direction (EMA 50)
  5M  → pullback to EMA 50 + strong candle + ADX + RSI + Volume
Pairs: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT perpetual futures

Strategy rules:
- 15M EMA 50: price above = BUY only | price below = SELL only
- ADX (14) > 25: trend must be strong, skip if choppy
- RSI (14): BUY requires RSI > 50 | SELL requires RSI < 50
- 5M pullback: price retraces toward EMA 50 then strong candle fires
- Candle body >= 50% of range (no doji)
- Volume: entry candle must be above average volume
- Session filter: London + New York sessions only (UTC)
- RR: 1:4
- Fixed risk: $0.25 per trade
- Daily trade limit: 6
- Consecutive loss pause: 3 losses = 2 hour pause
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

USE_TESTNET = False  # Live mode only

# ========================= MARKETS =========================
SYMBOLS = ["SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT"]

# ========================= TIMEFRAMES =========================
TF_TREND    = "15m"
TF_ENTRY    = "5m"
CANDLES_15M = 150
CANDLES_5M  = 80

# ========================= INDICATORS =========================
EMA_PERIOD    = 50    # EMA 50 for trend direction
ADX_PERIOD    = 14    # ADX period
ADX_MIN       = 25    # ADX must be > 25 (strong trend)
RSI_PERIOD    = 14    # RSI period
VOLUME_LOOKBACK = 20  # Candles to calculate average volume

# ========================= PULLBACK FILTER =========================
PULLBACK_ZONE_PCT     = 0.003   # Price must come within 0.3% of EMA50 on 5M
CANDLE_BODY_RATIO_MIN = 0.50    # Entry candle body >= 50% of range
SLOPE_LOOKBACK        = 5       # Candles to measure EMA50 slope

# ========================= SESSION FILTER =========================
# London session: 08:00-16:00 UTC
# New York session: 13:00-21:00 UTC
SESSION_START_UTC = 8    # 08:00 UTC
SESSION_END_UTC   = 21   # 21:00 UTC

# ========================= RISK =========================
RISK_PER_TRADE = 0.25   # Fixed $0.25 per trade
LEVERAGE       = 3
RR_RATIO       = 4.0    # 1:4 risk/reward
MIN_SL_PCT     = 0.0015 # Minimum SL distance = 0.15% of entry

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


def rsi_value(closes, period=14):
    """Calculate RSI from closes list"""
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
    """Calculate ADX from candles list [[ts,o,h,l,c,v],...]"""
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

    atr  = smooth(tr_list,  period)
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
    adx = float(np.mean(dx_list[-period:]))
    return adx


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
            await self.exchange.load_markets()
            await self.fetch_balance()
            logger.info(f"Connected to Bitget LIVE")
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
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                data = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                if not data:
                    logger.warning(f"OHLCV {symbol} {tf}: Empty response (attempt {attempt}/{max_retries})")
                    await asyncio.sleep(2 * attempt)
                    continue
                # Drop last (open/unconfirmed) candle
                return data[:-1] if len(data) > 1 else data
            except Exception as e:
                err = str(e)
                logger.warning(f"OHLCV {symbol} {tf} attempt {attempt}/{max_retries}: {err}")
                # Rate limit — wait longer
                if "rate limit" in err.lower() or "429" in err:
                    await asyncio.sleep(5 * attempt)
                elif attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
        logger.error(f"OHLCV {symbol} {tf}: All {max_retries} attempts failed")
        return []

    def build_signal(self, symbol, c15, c5):
        """
        Returns (Signal | None, reason_str, debug_dict)

        Logic:
        1. 15M EMA 50 direction → trend (UP/DOWN)
        2. 15M ADX > 25 → trend is strong enough
        3. 5M: price pulled back to within 0.3% of EMA 50
        4. 5M: strong candle fires in trend direction (body >= 50%)
        5. 5M: RSI > 50 for BUY | RSI < 50 for SELL
        6. 5M: entry candle volume > average volume
        7. SL: below/above entry candle low/high
        8. TP: entry + risk * 4 (1:4 RR)
        """
        dbg = {}

        # ── 15M TREND ──────────────────────────────────────
        if len(c15) < EMA_PERIOD + SLOPE_LOOKBACK + 5:
            return None, "15M warming up", dbg

        cl15   = [x[4] for x in c15]
        price15 = cl15[-1]

        ema50_15 = ema_value(cl15, EMA_PERIOD)
        if ema50_15 is None:
            return None, "EMA50 not ready", dbg

        # Slope check
        ema50_old = ema_value(cl15[:-SLOPE_LOOKBACK], EMA_PERIOD)
        ema_slope = (ema50_15 - ema50_old) if ema50_old else 0.0

        trend_up   = price15 > ema50_15 and ema_slope > 0
        trend_down = price15 < ema50_15 and ema_slope < 0

        dbg["ema50_15"] = round(ema50_15, 5)
        dbg["price15"]  = round(price15, 5)
        dbg["ema_slope"]= round(ema_slope, 6)
        dbg["trend"]    = "UP" if trend_up else "DOWN" if trend_down else "SIDE"

        if not trend_up and not trend_down:
            return None, f"15M sideways | Price:{price15:.4f} EMA50:{ema50_15:.4f}", dbg

        # ── ADX FILTER (15M) ───────────────────────────────
        adx = adx_value(c15, ADX_PERIOD)
        adx_ok = adx is not None and adx > ADX_MIN
        dbg["adx"]    = round(adx, 2) if adx else "N/A"
        dbg["adx_ok"] = adx_ok

        if not adx_ok:
            return None, f"ADX too low ({dbg['adx']} <= {ADX_MIN}) — market choppy", dbg

        # ── 5M ENTRY ───────────────────────────────────────
        if len(c5) < RSI_PERIOD + VOLUME_LOOKBACK + 5:
            return None, "5M warming up", dbg

        cl5   = [x[4] for x in c5]
        vol5  = [x[5] for x in c5]
        cur5  = c5[-1]
        prev5 = c5[-2]

        cur_open  = cur5[1]
        cur_high  = cur5[2]
        cur_low   = cur5[3]
        cur_close = cur5[4]
        cur_vol   = cur5[5]

        # EMA 50 on 5M for pullback zone
        ema50_5m = ema_value(cl5, EMA_PERIOD)
        if ema50_5m is None:
            return None, "5M EMA50 not ready", dbg

        # RSI on 5M
        rsi = rsi_value(cl5, RSI_PERIOD)
        rsi_ok_buy  = rsi is not None and rsi > 50
        rsi_ok_sell = rsi is not None and rsi < 50

        # Volume filter — current candle volume > average of last N candles
        avg_vol  = float(np.mean(vol5[-(VOLUME_LOOKBACK + 1):-1])) if len(vol5) > VOLUME_LOOKBACK else 0
        vol_ok   = avg_vol == 0 or cur_vol > avg_vol

        # Candle body
        cur_body = candle_body_ratio(cur5)
        body_ok  = cur_body >= CANDLE_BODY_RATIO_MIN

        # Candle direction
        bull_candle = cur_close > cur_open
        bear_candle = cur_close < cur_open

        # Pullback check — previous candle came within 0.3% of 5M EMA50
        prev_low  = prev5[3]
        prev_high = prev5[2]
        pullback_bull = abs(prev_low  - ema50_5m) / ema50_5m <= PULLBACK_ZONE_PCT
        pullback_bear = abs(prev_high - ema50_5m) / ema50_5m <= PULLBACK_ZONE_PCT

        dbg["ema50_5m"]      = round(ema50_5m, 5)
        dbg["rsi"]           = round(rsi, 2) if rsi else "N/A"
        dbg["rsi_ok_buy"]    = rsi_ok_buy
        dbg["rsi_ok_sell"]   = rsi_ok_sell
        dbg["vol_ok"]        = vol_ok
        dbg["cur_vol"]       = round(cur_vol, 2)
        dbg["avg_vol"]       = round(avg_vol, 2)
        dbg["body_ok"]       = body_ok
        dbg["pullback_bull"] = pullback_bull
        dbg["pullback_bear"] = pullback_bear

        # ── LONG SETUP ─────────────────────────────────────
        if trend_up and pullback_bull and bull_candle and body_ok and rsi_ok_buy and vol_ok:
            entry = float(cur_close)
            stop  = float(cur_low)

            min_sl = entry * MIN_SL_PCT
            if (entry - stop) < min_sl:
                stop = entry - min_sl

            if stop >= entry:
                return None, "Invalid long SL >= entry", dbg

            risk   = entry - stop
            target = entry + risk * RR_RATIO

            return Signal(
                "buy", symbol, entry, stop, target,
                f"15M UP | ADX:{dbg['adx']} | RSI:{dbg['rsi']} | Pullback+BullCandle | Body:{cur_body:.2f}"
            ), "LONG ✅", dbg

        # ── SHORT SETUP ────────────────────────────────────
        if trend_down and pullback_bear and bear_candle and body_ok and rsi_ok_sell and vol_ok:
            entry = float(cur_close)
            stop  = float(cur_high)

            min_sl = entry * MIN_SL_PCT
            if (stop - entry) < min_sl:
                stop = entry + min_sl

            if stop <= entry:
                return None, "Invalid short SL <= entry", dbg

            risk   = stop - entry
            target = entry - risk * RR_RATIO

            return Signal(
                "sell", symbol, entry, stop, target,
                f"15M DOWN | ADX:{dbg['adx']} | RSI:{dbg['rsi']} | Pullback+BearCandle | Body:{cur_body:.2f}"
            ), "SHORT ✅", dbg

        # ── NO SIGNAL — detailed reason ────────────────────
        if trend_up:
            if not pullback_bull:
                reason = f"Waiting pullback to 5M EMA50 ({ema50_5m:.4f}) — cur low:{prev_low:.4f}"
            elif not bull_candle:
                reason = "Pullback zone hit but no bullish candle yet"
            elif not body_ok:
                reason = f"Weak candle body {cur_body:.2f} < {CANDLE_BODY_RATIO_MIN}"
            elif not rsi_ok_buy:
                reason = f"RSI {dbg['rsi']} <= 50 — no buy momentum"
            elif not vol_ok:
                reason = f"Low volume — cur:{cur_vol:.0f} avg:{avg_vol:.0f}"
            else:
                reason = "Waiting for setup"
        elif trend_down:
            if not pullback_bear:
                reason = f"Waiting pullback to 5M EMA50 ({ema50_5m:.4f}) — cur high:{prev_high:.4f}"
            elif not bear_candle:
                reason = "Pullback zone hit but no bearish candle yet"
            elif not body_ok:
                reason = f"Weak candle body {cur_body:.2f} < {CANDLE_BODY_RATIO_MIN}"
            elif not rsi_ok_sell:
                reason = f"RSI {dbg['rsi']} >= 50 — no sell momentum"
            elif not vol_ok:
                reason = f"Low volume — cur:{cur_vol:.0f} avg:{avg_vol:.0f}"
            else:
                reason = "Waiting for setup"
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
                    fail_reason = []
                    if not c15:
                        fail_reason.append(f"15M: 0 candles")
                    if not c5:
                        fail_reason.append(f"5M: 0 candles")
                    msg = " | ".join(fail_reason) + " — possible rate limit. Retrying..."
                    logger.warning(f"{symbol}: {msg}")
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "why": msg,
                        "signal": None}
                    await asyncio.sleep(10)
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
    ])


def fmt_debug(sym, d):
    if not d:
        return f"📍 {sym.replace('/USDT:USDT','')} ⏳ No data yet\n"
    age          = int(time.time() - d.get("time", time.time()))
    signal       = d.get("signal") or "—"
    why          = d.get("why", "—")
    trend        = d.get("trend", "—")
    ema50_15     = d.get("ema50_15", "—")
    ema50_5m     = d.get("ema50_5m", "—")
    slope        = d.get("ema_slope", 0)
    adx          = d.get("adx", "—")
    adx_ok       = "✅" if d.get("adx_ok") else "❌"
    rsi          = d.get("rsi", "—")
    rsi_buy      = "✅" if d.get("rsi_ok_buy") else "—"
    rsi_sell     = "✅" if d.get("rsi_ok_sell") else "—"
    vol_ok       = "✅" if d.get("vol_ok") else "❌"
    body_ok      = "✅" if d.get("body_ok") else "❌"
    pb_bull      = "✅" if d.get("pullback_bull") else "⏳"
    pb_bear      = "✅" if d.get("pullback_bear") else "⏳"
    sym_c        = sym.replace("/USDT:USDT", "")
    return (
        f"📍 {sym_c} ({age}s)\n"
        f"15M: {trend} | EMA50:{ema50_15} Slope:{slope:.5f}\n"
        f"ADX: {adx} {adx_ok} | RSI: {rsi} Buy:{rsi_buy} Sell:{rsi_sell}\n"
        f"5M:  EMA50:{ema50_5m} | Pullback Bull:{pb_bull} Bear:{pb_bear}\n"
        f"Body:{body_ok} | Vol:{vol_ok}\n"
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
            f"LIVE | "
            f"Balance: {bot.balance_usdt:.2f} USDT",
            keyboard())

    elif q.data == "START":
        if not bot.exchange:
            await safe_edit(q, "❌ Connect first", keyboard())
            return
        bot.is_scanning = True
        for i, sym in enumerate(SYMBOLS):
            asyncio.create_task(bot.scan_symbol(sym, stagger=i * 12))
        await safe_edit(q,
            f"🔍 Scanner active\n"
            f"Pairs: SOL XRP ADA\n"
            f"Strategy: 15M EMA50 Trend + 5M Pullback\n"
            f"Filters: ADX>{ADX_MIN} | RSI50 | Volume | Body>=50%\n"
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
            f"🤖 {'ACTIVE' if bot.is_scanning else 'OFFLINE'} | 💰 LIVE\n"
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
        "Strategy: 15M EMA50 Trend + 5M Pullback\n"
        f"Filters: ADX>{ADX_MIN} | RSI50 | Volume | Body>=50%\n"
        "Pairs: SOL XRP ADA perpetuals\n"
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
        xrp  = [s for s in markets.keys() if "XRP"  in s and "USDT" in s]
        ada  = [s for s in markets.keys() if "ADA"  in s and "USDT" in s]
        msg = (
            f"📋 Available USDT Markets on Bitget\n\n"
            f"SOL pairs:\n" + "\n".join(sol[:10] or ["None found"]) + "\n\n"
            f"XRP pairs:\n" + "\n".join(xrp[:10] or ["None found"]) + "\n\n"
            f"ADA pairs:\n" + "\n".join(ada[:10] or ["None found"])
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
