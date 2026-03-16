"""
Bitget Crypto Futures Bot
Strategy: 15M/5M/1M Trend Pullback
  15M → trend direction (EMA50/EMA200 + slope)
  5M  → pullback quality (EMA20 near-touch + EMA50 slope + RSI)
  1M  → entry trigger (engulfing/rejection + close vs EMA20 + body + spike)
Pairs: BTCUSDT, ETHUSDT, SOLUSDT perpetual futures
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
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

# ========================= TIMEFRAMES =========================
TF_TREND   = "15m"
TF_SETUP   = "5m"
TF_TRIGGER = "1m"
CANDLES_15M = 250
CANDLES_5M  = 150
CANDLES_1M  = 60

# ========================= INDICATORS =========================
EMA_TREND_FAST     = 50
EMA_TREND_SLOW     = 200
EMA_PULLBACK       = 20
EMA_SETUP_SLOW     = 50
EMA_SLOPE_LOOKBACK = 10
RSI_PERIOD         = 14
ATR_PERIOD         = 14

# ========================= FILTERS =========================
RSI_MIN              = 40     # both directions
RSI_MAX              = 65     # both directions
EMA_SPREAD_ATR_MULT  = 0.20   # min EMA20/50 spread vs ATR
PULLBACK_ATR_BUFFER  = 0.30   # near-touch: within 0.3x ATR counts
BODY_RATIO_MIN       = 0.32   # min body ratio — no dojis
SPIKE_MULT           = 1.80   # spike block multiplier
SPIKE_LOOKBACK       = 20     # candles for avg body

# ========================= RISK =========================
RISK_PER_TRADE    = 0.01   # 1% of balance per trade
LEVERAGE          = 3
RR_RATIO          = 4.0    # 1:4 risk/reward
BREAKEVEN_RR      = 1.5    # move SL to entry when up 1.5x risk

# ========================= LIMITS =========================
MAX_TRADES_PER_DAY  = 6
MAX_CONSEC_LOSSES   = 3
COOLDOWN_SEC        = 300   # 5 min between trades
MAX_OPEN_POSITIONS  = 1

# ========================= OTHER =========================
TRADE_LOG_FILE            = "bitget_trades.json"
SCAN_INTERVAL_SEC         = 20
STATUS_REFRESH_COOLDOWN   = 10
TIMEZONE                  = "Africa/Lagos"


# ========================= INDICATORS =========================
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
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def atr_value(highs, lows, closes, period=14):
    highs  = np.array(highs,  dtype=float)
    lows   = np.array(lows,   dtype=float)
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = float(np.mean(trs[:period]))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return float(atr)


# ========================= PATTERNS =========================
def body_ratio(candle):
    rng = abs(candle[2] - candle[3])
    if rng == 0:
        return 0.0
    return abs(candle[4] - candle[1]) / rng


def is_bullish_engulfing(prev, cur):
    return (prev[4] < prev[1] and cur[4] > cur[1]
            and cur[4] >= prev[1] and cur[1] <= prev[4])


def is_bearish_engulfing(prev, cur):
    return (prev[4] > prev[1] and cur[4] < cur[1]
            and cur[4] <= prev[1] and cur[1] >= prev[4])


def is_bullish_rejection(c):
    o, h, l, cl = c[1], c[2], c[3], c[4]
    rng   = max(1e-10, h - l)
    body  = abs(cl - o)
    lower = min(o, cl) - l
    return lower / rng >= 0.45 and body / rng <= 0.55 and cl >= o


def is_bearish_rejection(c):
    o, h, l, cl = c[1], c[2], c[3], c[4]
    rng   = max(1e-10, h - l)
    body  = abs(cl - o)
    upper = h - max(o, cl)
    return upper / rng >= 0.45 and body / rng <= 0.55 and cl <= o


# ========================= SESSION =========================
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
    be_level: float   # breakeven trigger price
    atr:     float
    reason:  str
    pattern: str
    rsi:     float


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
        if time.time() < self.pause_until:
            return False, "Paused"
        if time.time() < self.cooldown_until:
            left = int(self.cooldown_until - time.time())
            return False, f"Cooldown {left}s"
        if len(self.active_positions) >= MAX_OPEN_POSITIONS:
            return False, "Max positions open"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Daily trade limit reached"
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

    def build_signal(self, symbol, c15, c5, c1):
        """
        Returns (Signal | None, reason_str, debug_dict)
        """
        dbg = {}

        # ── 15M TREND ──────────────────────────────────────
        if len(c15) < EMA_TREND_SLOW + 10:
            return None, "15M warming up", dbg

        cl15 = [x[4] for x in c15]
        hi15 = [x[2] for x in c15]
        lo15 = [x[3] for x in c15]

        ema50_15  = ema_value(cl15, EMA_TREND_FAST)
        ema200_15 = ema_value(cl15, EMA_TREND_SLOW)
        atr15     = atr_value(hi15, lo15, cl15, ATR_PERIOD)
        price15   = cl15[-1]

        # 15M EMA50 slope
        ema50_15_series = ema_series(cl15, EMA_TREND_FAST)
        slope_idx = -(EMA_SLOPE_LOOKBACK + 1)
        ema50_15_slope = 0.0
        if len(ema50_15_series) > abs(slope_idx) and ema50_15_series[slope_idx]:
            ema50_15_slope = ema50_15 - ema50_15_series[slope_idx]

        if None in (ema50_15, ema200_15, atr15):
            return None, "15M indicators not ready", dbg

        trend_up   = (ema50_15 > ema200_15
                      and price15 > ema50_15
                      and ema50_15_slope > 0)
        trend_down = (ema50_15 < ema200_15
                      and price15 < ema50_15
                      and ema50_15_slope < 0)

        dbg["ema50_15"]       = round(ema50_15, 4)
        dbg["ema200_15"]      = round(ema200_15, 4)
        dbg["15m_slope"]      = round(ema50_15_slope, 6)
        dbg["trend"]          = "UP" if trend_up else "DOWN" if trend_down else "SIDE"

        if not trend_up and not trend_down:
            return None, f"15M sideways | EMA50:{ema50_15:.2f} EMA200:{ema200_15:.2f}", dbg

        # ── 5M PULLBACK ────────────────────────────────────
        if len(c5) < EMA_SETUP_SLOW + EMA_SLOPE_LOOKBACK + 5:
            return None, "5M warming up", dbg

        cl5 = [x[4] for x in c5]
        hi5 = [x[2] for x in c5]
        lo5 = [x[3] for x in c5]

        ema20_5_series = ema_series(cl5, EMA_PULLBACK)
        ema50_5_series = ema_series(cl5, EMA_SETUP_SLOW)
        rsi5  = rsi_value(cl5, RSI_PERIOD)
        atr5  = atr_value(hi5, lo5, cl5, ATR_PERIOD)

        if not ema20_5_series or not ema50_5_series or rsi5 is None or atr5 is None:
            return None, "5M indicators not ready", dbg

        ema20_5 = ema20_5_series[-1]
        ema50_5 = ema50_5_series[-1]

        # 5M EMA50 slope
        ema50_5_old = ema50_5_series[-(EMA_SLOPE_LOOKBACK + 1)] if len(ema50_5_series) > EMA_SLOPE_LOOKBACK + 1 else None
        ema50_5_slope = (ema50_5 - ema50_5_old) if ema50_5_old else 0.0

        spread_ok      = abs(ema20_5 - ema50_5) >= EMA_SPREAD_ATR_MULT * atr5
        rsi_ok         = RSI_MIN <= rsi5 <= RSI_MAX
        slope_5_rising = ema50_5_slope > 0
        slope_5_falling= ema50_5_slope < 0

        # Near-touch pullback — within 0.3x ATR counts
        pb = c5[-1]
        pb_touch_bull  = pb[3] <= ema20_5 + (PULLBACK_ATR_BUFFER * atr5)
        pb_touch_bear  = pb[2] >= ema20_5 - (PULLBACK_ATR_BUFFER * atr5)
        pb_above_ema50 = pb[4] > ema50_5
        pb_below_ema50 = pb[4] < ema50_5

        dbg["ema20_5"]       = round(ema20_5, 4)
        dbg["ema50_5"]       = round(ema50_5, 4)
        dbg["5m_slope"]      = round(ema50_5_slope, 6)
        dbg["rsi5"]          = round(rsi5, 1)
        dbg["rsi_ok"]        = rsi_ok
        dbg["spread_ok"]     = spread_ok
        dbg["pb_touch_bull"] = pb_touch_bull
        dbg["pb_touch_bear"] = pb_touch_bear
        dbg["atr5"]          = round(atr5, 4)

        # ── 1M ENTRY TRIGGER ───────────────────────────────
        if len(c1) < SPIKE_LOOKBACK + 3:
            return None, "1M warming up", dbg

        prev1 = c1[-2]
        cur1  = c1[-1]

        bull_engulf  = is_bullish_engulfing(prev1, cur1)
        bear_engulf  = is_bearish_engulfing(prev1, cur1)
        bull_reject  = is_bullish_rejection(cur1)
        bear_reject  = is_bearish_rejection(cur1)
        bull_pattern = bull_engulf or bull_reject
        bear_pattern = bear_engulf or bear_reject

        close_above_ema20 = cur1[4] > ema20_5
        close_below_ema20 = cur1[4] < ema20_5
        br = body_ratio(cur1)
        body_ok = br >= BODY_RATIO_MIN

        recent_bodies = [abs(x[4] - x[1]) for x in c1[-(SPIKE_LOOKBACK+2):-2]]
        avg_body      = sum(recent_bodies) / len(recent_bodies) if recent_bodies else 0
        last_body     = abs(cur1[4] - cur1[1])
        spike_ok      = avg_body == 0 or last_body <= SPIKE_MULT * avg_body

        pattern_str = ("BullEngulf" if bull_engulf else
                       "BullReject" if bull_reject else
                       "BearEngulf" if bear_engulf else
                       "BearReject" if bear_reject else "None")

        dbg["pattern"]          = pattern_str
        dbg["close_above_ema20"]= close_above_ema20
        dbg["close_below_ema20"]= close_below_ema20
        dbg["body_ratio"]       = round(br, 2)
        dbg["body_ok"]          = body_ok
        dbg["spike_ok"]         = spike_ok

        # ── LONG SIGNAL ────────────────────────────────────
        if (trend_up and spread_ok and slope_5_rising
                and pb_touch_bull and pb_above_ema50
                and rsi_ok and bull_pattern
                and close_above_ema20 and body_ok and spike_ok):

            entry    = float(cur1[4])
            stop     = float(min(pb[3], cur1[3]) - atr5 * 0.2)
            if stop >= entry:
                return None, "Invalid long SL", dbg
            risk     = entry - stop
            target   = entry + risk * RR_RATIO
            be_level = entry + risk * BREAKEVEN_RR
            return Signal("buy", symbol, entry, stop, target, be_level,
                          float(atr5),
                          f"15M UP + 5M pullback EMA20 + RSI {rsi5:.1f} + {pattern_str}",
                          pattern_str, float(rsi5)), "LONG ✅", dbg

        # ── SHORT SIGNAL ───────────────────────────────────
        if (trend_down and spread_ok and slope_5_falling
                and pb_touch_bear and pb_below_ema50
                and rsi_ok and bear_pattern
                and close_below_ema20 and body_ok and spike_ok):

            entry    = float(cur1[4])
            stop     = float(max(pb[2], cur1[2]) + atr5 * 0.2)
            if stop <= entry:
                return None, "Invalid short SL", dbg
            risk     = stop - entry
            target   = entry - risk * RR_RATIO
            be_level = entry - risk * BREAKEVEN_RR
            return Signal("sell", symbol, entry, stop, target, be_level,
                          float(atr5),
                          f"15M DOWN + 5M pullback EMA20 + RSI {rsi5:.1f} + {pattern_str}",
                          pattern_str, float(rsi5)), "SHORT ✅", dbg

        # ── NO SIGNAL — detailed reason ────────────────────
        if not spread_ok:
            reason = f"5M ranging — spread too tight"
        elif trend_up and not slope_5_rising:
            reason = f"5M slope flat/falling — momentum weak"
        elif trend_down and not slope_5_falling:
            reason = f"5M slope flat/rising — momentum weak"
        elif trend_up and not pb_touch_bull:
            reason = f"Waiting 5M pullback to EMA20 ({ema20_5:.4f})"
        elif trend_down and not pb_touch_bear:
            reason = f"Waiting 5M pullback to EMA20 ({ema20_5:.4f})"
        elif not rsi_ok:
            reason = f"RSI {rsi5:.1f} outside {RSI_MIN}–{RSI_MAX}"
        elif trend_up and not bull_pattern:
            reason = "Waiting 1M bullish pattern"
        elif trend_down and not bear_pattern:
            reason = "Waiting 1M bearish pattern"
        elif trend_up and not close_above_ema20:
            reason = f"1M close not above EMA20 ({ema20_5:.4f})"
        elif trend_down and not close_below_ema20:
            reason = f"1M close not below EMA20 ({ema20_5:.4f})"
        elif not body_ok:
            reason = f"Weak body {br:.2f} < {BODY_RATIO_MIN}"
        elif not spike_ok:
            reason = "Spike candle blocked"
        else:
            reason = f"No setup | RSI:{rsi5:.1f} slope:{ema50_5_slope:.4f}"

        return None, reason, dbg

    async def set_leverage(self, symbol):
        try:
            await self.exchange.set_leverage(LEVERAGE, symbol)
        except Exception as e:
            logger.warning(f"Leverage set failed {symbol}: {e}")

    async def get_size(self, symbol, entry, stop):
        await self.fetch_balance()
        risk_amount   = max(1.0, self.balance_usdt * RISK_PER_TRADE)
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
                "be_level":  signal.be_level,
                "be_moved":  False,
                "opened_at": time.time(),
            }
            self.trades_today    += 1
            self.cooldown_until   = time.time() + COOLDOWN_SEC

            sym_clean = signal.symbol.replace("/USDT:USDT", "")
            await self.tg(
                f"🚀 {sym_clean} {signal.side.upper()}\n"
                f"Entry:  {avg_price:.4f}\n"
                f"Stop:   {signal.stop:.4f}\n"
                f"Target: {signal.target:.4f}\n"
                f"BE at:  {signal.be_level:.4f}\n"
                f"Size:   {size} contracts\n"
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
                    # Try to get closed PnL
                    pnl = 0.0
                    try:
                        history = await self.exchange.fetch_closed_orders(sym, limit=5)
                        if history:
                            pnl = sum(float(o.get("profit", 0) or 0) for o in history[-2:])
                    except Exception:
                        pass

                    if pnl < 0:
                        self.consec_losses  += 1
                        self.losses_today   += 1
                    else:
                        self.consec_losses   = 0
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
                        f"Streak: {self.consec_losses}"
                    )

                # Breakeven check
                elif sym in self.active_positions:
                    pos = self.active_positions[sym]
                    if not pos.get("be_moved"):
                        try:
                            ticker = await self.exchange.fetch_ticker(sym)
                            price  = float(ticker.get("last", 0))
                            if pos["side"] == "buy"  and price >= pos["be_level"]:
                                pos["be_moved"] = True
                                logger.info(f"BE: {sym} SL moved to entry {pos['entry']}")
                                await self.tg(f"🔒 {sym.replace('/USDT:USDT','')} SL moved to breakeven ({pos['entry']:.4f})")
                            elif pos["side"] == "sell" and price <= pos["be_level"]:
                                pos["be_moved"] = True
                                logger.info(f"BE: {sym} SL moved to entry {pos['entry']}")
                                await self.tg(f"🔒 {sym.replace('/USDT:USDT','')} SL moved to breakeven ({pos['entry']:.4f})")
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"Reconcile error: {e}")

    async def scan_symbol(self, symbol, stagger=0):
        await asyncio.sleep(stagger)
        # Mark as initializing immediately so status shows something
        self.market_debug[symbol] = {"time": time.time(), "why": "Initializing...", "signal": None}
        while self.is_scanning:
            try:
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                self._reset_daily()

                if not self.exchange:
                    self.market_debug[symbol] = {"time": time.time(), "why": "Not connected — press CONNECT", "signal": None}
                    await asyncio.sleep(5)
                    continue

                if symbol in self.active_positions:
                    continue

                can, gate = self.can_trade()
                if not can:
                    self.market_debug[symbol] = {
                        "time": time.time(), "why": gate, "signal": None}
                    continue

                self.market_debug[symbol] = {"time": time.time(), "why": "Fetching candles...", "signal": None}

                # Fetch candles sequentially to avoid rate limits
                c15 = await self.fetch_ohlcv(symbol, TF_TREND,   CANDLES_15M)
                await asyncio.sleep(0.3)
                c5  = await self.fetch_ohlcv(symbol, TF_SETUP,   CANDLES_5M)
                await asyncio.sleep(0.3)
                c1  = await self.fetch_ohlcv(symbol, TF_TRIGGER, CANDLES_1M)

                if not c15 or not c5 or not c1:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "why": f"Candle fetch failed — 15M:{len(c15)} 5M:{len(c5)} 1M:{len(c1)}. Check API connection.",
                        "signal": None}
                    continue

                signal, reason, dbg = self.build_signal(symbol, c15, c5, c1)
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
        return f"📍 {sym.replace('/USDT:USDT','')} ⏳ No data yet"
    age     = int(time.time() - d.get("time", time.time()))
    signal  = d.get("signal") or "—"
    why     = d.get("why", "—")
    trend   = d.get("trend", "—")
    e50_15  = d.get("ema50_15", "—")
    e200_15 = d.get("ema200_15", "—")
    slp15   = d.get("15m_slope", 0)
    e20_5   = d.get("ema20_5", "—")
    e50_5   = d.get("ema50_5", "—")
    slp5    = d.get("5m_slope", 0)
    rsi     = d.get("rsi5", "—")
    rsi_ok  = "✅" if d.get("rsi_ok") else "❌"
    spr_ok  = "✅" if d.get("spread_ok") else "❌"
    pb_b    = "✅" if d.get("pb_touch_bull") else "⏳"
    pb_s    = "✅" if d.get("pb_touch_bear") else "⏳"
    pat     = d.get("pattern", "—")
    body_ok = "✅" if d.get("body_ok") else "❌"
    spk_ok  = "✅" if d.get("spike_ok", True) else "❌ Spike"
    sym_c   = sym.replace("/USDT:USDT", "")
    return (
        f"📍 {sym_c} ({age}s)\n"
        f"15M: {trend} | EMA50:{e50_15} EMA200:{e200_15} Slope:{slp15:.5f}\n"
        f"5M:  EMA20:{e20_5} EMA50:{e50_5} Slope:{slp5:.5f}\n"
        f"5M:  RSI:{rsi} {rsi_ok} | Spread:{spr_ok} | PB Bull:{pb_b} Bear:{pb_s}\n"
        f"1M:  {pat} | Body:{body_ok} | Spike:{spk_ok}\n"
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
        import os
        os.environ["BITGET_TESTNET"] = "true"
        await safe_edit(q, "🧪 Switched to TESTNET — reconnect to apply", keyboard())

    elif q.data == "LIVE":
        import os
        os.environ["BITGET_TESTNET"] = "false"
        await safe_edit(q, "💰 Switched to LIVE — reconnect to apply", keyboard())

    elif q.data == "START":
        if not bot.exchange:
            await safe_edit(q, "❌ Connect first", keyboard())
            return
        bot.is_scanning = True
        for i, sym in enumerate(SYMBOLS):
            asyncio.create_task(bot.scan_symbol(sym, stagger=i*5))
        await safe_edit(q,
            f"🔍 Scanner active\n"
            f"Pairs: BTC ETH SOL\n"
            f"Strategy: 15M/5M/1M Pullback\n"
            f"RR: 1:{RR_RATIO} | Leverage: {LEVERAGE}x\n"
            f"Risk/trade: {RISK_PER_TRADE:.1%}",
            keyboard())

    elif q.data == "STOP":
        bot.is_scanning = False
        await safe_edit(q, "⏹️ Scanner stopped", keyboard())

    elif q.data == "PAUSE":
        bot.pause_until = time.time() + 86400
        await safe_edit(q, "⏸ Paused 24h", keyboard())

    elif q.data == "RESUME":
        bot.pause_until = 0
        bot.consec_losses = 0
        await safe_edit(q, "▶️ Resumed", keyboard())

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
            open_pos += (f"\n🔵 {sym_c} {pos['side'].upper()} @ {pos['entry']:.4f} "
                         f"| SL:{pos['stop']:.4f} TP:{pos['target']:.4f} "
                         f"| {'🔒 BE' if pos.get('be_moved') else '⏳'} ({age}s)")

        header = (
            f"🕒 {now_wat()}\n"
            f"🤖 {'ACTIVE' if bot.is_scanning else 'OFFLINE'} | "
            f"{'🧪 TESTNET' if USE_TESTNET else '💰 LIVE'}\n"
            f"💰 Balance: {bot.balance_usdt:.2f} USDT\n"
            f"📈 PnL Today: {bot.profit_today:+.4f} USDT\n"
            f"🎯 RR 1:{RR_RATIO} | Risk {RISK_PER_TRADE:.1%} | {LEVERAGE}x\n"
            f"📉 Streak: {bot.consec_losses}/{MAX_CONSEC_LOSSES} | "
            f"Trades: {bot.trades_today}/{MAX_TRADES_PER_DAY}\n"
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
        "Strategy: 15M trend + 5M pullback + 1M entry\n"
        "Pairs: BTC ETH SOL perpetuals\n"
        f"RR: 1:{RR_RATIO} | Leverage: {LEVERAGE}x\n"
        f"Risk/trade: {RISK_PER_TRADE:.1%}\n"
        "1. Press CONNECT\n"
        "2. Press START\n"
        "3. Press STATUS to monitor",
        reply_markup=keyboard()
    )


# ========================= MAIN =========================
if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Missing TELEGRAM_TOKEN env variable")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bot.run())
    app.run_polling(close_loop=False)
