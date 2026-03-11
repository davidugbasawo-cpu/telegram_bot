"""
Deriv Forex Binary Options Bot
Strategy: 3-Timeframe Pullback System
  15M → trend direction  (EMA50 > EMA200 = uptrend)
  5M  → pullback quality (EMA20 touch + EMA50 slope + RSI + spread)
  1M  → entry trigger    (engulfing/rejection + close vs EMA20 + body + spike)
Markets: EURUSD, GBPUSD, USDJPY, AUDUSD, GBPJPY
Expiry: 2 minutes
"""

import asyncio, time, logging, json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ===== CREDENTIALS =====
APP_ID       = 1089
DEMO_TOKEN   = "tIrfitLjqeBxCOM"
REAL_TOKEN   = "ZkOFWOlPtwnjqTS"
TELEGRAM_TOKEN = "8697638086:AAG00D0RXUAqXFTjy8-4XO4Bka2kBamo-VA"

# ===== MARKETS =====
MARKETS = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxGBPJPY"]

# ===== LIMITS =====
MAX_TRADES_PER_DAY      = 30
MAX_TRADES_PER_MARKET   = 6
MAX_LOSSES_PER_MARKET   = 3
CONSEC_LOSS_PAUSE_SEC   = 1800   # 30 min after 2 consecutive losses
MAX_CONSEC_LOSSES       = 5      # global stop
COOLDOWN_SEC            = 180    # 3 min after every trade

# ===== STRATEGY =====
TF_15M_SEC          = 900        # 15M candles — trend direction
TF_5M_SEC           = 300        # 5M candles  — pullback quality
TF_M1_SEC           = 60         # 1M candles  — entry trigger
CANDLES_15M         = 250        # enough for EMA200
CANDLES_5M          = 150        # enough for EMA50 slope + RSI + ATR
CANDLES_M1          = 60         # enough for pattern + body detection
EXPIRY_MIN          = 2          # 2 min expiry

# 15M indicators
EMA_TREND_FAST      = 50         # EMA50  — fast trend
EMA_TREND_SLOW      = 200        # EMA200 — slow trend
EMA_15M_ATR_MULT    = 0.5        # min distance price vs EMA50 on 15M

# 5M indicators
EMA_PULLBACK        = 20         # EMA20  — pullback level
EMA_5M_SLOW         = 50         # EMA50  — 5M trend confirmation
EMA_SLOPE_LOOKBACK  = 10         # candles back for EMA50 slope
RSI_PERIOD          = 14         # RSI period
RSI_MIN             = 40         # RSI lower bound (both directions)
RSI_MAX             = 60         # RSI upper bound (both directions)
ATR_PERIOD          = 14         # ATR for spread filter
EMA_SPREAD_ATR_MULT = 0.2        # min EMA20/EMA50 spread vs ATR

# 1M indicators
BODY_RATIO_MIN      = 0.32       # minimum body ratio — no dojis
SPIKE_MULT          = 1.5        # body > SPIKE_MULT * avg = spike block
SPIKE_LOOKBACK      = 20         # candles for average body calculation

# Weekend block only — trading all sessions to collect data
FOREX_WEEKEND_BLOCK = False  # trading all days including weekends

# ===== MARTINGALE =====
MARTINGALE_MULT      = 2
MARTINGALE_MAX_STEPS = 5
PAYOUT_TARGET        = 1.00
MIN_PAYOUT           = 0.35
MAX_STAKE_ALLOWED    = 10.00

# ===== RISK =====
DAILY_PROFIT_TARGET = 10.0
DAILY_LOSS_LIMIT    = -20.0
EQUITY_STOP_PCT     = 0.60
PROFIT_LOCK_TRIGGER = 5.0
PROFIT_LOCK_FLOOR   = 2.0

# ===== MISC =====
STATS_DAYS               = 30
STATUS_REFRESH_COOLDOWN_SEC = 10
TRADE_LOG_FILE           = "structure_trades.json"
SCAN_INTERVAL_SEC        = 20

# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema

def calculate_ema_series(closes, period):
    """Returns full EMA array"""
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = [float(np.mean(closes[:period]))]
    for c in closes[period:]:
        ema.append(c * k + ema[-1] * (1 - k))
    # pad front with None
    result = [None] * (period - 1) + ema
    return result

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1 + rs)), 2)

def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1: return None
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])))
    atrs = [float(np.mean(tr[:period]))]
    for t in tr[period:]:
        atrs.append((atrs[-1] * (period - 1) + t) / period)
    return atrs



def calculate_bollinger_bands(closes, period=20, std_dev=2.0):
    """Returns (upper, middle, lower) for last candle"""
    closes = [float(c) for c in closes]
    if len(closes) < period: return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = variance ** 0.5
    return mid + std_dev * std, mid, mid - std_dev * std

def calculate_stochastic(highs, lows, closes, k_period=5, d_period=3):
    """
    Stochastic Oscillator
    Returns (%K_now, %K_prev, %D_now) for last candle
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = 3-period SMA of %K
    """
    if len(closes) < k_period + d_period: return None, None, None
    k_values = []
    for i in range(len(closes)):
        if i < k_period - 1:
            k_values.append(None)
            continue
        window_highs = highs[i - k_period + 1: i + 1]
        window_lows  = lows[i  - k_period + 1: i + 1]
        highest = max(window_highs)
        lowest  = min(window_lows)
        if highest == lowest:
            k_values.append(50.0)
        else:
            k = (closes[i] - lowest) / (highest - lowest) * 100
            k_values.append(round(k, 2))
    # %D = SMA of last d_period %K values
    valid_k = [v for v in k_values if v is not None]
    if len(valid_k) < d_period + 1: return None, None, None
    k_now  = valid_k[-1]
    k_prev = valid_k[-2]
    d_now  = sum(valid_k[-d_period:]) / d_period
    return k_now, k_prev, d_now

def calculate_ema100(closes, period=100):
    """EMA100 for trend filter"""
    import numpy as np
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema

def find_swing_highs(highs, lookback=5):
    """Returns list of (index, value) for swing highs"""
    swings = []
    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
            swings.append((i, highs[i]))
    return swings

def find_swing_lows(lows, lookback=5):
    """Returns list of (index, value) for swing lows"""
    swings = []
    for i in range(lookback, len(lows) - lookback):
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
            swings.append((i, lows[i]))
    return swings

def get_structure_levels(highs, lows, lookback=5):
    """
    Returns last significant swing high and swing low
    These are the structure levels to watch for breaks
    """
    swing_highs = find_swing_highs(list(highs), lookback)
    swing_lows = find_swing_lows(list(lows), lookback)
    last_high = swing_highs[-1][1] if swing_highs else None
    last_low = swing_lows[-1][1] if swing_lows else None
    return last_high, last_low

def body_ratio(open_, close, high, low):
    candle_range = abs(high - low)
    if candle_range == 0: return 0
    return abs(close - open_) / candle_range

def is_engulfing_bullish(prev, cur):
    """Current green candle fully engulfs previous red candle"""
    if prev["close"] >= prev["open"]: return False   # prev must be bearish
    if cur["close"] <= cur["open"]:   return False   # cur must be bullish
    return cur["close"] >= prev["open"] and cur["open"] <= prev["close"]

def is_engulfing_bearish(prev, cur):
    """Current red candle fully engulfs previous green candle"""
    if prev["close"] <= prev["open"]: return False   # prev must be bullish
    if cur["close"] >= cur["open"]:   return False   # cur must be bearish
    return cur["close"] <= prev["open"] and cur["open"] >= prev["close"]

def is_rejection_bullish(candle):
    """Hammer: long lower wick, small body, closes green"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = max(1e-10, h - l)
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return (lower_wick / rng) >= 0.45 and (body / rng) <= 0.55 and c >= o

def is_rejection_bearish(candle):
    """Shooting star: long upper wick, small body, closes red"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = max(1e-10, h - l)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (upper_wick / rng) >= 0.45 and (body / rng) <= 0.55 and c <= o

def calc_ema_value(closes, period):
    """Returns single EMA value for last candle"""
    closes = [float(c) for c in closes]
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema

def calc_ema_series_full(closes, period):
    """Returns full EMA series as list"""
    closes = [float(c) for c in closes]
    if len(closes) < period: return []
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    result = [None] * period + [ema]
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
        result.append(ema)
    return result

def calc_rsi(closes, period=14):
    """Returns single RSI value"""
    closes = [float(c) for c in closes]
    if len(closes) < period + 2: return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_g  = sum(gains[:period]) / period
    avg_l  = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + avg_g / avg_l))

def calc_atr(highs, lows, closes, period=14):
    """Returns single ATR value"""
    if len(closes) < period + 2: return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        trs.append(tr)
    if len(trs) < period: return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

def is_forex_market_open():
    now_utc = datetime.now(ZoneInfo("UTC"))
    weekday = now_utc.weekday()
    hour_utc = now_utc.hour
    if FOREX_WEEKEND_BLOCK:
        if weekday == 5: return False, "Weekend — market closed (Saturday)"
        if weekday == 6: return False, "Weekend — market closed (Sunday)"
        if weekday == 4 and hour_utc >= 21: return False, "Weekend — market closing (Friday 9pm+ UTC)"
    return True, "OK"

def fmt_time_hhmmss(epoch):
    try: return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except: return "—"

def money2(x):
    return round(float(x), 2)

def session_bucket(ts):
    h = datetime.fromtimestamp(ts, ZoneInfo("UTC")).hour
    if 7 <= h < 12: return "London"
    elif 12 <= h < 17: return "NY Overlap"
    elif 17 <= h < 21: return "Late NY"
    else: return "Asian"

def build_candles_from_deriv(raw):
    candles = []
    for c in raw:
        candles.append({
            "epoch": int(c.get("epoch", 0)),
            "open":  float(c.get("open", 0)),
            "high":  float(c.get("high", 0)),
            "low":   float(c.get("low", 0)),
            "close": float(c.get("close", 0)),
        })
    candles.sort(key=lambda x: x["epoch"])
    return candles

# ============================================================
# BOT LOGIC
# ============================================================

class StructureBreakBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.account_type = "DEMO"
        self.balance = "0.00 USD"
        self.starting_balance = 0.0
        self.is_scanning = False

        # martingale
        self.martingale_step = 0
        self.martingale_halt = False

        # daily tracking
        self.trades_today = 0
        self.total_profit_today = 0.0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.max_loss_streak_today = 0
        self.last_reset_date = None

        # per market
        self.market_trades_today = {m: 0 for m in MARKETS}
        self.market_losses_today = {m: 0 for m in MARKETS}
        self.market_blocked = {m: False for m in MARKETS}
        self.market_pause_until = {m: 0 for m in MARKETS}

        # active trade
        self.active_trade_info = None
        self.active_market = None
        self.trade_start_time = 0

        # cooldown
        self.cooldown_until = 0
        self.pause_until = 0
        self.status_cooldown_until = 0

        # profit lock
        self.profit_lock_active = False

        # debug
        self.market_debug = {}
        self.last_processed_m5_t0 = {}

        # trade history
        self.trade_history = []
        self._load_trade_history()

    def _load_trade_history(self):
        try:
            if os.path.exists(TRADE_LOG_FILE):
                with open(TRADE_LOG_FILE, "r") as f:
                    self.trade_history = json.load(f)
        except: self.trade_history = []

    def _save_trade(self, record):
        self.trade_history.append(record)
        try:
            with open(TRADE_LOG_FILE, "w") as f:
                json.dump(self.trade_history[-500:], f)
        except: pass

    def _reset_daily_if_needed(self):
        today = datetime.now(ZoneInfo("Africa/Lagos")).date()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.trades_today = 0
            self.total_profit_today = 0.0
            self.total_losses_today = 0
            self.consecutive_losses = 0
            self.max_loss_streak_today = 0
            self.martingale_step = 0
            self.martingale_halt = False
            self.profit_lock_active = False
            self.market_trades_today = {m: 0 for m in MARKETS}
            self.market_losses_today = {m: 0 for m in MARKETS}
            self.market_blocked = {m: False for m in MARKETS}
            self.market_pause_until = {m: 0 for m in MARKETS}

    def _get_current_balance_float(self):
        try: return float(str(self.balance).replace(" USD", "").replace(",", ""))
        except: return self.starting_balance

    def _equity_ok(self):
        if self.starting_balance <= 0: return True, "OK", 1.0
        ratio = self._get_current_balance_float() / self.starting_balance
        if ratio <= EQUITY_STOP_PCT: return False, f"Equity stop — {ratio:.0%} of starting", 0.0
        if ratio >= 1.0: return True, "OK", 1.0
        if ratio >= 0.90: return True, "OK", 0.75
        if ratio >= 0.80: return True, "OK", 0.50
        return True, "OK", 0.25

    def _is_market_available(self, symbol):
        if self.market_blocked.get(symbol): return False, "Blocked for day"
        if time.time() < self.market_pause_until.get(symbol, 0):
            rem = int(self.market_pause_until[symbol] - time.time())
            return False, f"Paused {rem}s"
        if self.market_trades_today.get(symbol, 0) >= MAX_TRADES_PER_MARKET:
            return False, f"Max {MAX_TRADES_PER_MARKET} trades today"
        return True, "OK"

    def can_auto_trade(self):
        self._reset_daily_if_needed()
        if self.active_trade_info: return False, "Trade active"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if time.time() < self.pause_until: return False, "Paused"
        eq_ok, eq_msg, _ = self._equity_ok()
        if not eq_ok: return False, eq_msg
        if self.martingale_halt: return False, f"Martingale {MARTINGALE_MAX_STEPS} steps done"
        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            return False, f"Daily target +${DAILY_PROFIT_TARGET:.2f} reached"
        if self.total_profit_today <= DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit ${DAILY_LOSS_LIMIT:.2f} hit"
        if self.profit_lock_active and self.total_profit_today <= PROFIT_LOCK_FLOOR:
            return False, f"Profit lock protecting +${PROFIT_LOCK_FLOOR:.2f}"
        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, f"Stopped: {MAX_CONSEC_LOSSES} consecutive losses"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max {MAX_TRADES_PER_DAY} trades today"
        return True, "OK"

    def stats_30d(self):
        cutoff = time.time() - (STATS_DAYS * 24 * 3600)
        by_mkt = {}; by_sess = {}
        for t in self.trade_history:
            if t.get("time", 0) < cutoff: continue
            m = t.get("market", "?"); s = t.get("session", "?")
            w = 1 if t.get("result") == "WIN" else 0
            for d, k in [(by_mkt, m), (by_sess, s)]:
                if k not in d: d[k] = {"trades": 0, "wins": 0}
                d[k]["trades"] += 1; d[k]["wins"] += w
        def wr(v): return round(100 * v["wins"] / v["trades"], 1) if v["trades"] > 0 else 0.0
        return by_mkt, by_sess, wr

    async def connect(self):
        for attempt in range(3):
            try:
                token = DEMO_TOKEN if self.account_type == "DEMO" else REAL_TOKEN
                # Always destroy old connection first
                if self.api:
                    try: await self.api.disconnect()
                    except: pass
                    self.api = None
                await asyncio.sleep(1)
                self.api = DerivAPI(app_id=APP_ID)
                await asyncio.wait_for(self.api.authorize(token), timeout=15.0)
                await self.fetch_balance()
                try:
                    bal_val = float(str(self.balance).split()[0])
                    if self.starting_balance == 0.0: self.starting_balance = bal_val
                except: pass
                logger.info(f"Connected — {self.account_type} | {self.balance}")
                return True
            except Exception as e:
                logger.error(f"Connect error attempt {attempt+1}: {e}")
                self.api = None
                await asyncio.sleep(3 * (attempt + 1))
        return False

    async def ping(self):
        """Check if connection is alive"""
        if not self.api: return False
        try:
            await asyncio.wait_for(self.api.ping({"ping": 1}), timeout=5.0)
            return True
        except:
            return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = "{:.2f} {}".format(float(bal["balance"]["balance"]), bal["balance"]["currency"])
        except: pass

    # start_scanning() removed — scanning controlled by Telegram buttons

    async def safe_deriv_call(self, method, params, retries=3):
        for attempt in range(retries):
            try:
                fn = getattr(self.api, method)
                return await asyncio.wait_for(fn(params), timeout=15.0)
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def safe_send_tg(self, text, parse_mode=None):
        if not self.app: return
        try:
            for uid in getattr(self, "_known_chat_ids", []):
                try:
                    kwargs = {"chat_id": uid, "text": text[:4000]}
                    if parse_mode: kwargs["parse_mode"] = parse_mode
                    await self.app.bot.send_message(**kwargs)
                    return
                except: pass
        except Exception as e:
            logger.warning(f"TG send failed: {e}")

    async def fetch_candles_with_timeout(self, symbol, tf_sec, count):
        for attempt in range(4):
            try:
                # Ping first — if connection is stale, reconnect before fetching
                alive = await self.ping()
                if not alive:
                    logger.warning(f"Connection dead (attempt {attempt+1}) — reconnecting")
                    connected = await self.connect()
                    if not connected:
                        await asyncio.sleep(3)
                        continue
                    await asyncio.sleep(1)
                res = await asyncio.wait_for(
                    self.api.ticks_history({
                        "ticks_history": symbol, "style": "candles",
                        "granularity": tf_sec, "count": count, "end": "latest"
                    }), timeout=20.0)
                candles = build_candles_from_deriv(res.get("candles", []))
                if candles:
                    return candles
                logger.warning(f"Empty candles {symbol} attempt {attempt+1}")
                await asyncio.sleep(2)
            except asyncio.TimeoutError:
                logger.warning(f"Candle fetch timeout {symbol} attempt {attempt+1} — reconnecting")
                self.api = None
                await asyncio.sleep(3)
            except Exception as e:
                logger.warning(f"Candles failed {symbol} attempt {attempt+1}: {e}")
                self.api = None
                await asyncio.sleep(3 * (attempt + 1))
        logger.error(f"fetch_candles failed after 4 attempts: {symbol}")
        return []

    async def execute_trade(self, side, symbol, **kwargs):
        _, _, stake_mult = self._equity_ok()
        base_payout = money2(max(float(MIN_PAYOUT),
            money2(max(0.01, float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step))))))
        try:
            prop = await self.safe_deriv_call("proposal", {
                "proposal": 1, "amount": base_payout, "basis": "payout",
                "contract_type": side, "currency": "USD",
                "duration": int(EXPIRY_MIN), "duration_unit": "m",
                "symbol": symbol
            }, retries=6)
            if "error" in prop:
                await self.safe_send_tg(f"⚠️ Proposal error {symbol}: {prop['error'].get('message','?')}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            ask_price = float(prop.get("proposal", {}).get("ask_price", 0))
            proposal_id = prop.get("proposal", {}).get("id")
            if ask_price > float(MAX_STAKE_ALLOWED):
                await self.safe_send_tg(f"⛔️ Skipped: stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": ask_price}, retries=1)
            if "error" in buy:
                await self.safe_send_tg(f"⚠️ Buy error {symbol}: {buy['error'].get('message','?')}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            contract_id = buy.get("buy", {}).get("contract_id")
            self.active_trade_info = contract_id
            self.active_market = symbol
            self.trade_start_time = time.time()
            pair = symbol.replace("frx", "")
            icon = "📈" if side == "CALL" else "📉"
            await self.safe_send_tg(
                f"{icon} STRUCTURE BREAK — {side}\n"
                f"Pair: {pair}\n"
                f"Stake: ${ask_price:.2f} → Payout: ${base_payout:.2f}\n"
                f"Step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"Expiry: {EXPIRY_MIN}m\n"
                f"Session: {session_bucket(time.time())}\n"
                f"Pattern: {kwargs.get('pattern','?')}\n"
                f"5M EMA20: {kwargs.get('ema20','?')}\n"
                f"RSI: {kwargs.get('rsi','?')} | Trend: {kwargs.get('trend','?')}"
            )
            # wait for result
            await asyncio.sleep(EXPIRY_MIN * 60 + 10)
            await self._check_result(contract_id, symbol, side, ask_price, base_payout, kwargs)
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    async def _check_result(self, contract_id, symbol, side, stake, payout, kwargs):
        try:
            res = await self.safe_deriv_call("profit_table", {
                "profit_table": 1, "description": 1, "limit": 5
            }, retries=3)
            contracts = res.get("profit_table", {}).get("transactions", [])
            profit = None
            for c in contracts:
                if c.get("contract_id") == contract_id:
                    profit = float(c.get("sell_price", 0)) - float(c.get("buy_price", 0))
                    break
            if profit is None:
                logger.warning(f"Contract {contract_id} not found in profit_table — marking UNKNOWN")
                await self.safe_send_tg(f"⚠️ Could not verify result for contract {contract_id}. Check manually.")
                return
            won = profit > 0
            self.total_profit_today = round(self.total_profit_today + profit, 2)
            self.trades_today += 1
            self.market_trades_today[symbol] = self.market_trades_today.get(symbol, 0) + 1

            if won:
                self.consecutive_losses = 0
                self.martingale_step = 0
                result_str = "WIN"
                icon = "✅"
                if self.total_profit_today >= PROFIT_LOCK_TRIGGER:
                    self.profit_lock_active = True
            else:
                self.consecutive_losses += 1
                self.total_losses_today += 1
                self.max_loss_streak_today = max(self.max_loss_streak_today, self.consecutive_losses)
                self.market_losses_today[symbol] = self.market_losses_today.get(symbol, 0) + 1
                result_str = "LOSS"
                icon = "❌"
                if self.martingale_step < MARTINGALE_MAX_STEPS:
                    self.martingale_step += 1
                else:
                    self.martingale_halt = True
                if self.market_losses_today.get(symbol, 0) >= MAX_LOSSES_PER_MARKET:
                    self.market_blocked[symbol] = True
                if self.consecutive_losses >= 2:
                    self.market_pause_until[symbol] = time.time() + CONSEC_LOSS_PAUSE_SEC

            pair = symbol.replace("frx", "")
            self._save_trade({
                "time": time.time(), "market": pair, "side": side,
                "stake": stake, "payout": payout, "profit": profit,
                "result": result_str, "session": session_bucket(time.time()),
                "rsi": kwargs.get("rsi"), "body": kwargs.get("body"),
            })

            await self.safe_send_tg(
                f"{icon} {result_str} — {pair} {side}\n"
                f"P/L: {profit:+.2f} | Today: {self.total_profit_today:+.2f}\n"
                f"Streak: {self.consecutive_losses} | Step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"Balance: {self.balance}"
            )
        except Exception as e:
            logger.error(f"Result check error: {e}")
        finally:
            self.active_trade_info = None
            self.active_market = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    async def scan_market(self, symbol):
        while self.is_scanning:
            try:
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                self._reset_daily_if_needed()
                if self.active_trade_info:
                    continue

                # Weekend/session gate
                forex_open, forex_reason = is_forex_market_open()
                if not forex_open:
                    self.market_debug[symbol] = {"time": time.time(), "why": [forex_reason]}
                    continue

                ok_gate, gate = self.can_auto_trade()
                mkt_ok, mkt_msg = self._is_market_available(symbol)

                # ── FETCH ALL 3 TIMEFRAMES ───────────────────────────
                tf15_task = asyncio.create_task(
                    self.fetch_candles_with_timeout(symbol, TF_15M_SEC, CANDLES_15M))
                tf5_task  = asyncio.create_task(
                    self.fetch_candles_with_timeout(symbol, TF_5M_SEC,  CANDLES_5M))
                tf1_task  = asyncio.create_task(
                    self.fetch_candles_with_timeout(symbol, TF_M1_SEC,  CANDLES_M1))
                candles_15m, candles_5m, candles_1m = await asyncio.gather(
                    tf15_task, tf5_task, tf1_task)

                # Need enough candles for all indicators
                if len(candles_15m) < EMA_TREND_SLOW + 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": [f"Warming up 15M:{len(candles_15m)} 5M:{len(candles_5m)} 1M:{len(candles_1m)}"]}
                    continue

                # Stale data check — if latest 1M candle is > 3 min old, connection is frozen
                if candles_1m:
                    latest_epoch = candles_1m[-1].get("epoch", 0)
                    stale = latest_epoch > 0 and (time.time() - latest_epoch) > 180
                    last_recon = self._last_stale_reconnect.get(symbol, 0)
                    if stale and (time.time() - last_recon) > 60:
                        age = int(time.time() - latest_epoch)
                        logger.warning(f"Stale {symbol} — {age}s old, forcing full reconnect")
                        self._last_stale_reconnect[symbol] = time.time()
                        self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                            "why": [f"Stale data ({age}s) — reconnecting..."]}
                        try:
                            if self.api: await self.api.disconnect()
                        except: pass
                        self.api = None
                        await asyncio.sleep(2)
                        await self.connect()
                        await asyncio.sleep(3)
                        continue
                    elif stale:
                        # Already reconnected recently, just skip this cycle
                        self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                            "why": [f"Stale data — waiting for fresh candles"]}
                        await asyncio.sleep(5)
                        continue

                # Confirmed closed candles only (drop last open candle)
                c15 = candles_15m[:-1]
                c5  = candles_5m[:-1]
                c1  = candles_1m[:-1]

                if len(c15) < EMA_TREND_SLOW + 5: continue
                if len(c5)  < EMA_5M_SLOW + EMA_SLOPE_LOOKBACK + 5: continue
                if len(c1)  < SPIKE_LOOKBACK + 3: continue

                # ── 15M INDICATORS ────────────────────────────────────
                cl15 = [c["close"] for c in c15]
                hi15 = [c["high"]  for c in c15]
                lo15 = [c["low"]   for c in c15]

                ema50_15  = calc_ema_value(cl15, EMA_TREND_FAST)
                ema200_15 = calc_ema_value(cl15, EMA_TREND_SLOW)
                atr15     = calc_atr(hi15, lo15, cl15, ATR_PERIOD)
                price_15  = cl15[-1]

                if any(v is None for v in [ema50_15, ema200_15, atr15]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": ["15M indicators warming up"]}
                    continue

                # 15M trend
                trend_up   = (ema50_15 > ema200_15
                              and price_15 > ema50_15
                              and price_15 > ema200_15
                              and (price_15 - ema50_15) >= EMA_15M_ATR_MULT * atr15)
                trend_down = (ema50_15 < ema200_15
                              and price_15 < ema50_15
                              and price_15 < ema200_15
                              and (ema50_15 - price_15) >= EMA_15M_ATR_MULT * atr15)

                if not trend_up and not trend_down:
                    trend_label = "SIDEWAYS" if abs(ema50_15 - ema200_15) < atr15 * 0.5 else "WEAK"
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": [f"15M: {trend_label} — EMA50:{ema50_15:.5f} EMA200:{ema200_15:.5f}"]}
                    continue

                # ── 5M INDICATORS ─────────────────────────────────────
                cl5 = [c["close"] for c in c5]
                hi5 = [c["high"]  for c in c5]
                lo5 = [c["low"]   for c in c5]

                ema20_series = calc_ema_series_full(cl5, EMA_PULLBACK)
                ema50_series = calc_ema_series_full(cl5, EMA_5M_SLOW)
                rsi5         = calc_rsi(cl5, RSI_PERIOD)
                atr5         = calc_atr(hi5, lo5, cl5, ATR_PERIOD)

                if (not ema20_series or not ema50_series
                        or rsi5 is None or atr5 is None):
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": ["5M indicators warming up"]}
                    continue

                ema20_5  = ema20_series[-1]
                ema50_5  = ema50_series[-1]

                # EMA50 slope — compare now vs N candles ago
                slope_idx = -(EMA_SLOPE_LOOKBACK + 1)
                if abs(slope_idx) > len(ema50_series):
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": ["5M slope not ready"]}
                    continue
                ema50_slope   = ema50_5 - ema50_series[slope_idx]
                slope_rising  = ema50_slope >  0
                slope_falling = ema50_slope <  0

                # EMA spread filter — avoid flat ranging conditions
                ema_spread = abs(ema20_5 - ema50_5)
                spread_ok  = ema_spread >= EMA_SPREAD_ATR_MULT * atr5

                # RSI filter
                rsi_ok = RSI_MIN <= rsi5 <= RSI_MAX

                # 5M pullback touch — last confirmed 5M candle
                pb = c5[-1]
                pb_touched_ema20_bull = pb["low"]  <= ema20_series[-1]   # low touched EMA20
                pb_touched_ema20_bear = pb["high"] >= ema20_series[-1]   # high touched EMA20

                # 5M close must stay on trend side of EMA50 (no deep break)
                pb_close_above_ema50 = pb["close"] > ema50_5
                pb_close_below_ema50 = pb["close"] < ema50_5

                # ── 1M ENTRY TRIGGER ──────────────────────────────────
                if len(c1) < 3:
                    continue

                cl1 = [c["close"] for c in c1]
                hi1 = [c["high"]  for c in c1]
                lo1 = [c["low"]   for c in c1]

                prev1 = c1[-2]   # previous 1M candle
                cur1  = c1[-1]   # latest closed 1M candle

                # Pattern detection
                bull_engulf   = is_engulfing_bullish(prev1, cur1)
                bear_engulf   = is_engulfing_bearish(prev1, cur1)
                bull_reject   = is_rejection_bullish(cur1)
                bear_reject   = is_rejection_bearish(cur1)

                bull_pattern  = bull_engulf or bull_reject
                bear_pattern  = bear_engulf or bear_reject

                # 1M close must be back on correct side of 5M EMA20
                cur_close = cur1["close"]
                close_above_ema20 = cur_close > ema20_5
                close_below_ema20 = cur_close < ema20_5

                # Body ratio filter — no dojis
                _body = body_ratio(cur1["open"], cur1["close"], cur1["high"], cur1["low"])
                body_ok = _body >= BODY_RATIO_MIN

                # Spike block — no news candles
                bodies = [abs(c["close"] - c["open"]) for c in c1[-(SPIKE_LOOKBACK+2):-2]]
                avg_body  = sum(bodies) / len(bodies) if bodies else 0
                last_body = abs(cur1["close"] - cur1["open"])
                spike_ok  = not (avg_body > 0 and last_body > SPIKE_MULT * avg_body)

                # ── FULL SIGNAL LOGIC ─────────────────────────────────
                signal = None
                reason = "Scanning..."

                call_ready = (
                    trend_up              and   # 15M uptrend
                    slope_rising          and   # 5M EMA50 momentum rising
                    spread_ok             and   # 5M not flat/ranging
                    pb_touched_ema20_bull and   # 5M pulled back to EMA20
                    pb_close_above_ema50  and   # 5M didn't break through EMA50
                    rsi_ok                and   # RSI healthy
                    bull_pattern          and   # 1M engulfing or rejection
                    close_above_ema20     and   # 1M close back above EMA20
                    body_ok               and   # 1M meaningful body
                    spike_ok                    # not a spike candle
                )
                put_ready = (
                    trend_down            and   # 15M downtrend
                    slope_falling         and   # 5M EMA50 momentum falling
                    spread_ok             and   # 5M not flat/ranging
                    pb_touched_ema20_bear and   # 5M pulled back to EMA20
                    pb_close_below_ema50  and   # 5M didn't break through EMA50
                    rsi_ok                and   # RSI healthy
                    bear_pattern          and   # 1M engulfing or rejection
                    close_below_ema20     and   # 1M close back below EMA20
                    body_ok               and   # 1M meaningful body
                    spike_ok                    # not a spike candle
                )

                trend_label = "UPTREND" if trend_up else "DOWNTREND"

                if call_ready:
                    signal = "CALL"
                    reason = (f"CALL: 15M {trend_label} | 5M slope rising | "
                              f"Pullback EMA20({ema20_5:.5f}) | RSI {rsi5:.1f} | "
                              f"{'Engulf' if bull_engulf else 'Reject'} | Close>{ema20_5:.5f}")
                elif put_ready:
                    signal = "PUT"
                    reason = (f"PUT: 15M {trend_label} | 5M slope falling | "
                              f"Pullback EMA20({ema20_5:.5f}) | RSI {rsi5:.1f} | "
                              f"{'Engulf' if bear_engulf else 'Reject'} | Close<{ema20_5:.5f}")
                else:
                    # Detailed reason for what's missing
                    if not spread_ok:
                        reason = f"5M ranging — EMA spread {ema_spread:.5f} < min {EMA_SPREAD_ATR_MULT*atr5:.5f}"
                    elif not slope_rising and trend_up:
                        reason = f"5M EMA50 slope flat/falling ({ema50_slope:.5f}) — waiting for momentum"
                    elif not slope_falling and trend_down:
                        reason = f"5M EMA50 slope flat/rising ({ema50_slope:.5f}) — waiting for momentum"
                    elif trend_up and not pb_touched_ema20_bull:
                        reason = f"Waiting 5M pullback to EMA20 ({ema20_5:.5f}) — price {cl5[-1]:.5f}"
                    elif trend_down and not pb_touched_ema20_bear:
                        reason = f"Waiting 5M pullback to EMA20 ({ema20_5:.5f}) — price {cl5[-1]:.5f}"
                    elif not rsi_ok:
                        reason = f"RSI {rsi5:.1f} outside {RSI_MIN}–{RSI_MAX}"
                    elif not bull_pattern and trend_up:
                        reason = "Waiting 1M bullish engulfing or rejection"
                    elif not bear_pattern and trend_down:
                        reason = "Waiting 1M bearish engulfing or rejection"
                    elif trend_up and not close_above_ema20:
                        reason = f"1M close {cur_close:.5f} not above EMA20 {ema20_5:.5f}"
                    elif trend_down and not close_below_ema20:
                        reason = f"1M close {cur_close:.5f} not below EMA20 {ema20_5:.5f}"
                    elif not body_ok:
                        reason = f"1M weak body {_body:.2f} < {BODY_RATIO_MIN}"
                    elif not spike_ok:
                        reason = f"Spike block — body {last_body:.5f} > {SPIKE_MULT}x avg {avg_body:.5f}"
                    else:
                        reason = f"No setup — {trend_label} | EMA20:{ema20_5:.5f} RSI:{rsi5:.1f}"

                # ── DEBUG ─────────────────────────────────────────────
                pattern_str = ("BullEngulf" if bull_engulf else
                               "BullReject" if bull_reject else
                               "BearEngulf" if bear_engulf else
                               "BearReject" if bear_reject else "None")
                self.market_debug[symbol] = {
                    "time": time.time(), "gate": gate, "mkt_msg": mkt_msg,
                    "last_m5": c1[-1]["epoch"] if "epoch" in c1[-1] else 0,
                    "signal": signal,
                    "trend_label": trend_label,
                    "ema50_15": round(ema50_15, 5),
                    "ema200_15": round(ema200_15, 5),
                    "ema20_5": round(ema20_5, 5),
                    "ema50_5": round(ema50_5, 5),
                    "ema50_slope": round(ema50_slope, 6),
                    "slope_rising": slope_rising,
                    "slope_falling": slope_falling,
                    "spread_ok": spread_ok,
                    "pb_touch": pb_touched_ema20_bull or pb_touched_ema20_bear,
                    "rsi5": round(rsi5, 1),
                    "rsi_ok": rsi_ok,
                    "pattern": pattern_str,
                    "close_vs_ema20": close_above_ema20 if trend_up else close_below_ema20,
                    "body": round(_body, 2),
                    "body_ok": body_ok,
                    "spike_ok": spike_ok,
                    "mkt_losses": self.market_losses_today.get(symbol, 0),
                    "mkt_trades": self.market_trades_today.get(symbol, 0),
                    "why": [reason]
                }

                if not ok_gate or not mkt_ok: continue
                if signal is None: continue

                if signal == "CALL":
                    await self.execute_trade("CALL", symbol,
                        rsi=round(rsi5, 1), ema20=round(ema20_5, 5),
                        pattern=pattern_str, trend=trend_label)
                elif signal == "PUT":
                    await self.execute_trade("PUT", symbol,
                        rsi=round(rsi5, 1), ema20=round(ema20_5, 5),
                        pattern=pattern_str, trend=trend_label)

            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                await asyncio.sleep(10)

    async def _connection_watchdog(self):
        """Ping every 60s — reconnect immediately if connection is dead"""
        while True:
            await asyncio.sleep(60)
            try:
                if self.api:
                    alive = await self.ping()
                    if not alive:
                        logger.warning("Watchdog: connection dead — reconnecting")
                        await self.connect()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    async def run(self):
        """Simple keepalive — connect and scan controlled by Telegram buttons"""
        logger.info("Bot started — press DEMO or LIVE in Telegram to connect, then START to scan")
        asyncio.create_task(self._connection_watchdog())
        while True:
            await asyncio.sleep(60)


bot_logic = StructureBreakBot()

# ============================================================
# TELEGRAM UI
# ============================================================

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
        [InlineKeyboardButton("⏸ PAUSE", callback_data="PAUSE"),
         InlineKeyboardButton("▶️ RESUME", callback_data="RESUME"),
         InlineKeyboardButton("🔓 UNBLOCK", callback_data="UNBLOCK")],
    ])

def format_market_detail(sym, d):
    if not d: return f"\U0001f4cd {sym.replace('frx','')}\n\u23f3 No scan data yet"
    age        = int(time.time() - d.get("time", time.time()))
    signal     = d.get("signal") or "\u2014"
    why        = d.get("why", [])
    mkt_losses = d.get("mkt_losses", 0)
    mkt_trades = d.get("mkt_trades", 0)
    trend      = d.get("trend_label", "\u2014")
    ema50_15   = d.get("ema50_15",  "\u2014")
    ema200_15  = d.get("ema200_15", "\u2014")
    ema20_5    = d.get("ema20_5",   "\u2014")
    ema50_5    = d.get("ema50_5",   "\u2014")
    slope      = d.get("ema50_slope", 0)
    slope_str  = "Rising \u2191" if d.get("slope_rising") else ("Falling \u2193" if d.get("slope_falling") else "Flat")
    spread_ok  = "\u2705" if d.get("spread_ok") else "\u274c"
    pb_touch   = "\u2705 Touched" if d.get("pb_touch") else "\u23f3 Waiting"
    rsi5       = d.get("rsi5", "\u2014")
    rsi_ok     = "\u2705" if d.get("rsi_ok") else "\u274c"
    pattern    = d.get("pattern", "\u2014")
    close_ok   = "\u2705" if d.get("close_vs_ema20") else "\u274c"
    body       = d.get("body", "\u2014")
    body_ok    = "\u2705" if d.get("body_ok") else "\u274c"
    spike_ok   = "\u2705" if d.get("spike_ok", True) else "\u274c Spike"
    last_m1    = d.get("last_m5", 0)
    return (
        f"\U0001f4cd {sym.replace('frx','')} ({age}s ago)\n"
        f"Market: {d.get('mkt_msg','OK')} | {mkt_trades}/{MAX_TRADES_PER_MARKET} | {mkt_losses}/{MAX_LOSSES_PER_MARKET} losses\n"
        f"Last 1M: {fmt_time_hhmmss(last_m1)}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"15M Trend: {trend} | EMA50:{ema50_15} EMA200:{ema200_15}\n"
        f"5M Slope: {slope_str} ({slope:.6f}) | Spread: {spread_ok}\n"
        f"5M EMA20:{ema20_5} EMA50:{ema50_5}\n"
        f"5M Pullback: {pb_touch} | RSI:{rsi5} {rsi_ok}\n"
        f"1M Pattern: {pattern} | Close vs EMA20: {close_ok}\n"
        f"1M Body: {body} {body_ok} | Spike: {spike_ok}\n"
        f"Signal: {signal}\n"
        f"Why: {why[0] if why else chr(8212)}\n"
    )


async def _safe_answer(q, text=None, show_alert=False):
    try: await q.answer(text=text, show_alert=show_alert)
    except Exception as e: logger.warning(f"Callback answer: {e}")

async def _safe_edit(q, text, reply_markup=None):
    try:
        kwargs = {"text": text[:4000]}
        if reply_markup: kwargs["reply_markup"] = reply_markup
        await q.edit_message_text(**kwargs)
    except Exception as e: logger.warning(f"Edit failed: {e}")

async def btn_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return

    # Register chat ID
    if not hasattr(bot_logic, "_known_chat_ids"):
        bot_logic._known_chat_ids = set()
    bot_logic._known_chat_ids.add(q.message.chat_id)

    # Always answer immediately — stops button spinner
    try:
        await q.answer()
    except Exception:
        pass

    # Show working indicator
    try:
        await q.edit_message_text("⏳ Working...", reply_markup=main_keyboard())
    except Exception:
        pass

    try:
        if q.data == "SET_DEMO":
            bot_logic.account_type = "DEMO"
            ok = await bot_logic.connect()
            await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO connection failed", reply_markup=main_keyboard())

        elif q.data == "SET_REAL":
            bot_logic.account_type = "LIVE"
            ok = await bot_logic.connect()
            await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE connection failed", reply_markup=main_keyboard())

        elif q.data == "START_SCAN":
            if not bot_logic.api:
                await _safe_edit(q, "❌ Connect first — press DEMO or LIVE.", reply_markup=main_keyboard())
                return
            # Stop any existing tasks then restart fresh
            bot_logic.is_scanning = False
            await asyncio.sleep(0.3)
            bot_logic.is_scanning = True
            for sym in MARKETS:
                asyncio.create_task(bot_logic.scan_market(sym))
            await _safe_edit(q,
                "🔍 SCANNER ACTIVE\n"
                "✅ M5 structure | M1 EMA50 cross + RSI + confirm\n"
                "📡 EURUSD GBPUSD USDJPY AUDUSD GBPJPY",
                reply_markup=main_keyboard())

        elif q.data == "STOP_SCAN":
            bot_logic.is_scanning = False
            await _safe_edit(q, "⏹️ Scanner stopped. Press START to begin again.", reply_markup=main_keyboard())

        elif q.data == "TEST_BUY":
            if not bot_logic.api:
                await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
                return
            test_symbol = MARKETS[0]
            asyncio.create_task(bot_logic.execute_trade("CALL", test_symbol,
                rsi=55.0, body=0.45, level=0.0, ema50=0.0))
            await _safe_edit(q, f"🧪 Test CALL on {test_symbol.replace('frx','')}.", reply_markup=main_keyboard())

        elif q.data == "PAUSE":
            bot_logic.pause_until = time.time() + 86400
            await _safe_edit(q, "⏸ Bot paused for 24 hours.", reply_markup=main_keyboard())

        elif q.data == "RESUME":
            bot_logic.pause_until = 0
            await _safe_edit(q, "▶️ Bot resumed.", reply_markup=main_keyboard())

        elif q.data == "UNBLOCK":
            for m in MARKETS:
                bot_logic.market_blocked[m] = False
                bot_logic.market_pause_until[m] = 0
                bot_logic.market_losses_today[m] = 0
            await _safe_edit(q, "🔓 All pairs unblocked.", reply_markup=main_keyboard())

        elif q.data == "STATUS":
            now = time.time()
            if now < bot_logic.status_cooldown_until:
                await _safe_edit(q, f"⏳ Cooldown: {int(bot_logic.status_cooldown_until - now)}s", reply_markup=main_keyboard())
                return
            bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC
            try:
                await asyncio.wait_for(bot_logic.fetch_balance(), timeout=3.0)
            except Exception:
                pass
            now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
            _, gate = bot_logic.can_auto_trade()
            trade_status = "No Active Trade"
            if bot_logic.active_trade_info:
                rem = max(0, int(EXPIRY_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                trade_status = f"🚀 Active Trade ({bot_logic.active_market.replace('frx','')})\n⏳ Left: ~{rem}s"
            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))
            by_mkt, by_sess, wr = bot_logic.stats_30d()
            def fmt_stats(title, items):
                rows = [(k, wr(v), v["trades"], v["wins"]) for k, v in items.items()]
                rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
                lines_ = [f"{title} (last {STATS_DAYS}d):"]
                if not rows:
                    lines_.append("— No trades yet")
                    return "\n".join(lines_)
                for k, wrr, t, w in rows:
                    lines_.append(f"- {k}: {wrr:.1f}% ({w}/{t})")
                return "\n".join(lines_)
            stats_block = "📈 PERFORMANCE\n" + fmt_stats("Pairs", by_mkt) + "\n" + fmt_stats("Sessions", by_sess) + "\n"
            mkt_lines = ["🛡 Pair Status:"]
            for m in MARKETS:
                ml = bot_logic.market_losses_today.get(m, 0)
                mt = bot_logic.market_trades_today.get(m, 0)
                mb = bot_logic.market_blocked.get(m, False)
                mp = bot_logic.market_pause_until.get(m, 0)
                status = "🚫 BLOCKED" if mb else ("⏸ PAUSED" if time.time() < mp else "✅")
                mkt_lines.append(f"{status} {m.replace('frx','')}: {mt}/{MAX_TRADES_PER_MARKET} | {ml}/{MAX_LOSSES_PER_MARKET} losses")
            mkt_block = "\n".join(mkt_lines) + "\n"
            _, _, stake_mult = bot_logic._equity_ok()
            eq_ratio = bot_logic._get_current_balance_float() / bot_logic.starting_balance if bot_logic.starting_balance > 0 else 1.0
            pause_line = "Paused" if time.time() < bot_logic.pause_until else ""
            pause_line_nl = ("⏸ Paused\n") if time.time() < bot_logic.pause_until else ""
            scan_status = "ACTIVE" if bot_logic.is_scanning else "OFFLINE"
            lock_str = ("ON +$" + f"{PROFIT_LOCK_FLOOR:.2f}") if bot_logic.profit_lock_active else "OFF"
            header = (
                f"🕒 {now_time}\n"
                f"🤖 {scan_status} ({bot_logic.account_type})\n"
                f"{pause_line_nl}"
                f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"🧯 Max stake: ${MAX_STAKE_ALLOWED:.2f} | Mult: {stake_mult:.1f}x\n"
                f"💰 Equity: {eq_ratio:.0%} | 🔒 Lock: {lock_str}\n"
                f"🎯 Target: +${DAILY_PROFIT_TARGET:.2f} | Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
                f"📡 Pairs: EURUSD GBPUSD USDJPY AUDUSD GBPJPY\n"
                f"🧭 15M trend (EMA50/200) + 5M pullback (EMA20+slope) + 1M entry (engulf/reject) | {EXPIRY_MIN}m\n"
                f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
                f"{stats_block}{mkt_block}"
                f"💵 PnL: {bot_logic.total_profit_today:+.2f} | Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY}\n"
                f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Losses: {bot_logic.total_losses_today}\n"
                f"🚦 Gate: {gate}\n"
                f"💰 Balance: {bot_logic.balance}\n"
                f"\n/pause /resume /unblock /stats"
            )
            details = "\n\n📌 LIVE SCAN\n\n" + "\n\n".join([
                format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS
            ])
            await _safe_edit(q, header + details, reply_markup=main_keyboard())

    except Exception as e:
        logger.error(f"btn_handler error: {e}")
        try:
            await q.answer("⚠️ Error occurred. Try again.")
        except Exception:
            pass

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic._known_chat_ids = getattr(bot_logic, "_known_chat_ids", set())
    bot_logic._known_chat_ids.add(update.message.chat_id)
    await update.message.reply_text(
        "💎 Deriv Forex BB Pullback Bot\n"
        f"🧭 Strategy: Bollinger Band Pullback | EMA100 + BB + RSI\n"
        f"📊 CALL: Above EMA100 + Lower BB touch + RSI<40 up + Bullish candle\n"
        f"📊 PUT: Below EMA100 + Upper BB touch + RSI>60 down + Bearish candle\n"
        f"🌍 Pairs: EURUSD | GBPUSD | USDJPY | AUDUSD | GBPJPY\n"
        f"📅 Trades Mon-Fri all sessions (data collection mode)\n"
        f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps × {MARTINGALE_MULT}x\n"
        f"⏱ Expiry: {EXPIRY_MIN}m | Cooldown: {COOLDOWN_SEC//60}m\n"
        f"🛡 Max {MAX_TRADES_PER_MARKET} trades/pair | Stop after {MAX_LOSSES_PER_MARKET} losses/pair\n",
        reply_markup=main_keyboard()
    )

async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until = time.time() + 86400
    await update.message.reply_text("⏸ Bot paused for 24 hours.", reply_markup=main_keyboard())

async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until = 0
    await update.message.reply_text("▶️ Bot resumed.", reply_markup=main_keyboard())

async def cmd_unblock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for m in MARKETS:
        bot_logic.market_blocked[m] = False
        bot_logic.market_pause_until[m] = 0
        bot_logic.market_losses_today[m] = 0
    await update.message.reply_text("🔓 All pairs unblocked.", reply_markup=main_keyboard())

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    by_mkt, by_sess, wr = bot_logic.stats_30d()
    lines = [f"📈 STATS (last {STATS_DAYS}d)\n"]
    for k, v in sorted(by_mkt.items(), key=lambda x: -x[1]["trades"]):
        wins = v["wins"]; trades = v["trades"]
        lines.append(f"  {k}: {wr(v):.1f}% ({wins}/{trades})")
    lines.append("")
    for k, v in sorted(by_sess.items(), key=lambda x: -x[1]["trades"]):
        wins = v["wins"]; trades = v["trades"]
        lines.append(f"  {k}: {wr(v):.1f}% ({wins}/{trades})")
    lines.append(f"\nToday: {bot_logic.total_profit_today:+.2f} | Trades: {bot_logic.trades_today}")
    await update.message.reply_text("\n".join(lines), reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("unblock", cmd_unblock))
    app.add_handler(CommandHandler("stats", cmd_stats))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bot_logic.run())
    app.run_polling(close_loop=False)
