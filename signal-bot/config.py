import os

VERSION = '2026-11-25c (A1 + A2, тест B1–B4, новый формат)'

# --- secrets (Replit -> Secrets) ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# --- settings ---
from strategy import STRATEGIES

ALL_PAIRS = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurjpy', 'gbpjpy', 'xauusd']
# strategies: Secret BOT_STRATEGIES (comma list); unknown ids (e.g. old S1/T1-T8) are ignored; nothing valid -> all current strategies
ENABLED = [s.strip().upper() for s in os.environ.get('BOT_STRATEGIES', '').split(',') if s.strip().upper() in STRATEGIES] or list(STRATEGIES)
# pairs: only those the enabled strategies trade (optionally narrowed by Secret PAIRS)
_need = [p for p in ALL_PAIRS if any(p in STRATEGIES[k]['pairs'] for k in ENABLED)]
_env = [p.strip().lower() for p in os.environ.get('PAIRS', '').split(',') if p.strip()]
PAIRS = [p for p in _need if p in _env] or _need

HISTORY_DAYS = 35          # history kept in memory (needs >= 20 trading days for the ATR average; 30 and 35 days validated identical to the full-history backtest)
CANDLE_WAIT_SEC = 4        # wait after minute close before polling (let the feed finish the minute)
CANDLE_MAX_WAIT_SEC = 20   # max extra wait for a pair's last minute

DUKA_CODE = {p: f'{p[:3].upper()}-{p[3:].upper()}' for p in ALL_PAIRS}   # dukascopy-python instrument codes
DISPLAY = {p: f'{p[:3].upper()}/{p[3:].upper()}' for p in ALL_PAIRS}
DISPLAY['xauusd'] = 'XAU/USD (Gold)'
PIP = {p: (0.01 if p.endswith('jpy') else 0.0001) for p in ALL_PAIRS}
PIP['xauusd'] = 0.1      # gold: 1 pip = 0.10 USD
DIGITS = {p: (3 if p.endswith('jpy') else 5) for p in ALL_PAIRS}
DIGITS['xauusd'] = 2


def fmt_price(pair, v):
    try:
        return f'{float(v):.{DIGITS[pair]}f}'
    except (TypeError, ValueError):
        return '-'

DATA_DIR = os.environ.get('DATA_DIR', 'data')
SIGNALS_CSV = os.path.join(DATA_DIR, 'signals.csv')
