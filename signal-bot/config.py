import os

# --- secrets (Replit -> Secrets) ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# --- settings ---
ALL_PAIRS = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurjpy', 'gbpjpy', 'xauusd']
PAIRS = [p.strip().lower() for p in os.environ.get('PAIRS', ','.join(ALL_PAIRS)).split(',') if p.strip()]
ENABLED = [s.strip().upper() for s in os.environ.get('STRATEGIES', 'MORNING,DAY').split(',') if s.strip()]

HISTORY_DAYS = 35          # history kept in memory (needs >= 20 trading days for the ATR average; 30 and 35 days validated identical to the full-history backtest)
CANDLE_WAIT_SEC = 4        # wait after minute close before polling (let the feed finish the minute)
CANDLE_MAX_WAIT_SEC = 20   # max extra wait for a pair's last minute

DUKA_CODE = {p: f'{p[:3].upper()}-{p[3:].upper()}' for p in ALL_PAIRS}   # dukascopy-python instrument codes
DISPLAY = {p: f'{p[:3].upper()}/{p[3:].upper()}' for p in ALL_PAIRS}
DISPLAY['xauusd'] = 'XAU/USD (Gold)'

DATA_DIR = os.environ.get('DATA_DIR', 'data')
SIGNALS_CSV = os.path.join(DATA_DIR, 'signals.csv')
