import os

# --- secrets (Replit -> Secrets) ---
OANDA_TOKEN = os.environ.get('OANDA_TOKEN', '')
OANDA_ENV = os.environ.get('OANDA_ENV', 'practice')            # practice | live
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# --- settings ---
ALL_PAIRS = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurjpy', 'gbpjpy', 'xauusd']
PAIRS = [p.strip().lower() for p in os.environ.get('PAIRS', ','.join(ALL_PAIRS)).split(',') if p.strip()]
ENABLED = [s.strip().upper() for s in os.environ.get('STRATEGIES', 'MORNING,DAY').split(',') if s.strip()]

HISTORY_DAYS = 16          # history kept in memory (bot was validated on 16 days = identical to full-history backtest)
CANDLE_WAIT_SEC = 4        # wait after minute close before polling (OANDA finalises M1 on next tick)
CANDLE_MAX_WAIT_SEC = 20   # max extra wait for a pair's last minute

OANDA_URL = 'https://api-fxpractice.oanda.com' if OANDA_ENV == 'practice' else 'https://api-fxtrade.oanda.com'
OANDA_SYMBOL = {p: f'{p[:3].upper()}_{p[3:].upper()}' for p in ALL_PAIRS}
DISPLAY = {p: f'{p[:3].upper()}/{p[3:].upper()}' for p in ALL_PAIRS}
DISPLAY['xauusd'] = 'XAU/USD (Gold)'

DATA_DIR = os.environ.get('DATA_DIR', 'data')
SIGNALS_CSV = os.path.join(DATA_DIR, 'signals.csv')
