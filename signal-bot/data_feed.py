"""Dukascopy free chart feed (no registration, no token) via the `dukascopy-python` package.
Same source as the backtest. M1 candles, mid = (bid + ask) / 2, only minutes that had ticks (volume > 0),
only finished minutes. timestamp = candle open, ms UTC."""
import time
from datetime import datetime, timezone
import pandas as pd
import dukascopy_python as dp
import config

MIN = 60_000
COLS = ['timestamp', 'open', 'high', 'low', 'close']


def _side(pair, side, start_ms, end_ms):
    rows = dp.fetch_candles(config.DUKA_CODE[pair], 1, dp.TimeUnit.MINUTE, side,
                            datetime.fromtimestamp(start_ms / 1000, timezone.utc),
                            datetime.fromtimestamp(end_ms / 1000, timezone.utc))
    out = []
    for c in rows:
        ts = c.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append((int(ts.timestamp() * 1000), c.open, c.high, c.low, c.close, c.volume))
    return pd.DataFrame(out, columns=COLS + ['volume'])


def candles(pair, from_ms, to_ms=None, tries=4):
    """Finished M1 mid candles with open time in [from_ms, to_ms)."""
    now = int(time.time() * 1000)
    to_ms = min(to_ms or now, now)
    if to_ms - from_ms < MIN:
        return pd.DataFrame(columns=COLS)
    err = None
    for k in range(tries):
        try:
            b = _side(pair, dp.OfferSide.BID, from_ms, to_ms)
            a = _side(pair, dp.OfferSide.ASK, from_ms, to_ms)
            break
        except Exception as e:
            err = e
            time.sleep(2 * (k + 1))
    else:
        raise RuntimeError(f'Dukascopy feed failed for {pair}: {err}')
    if b.empty or a.empty:
        return pd.DataFrame(columns=COLS)
    m = b.merge(a, on='timestamp', suffixes=('', '_a'))
    m = m[(m.volume > 0) & (m.timestamp + MIN <= now)]          # ticked + finished minutes only (as in backtest data)
    for c in ['open', 'high', 'low', 'close']:
        m[c] = (m[c] + m[c + '_a']) / 2
    return m[COLS].drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def history(pair, days):
    now = int(time.time() * 1000)
    return candles(pair, now - days * 1440 * MIN, now)


def close_at(pair, t_ms):
    """Close of the last finished M1 candle that closes at or before t_ms (same rule as the backtest)."""
    d = candles(pair, t_ms - 20 * MIN, t_ms)
    d = d[d.timestamp + MIN <= t_ms]
    if d.empty or int(d.timestamp.iloc[-1]) + MIN < t_ms - 15 * MIN:
        return None
    return round(float(d.close.iloc[-1]), 6)
