"""OANDA v20 REST: M1 mid candles. Only complete candles are used. timestamp = candle open, ms UTC."""
import time
import requests
import pandas as pd
import config

MIN = 60_000
_s = requests.Session()


def _get(symbol, params, tries=4):
    url = f'{config.OANDA_URL}/v3/instruments/{symbol}/candles'
    headers = {'Authorization': f'Bearer {config.OANDA_TOKEN}', 'Accept-Datetime-Format': 'UNIX'}
    for k in range(tries):
        try:
            r = _s.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json().get('candles', [])
            if r.status_code in (401, 403):
                raise RuntimeError(f'OANDA auth error {r.status_code}: {r.text[:200]}')
            err = f'{r.status_code} {r.text[:200]}'
        except requests.RequestException as e:
            err = str(e)
        time.sleep(2 * (k + 1))
    raise RuntimeError(f'OANDA request failed for {symbol}: {err}')


def candles(pair, from_ms, count=5000):
    """Complete M1 mid candles starting at from_ms (inclusive)."""
    js = _get(config.OANDA_SYMBOL[pair], {'price': 'M', 'granularity': 'M1',
                                          'from': f'{from_ms / 1000:.0f}', 'count': count})
    rows = [(int(float(c['time']) * 1000), float(c['mid']['o']), float(c['mid']['h']),
             float(c['mid']['l']), float(c['mid']['c'])) for c in js if c.get('complete')]
    return pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close'])


def history(pair, days):
    now = int(time.time() * 1000)
    start = now - days * 1440 * MIN
    parts = []
    while start < now:
        d = candles(pair, start)
        if d.empty:   # nothing after 'start' yet (OANDA skips weekend gaps itself)
            break
        parts.append(d)
        start = int(d.timestamp.iloc[-1]) + MIN
        if len(d) < 4900 and start > now - 10 * MIN:
            break
    if not parts:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
    return pd.concat(parts, ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def close_at(pair, t_ms):
    """Close of the last complete M1 candle that closes at or before t_ms (same rule as the backtest)."""
    d = candles(pair, t_ms - 20 * MIN, count=40)
    d = d[d.timestamp + MIN <= t_ms]
    if d.empty or int(d.timestamp.iloc[-1]) + MIN < t_ms - 15 * MIN:
        return None
    return float(d.close.iloc[-1])
