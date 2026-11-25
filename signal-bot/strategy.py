"""Signal logic — identical to the backtest (bt/al_search.py, signal SLEEP_BRK + filter H1_EMA_W, checked in bt/al_check.py).

A1  USD/JPY · 08:00–09:00 New York (16:00–17:00 Dubai in US summer time, 17:00–18:00 in winter) · 1m · expiry 60 min
    Alligator (13/8/5 on median price, shifts 8/5/3, smoothing = SMMA/RMA) has been "asleep": mouth width
    (max line − min line) < 0.5 × ATR14 (Wilder) on each of the previous 10 candles. The candle closes outside the mouth
    (above all three lines -> CALL, below -> PUT) and the breakout is in the direction of the H1 trend
    (last finished UTC-hour close vs EMA50 of hourly closes). Trade WITH the breakout. Only the first signal of the day.
    Entry: manual ~1 min after the candle close (entry time = close + 1 min, must be 08:00–08:59 New York).
Backtest, first signal per day: 2020 58.7% · 2021 61.0% · 2022 58.8% · 2023 57.4% · 2024 66.0% · 2025 60.9% · 2026 68.1% (360 days).
"""
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

MIN = 60_000
NY = ZoneInfo('America/New_York')

STRATEGIES = {
    'A1': dict(name='Нью-Йорк 08–09 (Дубай 16–17 летом / 17–18 зимой)', what='выход Аллигатора из спячки по тренду H1',
               model='ALLI_SLEEP_BRK', tf=1, pairs=['usdjpy'], ny_start=480, ny_end=540, expiry=60,
               test=False, session='NY', first_per_day=True),
}
SESSIONS = {'NY': dict(name='Нью-Йорк 08–09 (USD/JPY)', report_utc=930)}   # daily report 19:30 Dubai, after the last 60-min expiry


def ny_minute(t_ms):
    """Minute of day in New York time (DST-aware) for an array of ms timestamps."""
    ny = pd.to_datetime(np.asarray(t_ms, dtype=np.int64), unit='ms', utc=True).tz_convert(NY)
    return np.asarray(ny.hour * 60 + ny.minute)


def in_window(te, s):
    """Entry time (candle close + 1 min) inside the strategy's New York window."""
    m = ny_minute(te)
    return (m >= s['ny_start']) & (m < s['ny_end'])


def _rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _bars(df, tf):
    key = df.timestamp.values // (tf * MIN)
    g = df.groupby(key, sort=True)
    b = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
    b['tc'] = (b.index.values + 1) * tf * MIN
    return b.reset_index(drop=True)


def _h1_trend(df):
    """UTC-hour candles, finished hours only: tc = close time, sign(close - EMA50)."""
    now = int(df.timestamp.iloc[-1]) + MIN
    g = df.groupby(df.timestamp.values // 3_600_000, sort=True)
    a = pd.DataFrame({'c': g.close.last(), 'tc': g.timestamp.max() + MIN})
    a = a[(a.index.values + 1) * 3_600_000 <= now].reset_index(drop=True)
    a['e50'] = a.c.ewm(span=50, adjust=False).mean()
    return a.tc.values.astype(np.int64), np.sign(a.c.values - a.e50.values)


def _alli_sleep_break(b):
    h, l, c = b.h, b.l, b.c
    med = (h + l) / 2
    jaw, teeth, lips = _rma(med, 13).shift(8), _rma(med, 8).shift(5), _rma(med, 5).shift(3)
    top = np.fmax(np.fmax(jaw, teeth), lips)
    bot = np.fmin(np.fmin(jaw, teeth), lips)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = _rma(tr, 14)
    width = (top - bot) / atr
    sleep = (width.rolling(10).max() < 0.5).shift(1).fillna(False).astype(bool)
    up = (sleep & (c > top)).values
    dn = (sleep & (c < bot)).values
    return np.where(up & ~dn, 1, np.where(dn & ~up, -1, 0))


def scan(m1: pd.DataFrame, pair: str, only=None) -> pd.DataFrame:
    """m1: columns timestamp (ms UTC, candle open), open, high, low, close (finished candles only).
    Returns every signal found in the given history: pair, strategy, t_close, t_entry (= t_close + 1 min),
    direction (+1 CALL / -1 PUT), expiry, price (candle close), lvl (unused, 0)."""
    df = m1.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    if len(df) < 3000:
        return pd.DataFrame()
    out = []
    for sid in (only or list(STRATEGIES)):
        s = STRATEGIES[sid]
        if pair not in s['pairs']:
            continue
        b = _bars(df, s['tf'])
        d = _alli_sleep_break(b)
        htc, hsg = _h1_trend(df)
        i = np.searchsorted(htc, b.tc.values, side='right') - 1
        trend = np.where(i >= 0, hsg[np.clip(i, 0, None)], 0)
        tc = b.tc.values.astype(np.int64)
        te = tc + MIN
        m = (d != 0) & (trend == d) & in_window(te, s)
        j = np.where(m)[0]
        if not len(j):
            continue
        r = pd.DataFrame({'pair': pair, 'strategy': sid, 't_close': tc[j], 't_entry': te[j], 'direction': d[j].astype(int),
                          'expiry': s['expiry'], 'price': b.c.values[j], 'lvl': 0})
        if s.get('first_per_day'):
            day = pd.to_datetime(r.t_entry, unit='ms', utc=True).dt.tz_convert(NY).dt.date
            r = r[~day.duplicated()]
        out.append(r)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values(['t_close', 'strategy']).reset_index(drop=True)
