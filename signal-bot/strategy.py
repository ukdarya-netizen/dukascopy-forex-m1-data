"""Signal logic — identical to the backtest (bt/al_search.py, signal SLEEP_BRK + filter H1_EMA_W, checked in bt/al_check.py).

A1  USD/JPY · 08:00–09:00 New York (16:00–17:00 Dubai in US summer time, 17:00–18:00 in winter) · 1m · expiry 60 min
    Alligator (13/8/5 on median price, shifts 8/5/3, smoothing = SMMA/RMA) has been "asleep": mouth width
    (max line − min line) < 0.5 × ATR14 (Wilder) on each of the previous 10 candles. The candle closes outside the mouth
    (above all three lines -> CALL, below -> PUT) and the breakout is in the direction of the H1 trend
    (last finished UTC-hour close vs EMA50 of hourly closes). Trade WITH the breakout. Only the first signal of the day.
    Entry: manual ~1 min after the candle close (entry time = close + 1 min, must be 08:00–08:59 New York).
Backtest, first signal per day: 2020 58.7% · 2021 61.0% · 2022 58.8% · 2023 57.4% · 2024 66.0% · 2025 60.9% · 2026 68.1% (360 days).

A2  USD/JPY · 22:00–23:00 New York (06:00–07:00 Dubai in US summer time, 07:00–08:00 in winter) · 1m · expiry 15 min
    (bt/bw_combo.py: FR_BRK_V | ALI_HTF | ADX35 | none, faded; checked in bt/bw_check.py)
    Williams-valid fractal (5 bars; up fractal high above the Alligator teeth / down fractal low below the teeth, confirmed
    2 bars later) is broken by a candle close. The trend is strong: ADX14 (Wilder) > 35 on 1m, and the Alligator on 5m
    (last finished 5m candle) is open in the direction of the breakout. Trade AGAINST the breakout. Only the first signal of the day.
Backtest, first signal per day: 2020 75% · 2021 55% · 2022 64% · 2023 69% · 2024 68% · 2025 62% · 2026 70% (297 days, 66.0%).
"""
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

MIN = 60_000
NY = ZoneInfo('America/New_York')

STRATEGIES = {
    'A2': dict(name='Утро (Дубай 06–07 летом / 07–08 зимой)', what='против пробоя фрактала Вильямса в сильном тренде (ADX > 35, Аллигатор 5m раскрыт)',
               model='FRACTAL_FADE', tf=1, pairs=['usdjpy'], ny_start=1320, ny_end=1380, expiry=15,
               test=False, session='MORNING', first_per_day=True),
    'A1': dict(name='Вечер (Дубай 16–17 летом / 17–18 зимой)', what='выход Аллигатора из спячки по тренду H1',
               model='ALLI_SLEEP_BRK', tf=1, pairs=['usdjpy'], ny_start=480, ny_end=540, expiry=60,
               test=False, session='NY', first_per_day=True),
}
SESSIONS = {'MORNING': dict(name='Утро (A2, USD/JPY)', report_utc=270),   # daily report 08:30 Dubai, after the last 15-min expiry
            'NY': dict(name='Вечер (A1, USD/JPY)', report_utc=930)}      # daily report 19:30 Dubai, after the last 60-min expiry


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


def _alligator(b):
    med = (b.h + b.l) / 2
    jaw, teeth, lips = _rma(med, 13).shift(8), _rma(med, 8).shift(5), _rma(med, 5).shift(3)
    ali = np.where((lips > teeth) & (teeth > jaw), 1, np.where((lips < teeth) & (teeth < jaw), -1, 0))
    return teeth, ali


def _fractal_fade(b, b5):
    """Returns (d, ok): d = breakout direction of a Williams-valid fractal (+1 up / -1 down), ok = ADX>35 and 5m Alligator open with d.
    The trade direction is -d."""
    h, l, c = b.h, b.l, b.c; n = len(b)
    teeth, _ = _alligator(b)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs())); atr = _rma(tr, 14)
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.)); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.))
    pdi = 100 * _rma(pdm, 14) / atr.values; ndi = 100 * _rma(ndm, 14) / atr.values
    adx = _rma(100 * (pdi - ndi).abs() / (pdi + ndi), 14).values
    hv, lv, th = h.values, l.values, teeth.values
    upf = np.zeros(n, bool); dnf = np.zeros(n, bool)
    upf[2:-2] = (hv[2:-2] > hv[1:-3]) & (hv[2:-2] > hv[:-4]) & (hv[2:-2] > hv[3:-1]) & (hv[2:-2] > hv[4:])
    dnf[2:-2] = (lv[2:-2] < lv[1:-3]) & (lv[2:-2] < lv[:-4]) & (lv[2:-2] < lv[3:-1]) & (lv[2:-2] < lv[4:])
    uc = np.zeros(n, bool); dc = np.zeros(n, bool); uc[4:] = upf[2:-2]; dc[4:] = dnf[2:-2]   # confirmed 2 bars later
    upv = pd.Series(np.where(uc & (np.roll(hv, 2) > np.roll(th, 2)), np.roll(hv, 2), np.nan)).ffill()
    dnv = pd.Series(np.where(dc & (np.roll(lv, 2) < np.roll(th, 2)), np.roll(lv, 2), np.nan)).ffill()
    lg = ((c > upv) & (c.shift(1) <= upv)).values; sh = ((c < dnv) & (c.shift(1) >= dnv)).values
    d = np.where(lg & ~sh, 1, np.where(sh & ~lg, -1, 0))
    _, ali5 = _alligator(b5)
    i5 = np.searchsorted(b5.tc.values, b.tc.values, side='right') - 1
    a5 = np.where(i5 >= 0, ali5[np.clip(i5, 0, None)], 0)
    return d, (adx > 35) & (a5 == d)


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
        tc = b.tc.values.astype(np.int64)
        te = tc + MIN
        if s['model'] == 'ALLI_SLEEP_BRK':
            d = _alli_sleep_break(b)
            htc, hsg = _h1_trend(df)
            i = np.searchsorted(htc, b.tc.values, side='right') - 1
            trend = np.where(i >= 0, hsg[np.clip(i, 0, None)], 0)
            m = (d != 0) & (trend == d) & in_window(te, s)
        else:   # FRACTAL_FADE: trade against the breakout
            d0, ok = _fractal_fade(b, _bars(df, s['tf'] * 5))
            m = (d0 != 0) & ok & in_window(te, s)
            d = -d0
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
