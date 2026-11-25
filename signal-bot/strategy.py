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

TEST strategies B1–B4: Dubai morning (fixed Dubai time, UTC+4), all 10 pairs, every signal (no per-day limit).
Backtest win rates 2020–22 / 2023–25.06 / 2025.07–26.10 (all pairs together) — near breakeven, kept for live testing on request.
B1  05:00–07:00 · 1m · 15 min · three candles of one colour into a 20-bar extreme (3 red at the 20-bar low -> CALL, mirror -> PUT)
    AND Williams %R(28) has already turned through its EMA8 in the trade direction.  55.1 / 53.7 / 57.5 %, ~68/month (bt/streak_wr.py)
B2  05:00–07:00 · 15m · 5 min · a candle closes through the last Williams fractal level (5-bar fractal, confirmed 2 bars later)
    and this candle completes a morning star (break up) / evening star (break down) -> trade AGAINST the break.  58.5 / 57.5 / 54.3 %, ~21/month (bt/frac_pa.py)
B3  06:00–07:00 · 15m · 15 min · marubozu (body >= 90% of range and >= ATR14) in a calm market (ATR14 < 0.8 x 20-day avg) -> AGAINST it.
    55.4 / 58.0 / 57.8 %, ~13/month (bt/pa_morning.py)
B4  06:00–07:00 · 15m · 3 min · doji (body <= 10% of range) at the 10-bar low/high in a calm market -> reversal.  58.1 / 55.3 / 55.6 %, ~12/month (bt/pa_morning.py)
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
    'B1': dict(name='Утро 05–07', what='3 свечи одного цвета в 20-свечной экстремум + Williams %R(28) развернулся через EMA8 → разворот',
               model='STREAK_WILLY', tf=1, pairs='all', dxb_start=300, dxb_end=420, expiry=15, test=True, session='MORNING'),
    'B2': dict(name='Утро 05–07', what='пробой фрактала Вильямса свечой «утренняя/вечерняя звезда» → против пробоя',
               model='FRACTAL_STAR', tf=15, pairs='all', dxb_start=300, dxb_end=420, expiry=5, test=True, session='MORNING'),
    'B3': dict(name='Утро 06–07', what='марубозу на спокойном рынке → против свечи',
               model='MARUBOZU_CALM', tf=15, pairs='all', dxb_start=360, dxb_end=420, expiry=15, test=True, session='MORNING'),
    'B4': dict(name='Утро 06–07', what='доджи у 10-свечного экстремума на спокойном рынке → разворот',
               model='DOJI_CALM', tf=15, pairs='all', dxb_start=360, dxb_end=420, expiry=3, test=True, session='MORNING'),
}
ALL_PAIRS = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurjpy', 'gbpjpy', 'xauusd']
for _s in STRATEGIES.values():
    if _s['pairs'] == 'all':
        _s['pairs'] = list(ALL_PAIRS)
SESSIONS = {'MORNING': dict(name='Утро (A2 USD/JPY, тест B1–B4)', report_utc=270),   # daily report 08:30 Dubai, after the last morning expiry
            'NY': dict(name='Вечер (A1, USD/JPY)', report_utc=930)}      # daily report 19:30 Dubai, after the last 60-min expiry


def ny_minute(t_ms):
    """Minute of day in New York time (DST-aware) for an array of ms timestamps."""
    ny = pd.to_datetime(np.asarray(t_ms, dtype=np.int64), unit='ms', utc=True).tz_convert(NY)
    return np.asarray(ny.hour * 60 + ny.minute)


def in_window(te, s):
    """Entry time (candle close + 1 min) inside the strategy's window: New York time (DST-aware) or fixed Dubai time (UTC+4)."""
    if 'dxb_start' in s:
        m = ((np.asarray(te, dtype=np.int64) // MIN) + 240) % 1440
        return (m >= s['dxb_start']) & (m < s['dxb_end'])
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


def _ds(lg, sh):
    lg = np.asarray(pd.Series(lg).fillna(False), bool); sh = np.asarray(pd.Series(sh).fillna(False), bool)
    return np.where(lg & ~sh, 1, np.where(sh & ~lg, -1, 0))


def _candles(b, tf):
    """Candle patterns exactly as in bt/pa_morning.py. Returns dict of direction arrays + calm mask."""
    o, h, l, c = b.o, b.h, b.l, b.c
    rng = h - l; body = (c - o).abs(); up = c > o; dn = c < o
    tr = np.maximum(rng, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs())); atr = tr.rolling(14).mean()
    lo10 = l.rolling(10).min(); hi10 = h.rolling(10).max()
    small = body.shift() <= 0.3 * rng.shift()
    doji = (body <= 0.1 * rng) & (rng > 0)
    out = {'STAR': _ds(dn.shift(2) & small & up & (c > (o.shift(2) + c.shift(2)) / 2), up.shift(2) & small & dn & (c < (o.shift(2) + c.shift(2)) / 2)),
           'MARUBOZU': _ds((body >= 0.9 * rng) & (body >= atr) & up, (body >= 0.9 * rng) & (body >= atr) & dn),
           'DOJI_EXT': _ds(doji & (l <= lo10), doji & (h >= hi10))}
    calm = (atr / atr.rolling(int(20 * 1440 / tf), min_periods=200).mean() < 0.8).values
    return out, calm


def _streak_willy(b):
    o, h, l, c = b.o, b.h, b.l, b.c
    up = c > o; dn = c < o
    sd = (dn & dn.shift() & dn.shift(2) & (l <= l.rolling(20).min())).values
    su = (up & up.shift() & up.shift(2) & (h >= h.rolling(20).max())).values
    d = np.where(sd, 1, np.where(su, -1, 0))
    w28 = (-100 * (h.rolling(28).max() - c) / (h.rolling(28).max() - l.rolling(28).min())).values
    e8 = pd.Series(w28).ewm(span=8, adjust=False).mean().values
    ok = np.where(d > 0, w28 > e8, w28 < e8)
    return np.where(ok, d, 0)


def _fractal_break(b):
    """+1 = close breaks above the last confirmed up-fractal high, -1 = below the last down-fractal low (bt/frac_pa.py)."""
    h, l, c = b.h.values, b.l.values, b.c.values; n = len(b)
    upf = np.zeros(n, bool); dnf = np.zeros(n, bool)
    upf[2:-2] = (h[2:-2] > h[1:-3]) & (h[2:-2] > h[:-4]) & (h[2:-2] > h[3:-1]) & (h[2:-2] > h[4:])
    dnf[2:-2] = (l[2:-2] < l[1:-3]) & (l[2:-2] < l[:-4]) & (l[2:-2] < l[3:-1]) & (l[2:-2] < l[4:])
    uc = np.r_[False, False, upf[:-2]]; dc = np.r_[False, False, dnf[:-2]]
    ul = pd.Series(np.where(uc, np.roll(h, 2), np.nan)).ffill().shift(1).values
    dl = pd.Series(np.where(dc, np.roll(l, 2), np.nan)).ffill().shift(1).values
    pc = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid='ignore'):
        return np.where((c > ul) & (pc <= ul), 1, np.where((c < dl) & (pc >= dl), -1, 0))


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
        elif s['model'] == 'FRACTAL_FADE':   # trade against the breakout
            d0, ok = _fractal_fade(b, _bars(df, s['tf'] * 5))
            m = (d0 != 0) & ok & in_window(te, s)
            d = -d0
        elif s['model'] == 'STREAK_WILLY':
            d = _streak_willy(b)
            m = (d != 0) & in_window(te, s)
        else:
            P_, calm = _candles(b, s['tf'])
            if s['model'] == 'FRACTAL_STAR':      # star candle breaks the fractal level in the star's direction -> fade
                br = _fractal_break(b)
                m = (br != 0) & (P_['STAR'] == br) & in_window(te, s)
                d = -br
            elif s['model'] == 'MARUBOZU_CALM':   # fade
                m = (P_['MARUBOZU'] != 0) & calm & in_window(te, s)
                d = -P_['MARUBOZU']
            else:                                  # DOJI_CALM: reversal (classic direction)
                m = (P_['DOJI_EXT'] != 0) & calm & in_window(te, s)
                d = P_['DOJI_EXT']
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
