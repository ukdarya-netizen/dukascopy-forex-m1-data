"""Signal logic — identical to the backtest (same kernels, same parameters).

MORNING  Dubai 05:00-06:00: 5m candle closes beyond an untaken liquidity level for the first time (LEVEL_BREAK),
         H1 market structure points the same way as the break -> trade AGAINST the break. Expiry 15 min.
DAY      Dubai 14:00-15:00: sweep of a liquidity level -> MSS within 10 candles with a displacement candle
         (body >= 1 ATR14) -> trade AGAINST the MSS (towards the swept side). Expiry 5 min.
Entry: manual, within ~1 minute after the 5m candle close (backtest assumed entry 1 minute after close).
Window rule: entry (close + 1 min) and expiry must both lie inside the window.
"""
import numpy as np
import pandas as pd
from kernels import swings_asof, structure_dir, liquidity_events, sweep_mss_chains

MIN = 60_000
TF = 5
MSS_BARS = 10

STRATEGIES = {
    'MORNING': dict(name='Утро 05–06', start_utc=60, end_utc=120, expiry=15),   # Dubai = UTC+4
    'DAY':     dict(name='День 14–15', start_utc=600, end_utc=660, expiry=5),
}


def _with_ny(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    m1['ny'] = pd.to_datetime(m1.timestamp, unit='ms', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    return m1


def _context(df):
    """Daily/weekly/session levels keyed by trading day (day starts 17:00 NY)."""
    ny = df.ny
    tday = (ny + pd.Timedelta(hours=7)).dt.floor('D')
    hr = ny.dt.hour
    g = df.groupby(tday)
    day = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
    day['pdh'] = day.h.shift(1); day['pdl'] = day.l.shift(1)
    wk = day.index - pd.to_timedelta(day.index.weekday, unit='D')
    gw = day.groupby(wk)
    wkdf = pd.DataFrame({'h': gw.h.max(), 'l': gw.l.min()})
    wkdf['pwh'] = wkdf.h.shift(1); wkdf['pwl'] = wkdf.l.shift(1)
    day['pwh'] = wkdf.pwh.reindex(wk).values; day['pwl'] = wkdf.pwl.reindex(wk).values
    a = hr >= 20
    asia = df[a].groupby(tday[a])
    day['asia_h'] = asia.high.max(); day['asia_l'] = asia.low.min()
    lm = (hr >= 2) & (hr < 5)
    lon = df[lm].groupby(tday[lm])
    day['lon_h'] = lon.high.max(); day['lon_l'] = lon.low.min()
    return day


def _bars(df, tf):
    key = df.timestamp.values // (tf * MIN)
    g = df.groupby(key, sort=True)
    b = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
    b['t0'] = b.index.values * tf * MIN
    b['tc'] = b.t0 + tf * MIN
    b = b.reset_index(drop=True)
    tr = np.maximum(b.h - b.l, np.maximum((b.h - b.c.shift()).abs(), (b.l - b.c.shift()).abs()))
    b['atr'] = tr.rolling(14, min_periods=14).mean()
    return b


def _h1(df):
    key = df.timestamp.values // 3_600_000          # = NY hour (NY offset is whole hours)
    g = df.groupby(key, sort=True)
    b = pd.DataFrame({'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last(),
                      'tc': g.timestamp.max() + MIN})
    # live history ends mid-hour: an unfinished H1 candle must not count as closed
    now = int(df.timestamp.iloc[-1]) + MIN
    b = b[(b.index.values + 1) * 3_600_000 <= now].reset_index(drop=True)
    sh, shi, sl, sli = swings_asof(b.h.values, b.l.values, 2)
    b['st'] = structure_dir(b.c.values, sh, sl)
    return b


def _asof(tc, vals, t):
    i = np.searchsorted(tc, t, side='right') - 1
    return np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan)


def scan(m1: pd.DataFrame, pair: str) -> pd.DataFrame:
    """m1: columns timestamp (ms UTC, candle open), open, high, low, close.
    Returns every signal of both strategies found in the given history:
    pair, strategy, t_close (5m candle close, ms), t_entry (= t_close + 1 min), direction (+1 CALL / -1 PUT),
    expiry, price (5m close), level type."""
    df = _with_ny(m1)
    if len(df) < 3000:
        return pd.DataFrame()
    day = _context(df)
    b = _bars(df, TF)
    O, H, L, C, A = (b[c].values.astype(np.float64) for c in ['o', 'h', 'l', 'c', 'atr'])
    n = len(b)
    ny0 = pd.to_datetime(b.t0, unit='ms', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    tday = (ny0 + pd.Timedelta(hours=7)).dt.floor('D')
    hr = ny0.dt.hour.values
    dl = day.reindex(tday.values)
    intraday = hr < 17
    hiLv = np.full((n, 4), np.nan); loLv = np.full((n, 4), np.nan)
    hiLv[:, 0] = np.where(intraday, dl.asia_h.values, np.nan); loLv[:, 0] = np.where(intraday, dl.asia_l.values, np.nan)
    lonv = intraday & (hr >= 5)
    hiLv[:, 1] = np.where(lonv, dl.lon_h.values, np.nan); loLv[:, 1] = np.where(lonv, dl.lon_l.values, np.nan)
    hiLv[:, 2] = dl.pdh.values; loLv[:, 2] = dl.pdl.values
    hiLv[:, 3] = dl.pwh.values; loLv[:, 3] = dl.pwl.values
    uk, ut, ul, dk, dt_, dlv = liquidity_events(O, H, L, C, A, hiLv, loLv, 2)
    sh, shi, sl, sli = swings_asof(H, L, 2)
    tc = b.tc.values.astype(np.int64)
    h1 = _h1(df)
    out = []

    def inwin(te, s):
        um = (te // MIN) % 1440
        return (um >= s['start_utc']) & (um + s['expiry'] <= s['end_utc'])

    # MORNING: fade LEVEL_BREAK, H1 structure agrees with the break
    s = STRATEGIES['MORNING']
    ju = np.where(uk == 2)[0]; jd = np.where(dk == 2)[0]
    j = np.concatenate([ju, jd]); d = np.concatenate([np.ones(len(ju)), -np.ones(len(jd))])
    lt = np.concatenate([ut[ju], dt_[jd]])
    te = tc[j] + MIN
    h1s = np.nan_to_num(_asof(h1.tc.values, h1.st.values.astype(float), te))
    m = (h1s * d == 1) & inwin(te, s)
    if m.any():
        out.append(pd.DataFrame({'pair': pair, 'strategy': 'MORNING', 't_close': tc[j[m]], 't_entry': te[m],
                                 'direction': (-d[m]).astype(int), 'expiry': s['expiry'], 'price': C[j[m]], 'lvl': lt[m]}))
    # DAY: fade SWEEP -> MSS with displacement >= 1 ATR
    s = STRATEGIES['DAY']
    ch = sweep_mss_chains(O, H, L, C, A, uk, ut, ul, dk, dt_, dlv, sh, shi, sl, sli, MSS_BARS)
    if len(ch):
        chd = pd.DataFrame(ch[:, :7], columns=['j', 'm', 'd', 'tp', 'X', 'E', 'disp'])
        chd = chd.sort_values(['m', 'd', 'tp']).drop_duplicates(['m', 'd'], keep='last')
        mm = chd.m.values.astype(np.int64); d = chd.d.values
        te = tc[mm] + MIN
        k = (chd.disp.values >= 1) & inwin(te, s)
        if k.any():
            out.append(pd.DataFrame({'pair': pair, 'strategy': 'DAY', 't_close': tc[mm[k]], 't_entry': te[k],
                                     'direction': (-d[k]).astype(int), 'expiry': s['expiry'], 'price': C[mm[k]],
                                     'lvl': chd.tp.values[k]}))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values('t_close').reset_index(drop=True)
