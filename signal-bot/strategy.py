"""Signal logic — identical to the backtest (same kernels, same parameters, same features).

All strategies trade AGAINST the detected move ("fade"). Filters are written relative to the ORIGINAL move direction d:
+1 = the filter points the same way as the move, -1 = opposite. Entry: manual ~1 min after candle close.
Window rule (Dubai time): entry (close + 1 min) and expiry must both lie inside the window.

S1  Утро 06–07   15m  LEVEL_BREAK  ATR14(15m) < 0.75 x 20-day avg                       expiry 10   (main strategy)
T1  Утро 05–06    5m  IFVG         H1 close vs EMA50 same way as move, ADX14(H1) >= 30    expiry 30   (test)
T2  Утро 07–08   15m  IFVG         H1 structure AND H1 EMA50 opposite to the move         expiry 10   (test)
T3  Утро 07–09    3m  IFVG         ATR14(3m) 1–1.5 x avg, ATR14(H1) >= 1.5 x avg           expiry 10   (test)
T4  Вечер 21–23   3m  LEVEL_BREAK  ATR14(3m) 0.5–0.75 x avg, ATR14(H1) 0.5–0.75 x avg      expiry 30   (test)
LEVEL_BREAK: candle closes beyond an untaken liquidity level for the first time.
IFVG: a displacement FVG (middle body >= 0.8 ATR14) is closed through (inversion).
"""
import numpy as np
import pandas as pd
from kernels import swings_asof, structure_dir, liquidity_events, fvg_list, ifvg_events

MIN = 60_000

# windows in UTC minutes of day (Dubai = UTC+4, no DST)
STRATEGIES = {
    'S1': dict(name='Утро 06–07', what='против пробоя уровня', model='LEVEL_BREAK', tf=15, start_utc=120, end_utc=180, expiry=10,
               test=False, session='MORNING', f=dict(atr_tf=(None, 0.75))),
    'T1': dict(name='Утро 05–06', what='против IFVG', model='IFVG', tf=5, start_utc=60, end_utc=120, expiry=30,
               test=True, session='MORNING', f=dict(ema50_h1=1, adx_h1=(30, None))),
    'T2': dict(name='Утро 07–08', what='против IFVG, по тренду H1', model='IFVG', tf=15, start_utc=180, end_utc=240, expiry=10,
               test=True, session='MORNING', f=dict(struct_h1=-1, ema50_h1=-1)),
    'T3': dict(name='Утро 07–09', what='против IFVG при высокой волатильности', model='IFVG', tf=3, start_utc=180, end_utc=300, expiry=10,
               test=True, session='MORNING', f=dict(atr_tf=(1.0, 1.5), atr_h1=(1.5, None))),
    'T5': dict(name='Утро 05–06', what='против пробоя уровня при спокойном рынке', model='LEVEL_BREAK', tf=5, start_utc=60, end_utc=120, expiry=15,
               test=True, session='MORNING', f=dict(atr_tf=(None, 0.75), adx_h1=(None, 20))),
    'T6': dict(name='Утро 05–07', what='против пробоя inside bar, по тренду H1', model='INSIDE_BAR_BREAK', tf=15, start_utc=60, end_utc=180, expiry=2,
               test=True, session='MORNING', f=dict(ema50_h1=1)),
    'T7': dict(name='Утро 05–09', what='разворот от доджи у экстремума при высокой волатильности', model='DOJI_EXTREME', tf=5, start_utc=60, end_utc=300,
               expiry=5, test=True, session='MORNING', fade=False, f=dict(atr_tf=(1.5, None))),
    'T4': dict(name='Вечер 21–23', what='против пробоя уровня', model='LEVEL_BREAK', tf=3, start_utc=1020, end_utc=1140, expiry=30,
               test=True, session='EVENING', f=dict(atr_tf=(0.5, 0.75), atr_h1=(0.5, 0.75))),
}
SESSIONS = {'MORNING': dict(name='Утро 05–09', report_utc=330), 'EVENING': dict(name='Вечер 21–23', report_utc=1170)}  # reports 09:30 / 23:30 Dubai


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
    b['atr_ratio'] = b.atr / b.atr.rolling(int(20 * 1440 / tf), min_periods=200).mean()
    return b


def _levels(df, day, tf):
    b = _bars(df, tf)
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
    ev = liquidity_events(O, H, L, C, A, hiLv, loLv, 2)
    return b, (O, H, L, C, A), ev


def _inwin(te, s):
    um = (te // MIN) % 1440
    return (um >= s['start_utc']) & (um < s['end_utc']) & (um + s['expiry'] <= s['end_utc'])


def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=h.index).ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=h.index).ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=a, adjust=False).mean()


def _h1_features(df):
    """H1 features exactly as in the backtest. Unfinished last hour is dropped (live history ends mid-hour)."""
    now = int(df.timestamp.iloc[-1]) + MIN
    # (a) UTC-hour candles: EMA50, ADX14, ATR ratio
    g = df.groupby(df.timestamp.values // 3_600_000, sort=True)
    a = pd.DataFrame({'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last(), 'tc': g.timestamp.max() + MIN})
    a = a[(a.index.values + 1) * 3_600_000 <= now].reset_index(drop=True)
    a['e50'] = a.c.ewm(span=50, adjust=False).mean()
    a['adx'] = _adx(a.h, a.l, a.c)
    tr = np.maximum(a.h - a.l, np.maximum((a.h - a.c.shift()).abs(), (a.l - a.c.shift()).abs()))
    at = tr.rolling(14).mean(); a['atr_r'] = at / at.rolling(480, min_periods=100).mean()
    # (b) market structure on NY-hour candles (as in the backtest)
    g2 = df.groupby(df.ny.dt.floor('h'), sort=True)
    s = pd.DataFrame({'h': g2.high.max(), 'l': g2.low.min(), 'c': g2.close.last(), 'tc': g2.timestamp.max() + MIN,
                      'last': g2.timestamp.max()}).reset_index(drop=True)
    s = s.iloc[:-1] if len(s) and (int(s['last'].iloc[-1]) // 3_600_000 + 1) * 3_600_000 > now else s
    s = s.reset_index(drop=True)
    sh, shi, sl, sli = swings_asof(s.h.values, s.l.values, 2)
    s['st'] = structure_dir(s.c.values, sh, sl)
    return a, s


def _asof(tc, vals, t):
    i = np.searchsorted(tc, t, side='right') - 1
    return np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan)


def _rng(x, lo, hi):
    ok = ~np.isnan(x)
    if lo is not None: ok &= x >= lo
    if hi is not None: ok &= x < hi
    return ok


def scan(m1: pd.DataFrame, pair: str, only=None) -> pd.DataFrame:
    """m1: columns timestamp (ms UTC, candle open), open, high, low, close (finished candles only).
    Returns every signal found in the given history: pair, strategy, t_close, t_entry (= t_close + 1 min),
    direction (+1 CALL / -1 PUT, already the faded/trade direction), expiry, price (candle close), lvl."""
    df = _with_ny(m1)
    if len(df) < 3000:
        return pd.DataFrame()
    only = only or list(STRATEGIES)
    day = _context(df)
    h1a, h1s = _h1_features(df)
    cache = {}
    out = []
    for sid in only:
        s = STRATEGIES[sid]; tf = s['tf']
        if tf not in cache:
            cache[tf] = _levels(df, day, tf)
        b, (O, H, L, C, A), (uk, ut, ul, dk, dt_, dlv) = cache[tf]
        tc = b.tc.values.astype(np.int64)
        if s['model'] == 'LEVEL_BREAK':
            ju = np.where(uk == 2)[0]; jd = np.where(dk == 2)[0]
            j = np.concatenate([ju, jd]); d = np.concatenate([np.ones(len(ju)), -np.ones(len(jd))])
            lt = np.concatenate([ut[ju], dt_[jd]])
        elif s['model'] == 'IFVG':
            fv = fvg_list(O, H, L, C, A, 0.8)
            iv = ifvg_events(C, fv, len(C), 100) if len(fv) else np.zeros((0, 5))
            j = iv[:, 0].astype(np.int64); d = iv[:, 1]; lt = np.zeros(len(j))
        else:   # candle patterns
            o_, h_, l_, c_ = (pd.Series(x) for x in (O, H, L, C))
            if s['model'] == 'INSIDE_BAR_BREAK':
                inside = (h_.shift() < h_.shift(2)) & (l_.shift() > l_.shift(2))
                lg = inside & (c_ > h_.shift(2)); sh = inside & (c_ < l_.shift(2))
            else:   # DOJI_EXTREME
                rng = h_ - l_; doji = ((c_ - o_).abs() <= 0.1 * rng) & (rng > 0)
                lg = doji & (l_ <= l_.rolling(10).min()); sh = doji & (h_ >= h_.rolling(10).max())
            ju = np.where(lg.fillna(False).values)[0]; jd = np.where(sh.fillna(False).values)[0]
            j = np.concatenate([ju, jd]); d = np.concatenate([np.ones(len(ju)), -np.ones(len(jd))]); lt = np.zeros(len(j))
        if not len(j):
            continue
        te = tc[j] + MIN
        m = _inwin(te, s)
        f = s['f']
        if 'atr_tf' in f: m &= _rng(b.atr_ratio.values[j], *f['atr_tf'])
        if 'atr_h1' in f: m &= _rng(_asof(h1a.tc.values, h1a.atr_r.values, te), *f['atr_h1'])
        if 'adx_h1' in f: m &= _rng(_asof(h1a.tc.values, h1a.adx.values, te), *f['adx_h1'])
        if 'ema50_h1' in f:
            i = np.searchsorted(h1a.tc.values, te, side='right') - 1
            e = np.where(i >= 0, np.sign(h1a.c.values[np.clip(i, 0, None)] - h1a.e50.values[np.clip(i, 0, None)]) * d, 0)
            m &= e == f['ema50_h1']
        if 'struct_h1' in f:
            st = np.nan_to_num(_asof(h1s.tc.values, h1s.st.values.astype(float), te)) * d
            m &= st == f['struct_h1']
        if m.any():
            out.append(pd.DataFrame({'pair': pair, 'strategy': sid, 't_close': tc[j[m]], 't_entry': te[m],
                                     'direction': ((-d[m]) if s.get('fade', True) else d[m]).astype(int), 'expiry': s['expiry'], 'price': C[j[m]], 'lvl': lt[m]}))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values(['t_close', 'strategy']).reset_index(drop=True)
