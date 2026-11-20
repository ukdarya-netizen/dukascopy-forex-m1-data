"""Numba kernels for ICT detection. All detection uses only data known at bar close (no lookahead).
Type codes for liquidity levels: 1 SW (swing), 2 EQ (equal highs/lows), 3 ASIA, 4 LON (London KZ range), 5 PD (prev day), 6 PW (prev week)
"""
import numpy as np
from numba import njit

@njit(cache=True)
def swings_asof(H, L, k):
    """Fractal pivots k left / k right. Pivot at i is CONFIRMED at bar i+k (known at its close).
    Returns, for every bar j, the last confirmed swing high/low price and pivot index as of close of j."""
    n = len(H)
    sh = np.full(n, np.nan); shi = np.full(n, -1)
    sl = np.full(n, np.nan); sli = np.full(n, -1)
    csh = np.nan; cshi = -1; csl = np.nan; csli = -1
    for j in range(n):
        i = j - k
        if i >= k:
            ok = True
            for q in range(i - k, i + k + 1):
                if (q < i and H[q] >= H[i]) or (q > i and H[q] > H[i]):
                    ok = False; break
            if ok:
                csh = H[i]; cshi = i
            ok = True
            for q in range(i - k, i + k + 1):
                if q < i and L[q] <= L[i]:
                    ok = False; break
                if q > i and L[q] < L[i]:
                    ok = False; break
            if ok:
                csl = L[i]; csli = i
        sh[j] = csh; shi[j] = cshi; sl[j] = csl; sli[j] = csli
    return sh, shi, sl, sli

@njit(cache=True)
def structure_dir(C, sh, sl):
    """Market structure as of each bar close: +1 after a close above the last confirmed swing high, -1 after close below last swing low."""
    n = len(C); out = np.zeros(n, np.int8); st = 0
    for j in range(1, n):
        if not np.isnan(sh[j-1]) and C[j] > sh[j-1]:
            st = 1
        elif not np.isnan(sl[j-1]) and C[j] < sl[j-1]:
            st = -1
        out[j] = st
    return out

@njit(cache=True)
def liquidity_events(O, H, L, C, atr, hiLv, loLv, k):
    """hiLv/loLv: (n,4) arrays with HTF levels valid at bar j, columns = ASIA, LON, PD, PW (type 3..6), NaN if not valid.
    Internal swing liquidity (type 1) and equal highs/lows (type 2) are tracked here from confirmed fractal pivots.
    For every bar returns the first-time take of the most significant level on each side:
      kind 1 = sweep (wick through, close back), 2 = break (close beyond)."""
    n = len(H)
    up_kind = np.zeros(n, np.int8); up_type = np.zeros(n, np.int8); up_lv = np.full(n, np.nan)
    dn_kind = np.zeros(n, np.int8); dn_type = np.zeros(n, np.int8); dn_lv = np.full(n, np.nan)
    CAP = 64; MAXAGE = 500
    shp = np.full(CAP, np.nan); shi = np.full(CAP, -1); sheq = np.zeros(CAP, np.int8)
    slp = np.full(CAP, np.nan); sli = np.full(CAP, -1); sleq = np.zeros(CAP, np.int8)
    lastH = np.full(4, np.nan); takenH = np.zeros(4, np.int8)
    lastL = np.full(4, np.nan); takenL = np.zeros(4, np.int8)
    for j in range(n):
        bt = 0; blv = np.nan
        # HTF high-side
        for c in range(4):
            v = hiLv[j, c]
            if v != lastH[c] and not (np.isnan(v) and np.isnan(lastH[c])):
                lastH[c] = v; takenH[c] = 0
            if not np.isnan(v) and takenH[c] == 0 and H[j] > v:
                takenH[c] = 1
                t = c + 3
                if t > bt:
                    bt = t; blv = v
        # swings high-side
        for s in range(CAP):
            if shi[s] >= 0:
                if j - shi[s] > MAXAGE:
                    shi[s] = -1; continue
                if H[j] > shp[s]:
                    t = 2 if sheq[s] == 1 else 1
                    if t > bt or (t == bt and t <= 2 and shp[s] > blv):
                        bt = t; blv = shp[s]
                    shi[s] = -1
        if bt > 0:
            up_type[j] = bt; up_lv[j] = blv
            up_kind[j] = 1 if C[j] <= blv else 2
        bt = 0; blv = np.nan
        for c in range(4):
            v = loLv[j, c]
            if v != lastL[c] and not (np.isnan(v) and np.isnan(lastL[c])):
                lastL[c] = v; takenL[c] = 0
            if not np.isnan(v) and takenL[c] == 0 and L[j] < v:
                takenL[c] = 1
                t = c + 3
                if t > bt:
                    bt = t; blv = v
        for s in range(CAP):
            if sli[s] >= 0:
                if j - sli[s] > MAXAGE:
                    sli[s] = -1; continue
                if L[j] < slp[s]:
                    t = 2 if sleq[s] == 1 else 1
                    if t > bt or (t == bt and t <= 2 and slp[s] < blv):
                        bt = t; blv = slp[s]
                    sli[s] = -1
        if bt > 0:
            dn_type[j] = bt; dn_lv[j] = blv
            dn_kind[j] = 1 if C[j] >= blv else 2
        # outside bar taking both sides: ambiguous, drop both
        if up_kind[j] > 0 and dn_kind[j] > 0:
            up_kind[j] = 0; dn_kind[j] = 0; up_type[j] = 0; dn_type[j] = 0
        # confirm pivot i = j-k at close of j (added after takes, so it cannot be "taken" by its own bars)
        i = j - k
        if i >= k:
            ok = True
            for q in range(i - k, i + k + 1):
                if (q < i and H[q] >= H[i]) or (q > i and H[q] > H[i]):
                    ok = False; break
            if ok:
                # still untaken by bars i+1..j by construction (H[q] <= H[i])
                tol = 0.1 * atr[j] if not np.isnan(atr[j]) else 0.0
                free = -1
                for s in range(CAP):
                    if shi[s] >= 0 and abs(shp[s] - H[i]) <= tol:
                        sheq[s] = 1
                        if H[i] > shp[s]:
                            shp[s] = H[i]
                        free = -2; break
                if free != -2:
                    oldest = 0; oi = 1 << 60
                    for s in range(CAP):
                        if shi[s] < 0:
                            oldest = s; break
                        if shi[s] < oi:
                            oi = shi[s]; oldest = s
                    shp[oldest] = H[i]; shi[oldest] = i; sheq[oldest] = 0
            ok = True
            for q in range(i - k, i + k + 1):
                if (q < i and L[q] <= L[i]) or (q > i and L[q] < L[i]):
                    ok = False; break
            if ok:
                tol = 0.1 * atr[j] if not np.isnan(atr[j]) else 0.0
                free = -1
                for s in range(CAP):
                    if sli[s] >= 0 and abs(slp[s] - L[i]) <= tol:
                        sleq[s] = 1
                        if L[i] < slp[s]:
                            slp[s] = L[i]
                        free = -2; break
                if free != -2:
                    oldest = 0; oi = 1 << 60
                    for s in range(CAP):
                        if sli[s] < 0:
                            oldest = s; break
                        if sli[s] < oi:
                            oi = sli[s]; oldest = s
                    slp[oldest] = L[i]; sli[oldest] = i; sleq[oldest] = 0
    return up_kind, up_type, up_lv, dn_kind, dn_type, dn_lv

@njit(cache=True)
def sweep_mss_chains(O, H, L, C, atr, up_kind, up_type, up_lv, dn_kind, dn_type, dn_lv,
                     sh, shi, sl, sli, maxbars):
    """After a sweep at bar j, look for MSS within maxbars: close beyond the last confirmed opposite swing (as of j).
    Invalidated if price closes back beyond the swept level before MSS.
    Returns arrays per chain: j, m, dir, type, X (sweep extreme), E (leg extreme at m), disp (max body/atr in leg),
    fvg_lo, fvg_hi (latest FVG inside leg, NaN if none), ob_trig, brk_trig."""
    n = len(H)
    out = np.full((n // 2 + 10, 11), np.nan)
    cnt = 0
    for j in range(1, n):
        for side in range(2):
            if side == 0:
                if up_kind[j] != 1: continue
                d = -1; lv = up_lv[j]; tp = up_type[j]; ref = sl[j]; refi = sli[j]
            else:
                if dn_kind[j] != 1: continue
                d = 1; lv = dn_lv[j]; tp = dn_type[j]; ref = sh[j]; refi = shi[j]
            if np.isnan(ref) or refi < 0: continue
            X = H[j] if d == -1 else L[j]
            disp = 0.0
            m = -1
            for q in range(j + 1, min(n, j + 1 + maxbars)):
                if d == -1:
                    if H[q] > X: X = H[q]
                    if C[q] > lv: break
                    b = (O[q] - C[q]) / atr[q] if atr[q] > 0 else 0.0
                    if b > disp: disp = b
                    if C[q] < ref:
                        m = q; break
                else:
                    if L[q] < X: X = L[q]
                    if C[q] < lv: break
                    b = (C[q] - O[q]) / atr[q] if atr[q] > 0 else 0.0
                    if b > disp: disp = b
                    if C[q] > ref:
                        m = q; break
            if m < 0: continue
            # leg extreme
            E = L[j] if d == -1 else H[j]
            for q in range(j, m + 1):
                if d == -1 and L[q] < E: E = L[q]
                if d == 1 and H[q] > E: E = H[q]
            # latest FVG in leg (k from j+2..m), gap between k-2 and k
            flo = np.nan; fhi = np.nan
            for q in range(m, j + 1, -1):
                if q - 2 < 0: break
                if d == -1 and L[q-2] > H[q]:
                    flo = H[q]; fhi = L[q-2]; break
                if d == 1 and H[q-2] < L[q]:
                    flo = H[q-2]; fhi = L[q]; break
            # OB: last opposite candle at/before sweep bar (scan back 10)
            obt = np.nan
            for q in range(j, max(-1, j - 10), -1):
                if d == -1 and C[q] > O[q]:
                    obt = O[q]; break
                if d == 1 and C[q] < O[q]:
                    obt = O[q]; break
            # Breaker: the candle that formed the broken swing (last same-type candle at/before ref pivot)
            bkt = np.nan
            for q in range(refi, max(-1, refi - 5), -1):
                if d == -1 and C[q] < O[q]:
                    bkt = 0.5 * (O[q] + C[q]); break
                if d == 1 and C[q] > O[q]:
                    bkt = 0.5 * (O[q] + C[q]); break
            out[cnt, 0] = j; out[cnt, 1] = m; out[cnt, 2] = d; out[cnt, 3] = tp
            out[cnt, 4] = X; out[cnt, 5] = E; out[cnt, 6] = disp
            out[cnt, 7] = flo; out[cnt, 8] = fhi; out[cnt, 9] = obt; out[cnt, 10] = bkt
            cnt += 1
            if cnt >= out.shape[0]: return out[:cnt]
    return out[:cnt]

@njit(cache=True)
def fvg_list(O, H, L, C, atr, mindisp):
    """All displacement FVGs: gap between bars k-2 and k, middle bar body >= mindisp*ATR. Known at close of k.
    rows: k, dir, lo, hi, dispratio"""
    n = len(H)
    out = np.full((n, 5), np.nan); cnt = 0
    for k in range(2, n):
        if not (atr[k] > 0): continue
        body = abs(C[k-1] - O[k-1]) / atr[k]
        if body < mindisp: continue
        if L[k] > H[k-2] and C[k-1] > O[k-1]:
            out[cnt,0]=k; out[cnt,1]=1.0; out[cnt,2]=H[k-2]; out[cnt,3]=L[k]; out[cnt,4]=body; cnt += 1
        elif H[k] < L[k-2] and C[k-1] < O[k-1]:
            out[cnt,0]=k; out[cnt,1]=-1.0; out[cnt,2]=H[k]; out[cnt,3]=L[k-2]; out[cnt,4]=body; cnt += 1
    return out[:cnt]

@njit(cache=True)
def ifvg_events(C, fv, n, maxage):
    """Inversion: a bullish FVG closed below its bottom -> bearish signal (and mirror). One inversion per FVG.
    rows: j (inversion bar), dir, lo, hi, k (fvg bar)"""
    out = np.full((len(fv), 5), np.nan); cnt = 0
    for r in range(len(fv)):
        k = int(fv[r, 0]); d = fv[r, 1]; lo = fv[r, 2]; hi = fv[r, 3]
        for j in range(k + 1, min(n, k + 1 + maxage)):
            if d == 1 and C[j] < lo:
                out[cnt,0]=j; out[cnt,1]=-1.0; out[cnt,2]=lo; out[cnt,3]=hi; out[cnt,4]=k; cnt += 1; break
            if d == -1 and C[j] > hi:
                out[cnt,0]=j; out[cnt,1]=1.0; out[cnt,2]=lo; out[cnt,3]=hi; out[cnt,4]=k; cnt += 1; break
    return out[:cnt]

@njit(cache=True)
def retrace_entry(m1_ct, m1_H, m1_L, start_t, end_t, d, trig, cancel):
    """For each request find first M1 bar (close time in (start_t, end_t]) where price retraces to trig
    (d=-1: high>=trig, d=+1: low<=trig). Cancel if price breaches cancel first (or in the same bar).
    Returns M1 index of entry bar (entry at its close) or -1."""
    R = len(start_t); res = np.full(R, -1, np.int64)
    for r in range(R):
        if np.isnan(trig[r]): continue
        i = np.searchsorted(m1_ct, start_t[r], side='right')
        while i < len(m1_ct) and m1_ct[i] <= end_t[r]:
            if d[r] == -1:
                if not np.isnan(cancel[r]) and m1_H[i] > cancel[r]: break
                if m1_H[i] >= trig[r]:
                    res[r] = i; break
            else:
                if not np.isnan(cancel[r]) and m1_L[i] < cancel[r]: break
                if m1_L[i] <= trig[r]:
                    res[r] = i; break
            i += 1
    return res
