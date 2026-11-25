"""DAILY / WEEKLY / MONTHLY reports from data/signals.csv (plain text for Telegram), with statistics per strategy.
Schedule (Dubai time, UTC+4, no DST):
  DAILY   - per session (strategy.SESSIONS): morning (A2, B1-B4) at 08:30, evening (A1) at 19:30 Dubai (after the last expiry), Mon-Fri
  WEEKLY  - Saturday 15:00, Dubai dates Sunday..Saturday of that week
  MONTHLY - 1st of month 06:00, previous calendar month
A report missed because the bot was down is sent late (within 20 h); sent reports are remembered in data/reports_sent.json."""
import json, os
from datetime import datetime, timedelta, timezone
import pandas as pd
import config
from strategy import STRATEGIES, SESSIONS

DXB = timezone(timedelta(hours=4))
WD = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
MONTHS = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
SENT_FILE = os.path.join(config.DATA_DIR, 'reports_sent.json')


def _load_log():
    if not os.path.exists(config.SIGNALS_CSV):
        return pd.DataFrame(columns=['date_dubai', 'strategy', 'pair', 'direction', 'entry_dubai', 'expiry_min',
                                     'entry_price', 'expiry_price', 'pips', 'result'])
    d = pd.read_csv(config.SIGNALS_CSV, dtype=str)
    if 'pips' not in d.columns:
        d['pips'] = None
    d['strategy'] = d.strategy.replace({'MORNING': 'S1'})
    d['pips_f'] = pd.to_numeric(d.pips, errors='coerce')
    return d


def _fp(v):
    return '—' if v is None or v != v else f'{v:+.1f} п.'


def _wl(t):
    w = int((t.result == 'WIN').sum()); l = int((t.result == 'LOSS').sum()); ti = int((t.result == 'TIE').sum())
    nd = int((t.result == 'NO DATA').sum())
    wr = f'{w / (w + l) * 100:.0f}%' if w + l else '—'
    return w, l, ti, nd, wr


def _pips_txt(t):
    """Pips: currencies and gold are not comparable -> shown separately."""
    parts = []
    fx = t[t.pair != 'xauusd']; au = t[t.pair == 'xauusd']
    if len(fx): parts.append(f'валюты {_fp(fx.pips_f.sum())}')
    if len(au): parts.append(f'золото {_fp(au.pips_f.sum())}')
    return ' · '.join(parts) if parts else '—'


def _wr_icon(w, l):
    if not w + l: return '⚪'
    x = w / (w + l)
    return '🟢' if x >= 0.5405 else ('🟡' if x >= 0.5 else '🔴')


def _slabel(k):
    st = STRATEGIES.get(k)
    if not st: return k
    return f'{"🧪" if st["test"] else "🔔"} {k} {st["name"]}'


def _summary(t):
    w, l, ti, nd, wr = _wl(t)
    return [f'📌 Сделок: {len(t)}   ✅ {w}   ❌ {l}   ➖ {ti}' + (f'   ❔ {nd}' if nd else ''),
            f'{_wr_icon(w, l)} Винрейт: {wr}  (безубыток при выплате 85% — 54%)',
            f'📏 Пункты: {_pips_txt(t)}']


def _row(g):
    w, l, ti, nd, wr = _wl(g)
    return f'{len(g)} сд. · ✅ {w} ❌ {l} · {_wr_icon(w, l)} {wr} · {_pips_txt(g)}'


def _line(r, with_date):
    mark = {'WIN': '✅', 'LOSS': '❌', 'TIE': '➖'}.get(r.result, '❔')
    arrow = '⬆️' if r.direction == 'CALL' else '⬇️'
    d = ''
    if with_date:
        dt = datetime.strptime(r.date_dubai, '%Y-%m-%d'); d = f'{dt:%d.%m} {WD[dt.weekday()]} '
    return (f'{mark} {d}{r.entry_dubai} · {r.strategy} · {config.DISPLAY.get(r.pair, r.pair)} {arrow} · {r.expiry_min} мин · '
            f'{_fp(r.pips_f)}\n      {config.fmt_price(r.pair, r.entry_price)} → {config.fmt_price(r.pair, r.expiry_price)}')


def build(kind, t, title_line, ids, tag=''):
    head = {'DAILY': '📊 DAILY REPORT', 'WEEKLY': '🗓 WEEKLY REPORT', 'MONTHLY': '📆 MONTHLY REPORT'}[kind]
    tags = f'\n\n#report #{kind.lower()}' + (f' #{tag}' if tag else '')
    lines = [head, f'📅 {title_line}', '']
    if t.empty:
        lines += ['💤 Сделок не было.', '', '🧩 Стратегии: ' + ', '.join(ids)]
        return '\n'.join(lines) + tags
    lines += ['🏆 ИТОГО'] + _summary(t) + ['', '🧩 ПО СТРАТЕГИЯМ']
    t = t.sort_values(['date_dubai', 'entry_dubai', 'strategy'])
    for k in ids:
        g = t[t.strategy == k]
        lines.append(f'{_slabel(k)}\n      ' + (_row(g) if len(g) else 'сделок не было'))
    if kind in ('WEEKLY', 'MONTHLY'):
        lines += ['', '📆 ПО ДНЯМ' if kind == 'WEEKLY' else '📆 ПО НЕДЕЛЯМ']
        if kind == 'WEEKLY':
            for k, g in t.groupby('date_dubai', sort=True):
                dt = datetime.strptime(k, '%Y-%m-%d'); lines.append(f'▫️ {dt:%d.%m} {WD[dt.weekday()]}: {_row(g)}')
        else:
            wk = pd.to_datetime(t.date_dubai).dt.to_period('W-SAT').dt.start_time.dt.strftime('%d.%m')
            for k, g in t.assign(wk=wk.values).groupby('wk', sort=True):
                lines.append(f'▫️ неделя с {k}: {_row(g)}')
        lines += ['', '💱 ПО ПАРАМ']
        for k, g in t.groupby('pair', sort=True):
            lines.append(f'▫️ {config.DISPLAY.get(k, k)}: {_row(g)}')
    lines += ['', '📋 СДЕЛКИ (время Дубая, цена входа → цена экспирации)']
    lines += [_line(r, kind != 'DAILY') for r in t.itertuples()]
    return '\n'.join(lines) + tags


class Scheduler:
    def __init__(self, enabled):
        self.enabled = enabled
        self.silent = not os.path.exists(SENT_FILE)   # very first start: don't fire reports for periods before the bot existed
        try:
            self.sent = set(json.load(open(SENT_FILE)))
        except Exception:
            self.sent = set()

    def _mark(self, key):
        self.sent.add(key)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        json.dump(sorted(self.sent)[-500:], open(SENT_FILE, 'w'))

    def due(self, now_ms, pending_strategies=()):
        """Returns list of report texts to send now."""
        now = datetime.fromtimestamp(now_ms / 1000, DXB).replace(tzinfo=None)
        out = []
        # DAILY (per session, Mon-Fri)
        for sess, sd in SESSIONS.items():
            ids = [k for k in self.enabled if STRATEGIES[k]['session'] == sess]
            if not ids:
                continue
            h, m = divmod((sd['report_utc'] + 240) % 1440, 60)
            sched = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if sched > now: sched -= timedelta(days=1)
            key = f'DAILY:{sess}:{sched:%Y-%m-%d}'
            if sched.weekday() < 5 and key not in self.sent and now - sched < timedelta(hours=20) \
                    and not any(k in pending_strategies for k in ids):
                log = _load_log(); day = f'{sched:%Y-%m-%d}'
                t = log[(log.date_dubai == day) & log.strategy.isin(ids)]
                out.append(build('DAILY', t, f'{sd["name"]} · {sched:%d.%m.%Y}, {WD[sched.weekday()]} (Дубай)', ids, sess.lower()))
                self._mark(key)
        # WEEKLY Saturday 15:00
        sat = now - timedelta(days=(now.weekday() - 5) % 7)
        sched = sat.replace(hour=15, minute=0, second=0, microsecond=0)
        if sched > now: sched -= timedelta(days=7)
        key = f'WEEKLY:{sched:%Y-%m-%d}'
        if key not in self.sent and now - sched < timedelta(hours=20):
            start = (sched - timedelta(days=6)).date(); end = sched.date()
            log = _load_log()
            dd = pd.to_datetime(log.date_dubai).dt.date if len(log) else pd.Series(dtype=object)
            t = log[(dd >= start) & (dd <= end) & log.strategy.isin(self.enabled)] if len(log) else log
            out.append(build('WEEKLY', t, f'Неделя {start:%d.%m} — {end:%d.%m.%Y} (Дубай)', self.enabled))
            self._mark(key)
        # MONTHLY 1st 06:00
        sched = now.replace(day=1, hour=6, minute=0, second=0, microsecond=0)
        if sched > now: sched = (sched - timedelta(days=1)).replace(day=1)
        key = f'MONTHLY:{sched:%Y-%m}'
        if key not in self.sent and now - sched < timedelta(hours=20):
            prev = (sched - timedelta(days=1)).replace(day=1)
            log = _load_log()
            t = log[log.date_dubai.str.startswith(f'{prev:%Y-%m}') & log.strategy.isin(self.enabled)] if len(log) else log
            out.append(build('MONTHLY', t, f'{MONTHS[prev.month - 1].capitalize()} {prev.year} (Дубай)', self.enabled))
            self._mark(key)
        if self.silent:
            self.silent = False
            return []
        return out
