"""DAILY / WEEKLY / MONTHLY reports from data/signals.csv (plain text for Telegram).
Schedule (Dubai time, UTC+4, no DST):
  DAILY   - per strategy, at its report time (morning: 07:30), Mon-Fri
  WEEKLY  - Saturday 15:00, Dubai dates Sunday..Saturday of that week
  MONTHLY - 1st of month 06:00, previous calendar month
A report missed because the bot was down is sent late (within 20 h); sent reports are remembered in data/reports_sent.json."""
import json, os
from datetime import datetime, timedelta, timezone
import pandas as pd
import config

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
    d['pips_f'] = pd.to_numeric(d.pips, errors='coerce')
    return d


def _fp(v):
    return '- п.' if v is None or v != v else f'{v:+.1f} п.'


def _summary(t):
    w = int((t.result == 'WIN').sum()); l = int((t.result == 'LOSS').sum())
    ti = int((t.result == 'TIE').sum()); nd = int((t.result == 'NO DATA').sum())
    wr = f'{w / (w + l) * 100:.0f}%' if w + l else '-'
    lines = [f'Сделок: {len(t)} (WIN {w} / LOSS {l} / TIE {ti}' + (f' / нет данных {nd}' if nd else '') + ')',
             f'Винрейт: {wr}']
    # pips of gold (0.1) and currencies are not comparable -> summed separately
    for nm, part in [('валюты', t[t.pair != 'xauusd']), ('золото', t[t.pair == 'xauusd'])]:
        if len(part):
            pw = part.pips_f[part.result == 'WIN'].sum(); pl = part.pips_f[part.result == 'LOSS'].sum()
            lines.append(f'Пункты, {nm}: {_fp(part.pips_f.sum())} (по WIN {_fp(pw)}, по LOSS {_fp(pl)})')
    return lines, w, l


def _line(r, with_date):
    d = ''
    if with_date:
        dt = datetime.strptime(r.date_dubai, '%Y-%m-%d')
        d = f'{dt:%d.%m} {WD[dt.weekday()]} '
    return (f'{d}{r.entry_dubai} {config.DISPLAY.get(r.pair, r.pair)} {r.direction} {r.expiry_min} мин — '
            f'{r.result} {_fp(r.pips_f)} ({config.fmt_price(r.pair, r.entry_price)} → {config.fmt_price(r.pair, r.expiry_price)})')


def _group(t, key, label):
    out = []
    for k, g in t.groupby(key, sort=True):
        w = int((g.result == 'WIN').sum()); l = int((g.result == 'LOSS').sum())
        wr = f'{w / (w + l) * 100:.0f}%' if w + l else '-'
        fx = g[g.pair != 'xauusd'].pips_f.sum(); au = g[g.pair == 'xauusd']
        pp = _fp(fx) if (g.pair != 'xauusd').any() else ''
        if len(au):
            pp = (pp + ', ' if pp else '') + f'золото {_fp(au.pips_f.sum())}'
        out.append(f'{label(k)}: {len(g)} сд., {w}/{l}, {wr}, {pp}')
    return out


def build(kind, t, title_line, strategy_names):
    lines = [f'{kind} REPORT', title_line, 'Стратегия: ' + ', '.join(strategy_names), '']
    if t.empty:
        lines.append('Сделок не было.')
        return '\n'.join(lines)
    s, _, _ = _summary(t)
    lines += s
    t = t.sort_values(['date_dubai', 'entry_dubai'])
    if kind in ('WEEKLY', 'MONTHLY'):
        lines += ['', 'По дням:' if kind == 'WEEKLY' else 'По неделям:']
        if kind == 'WEEKLY':
            lines += _group(t, 'date_dubai', lambda k: f'{datetime.strptime(k, "%Y-%m-%d"):%d.%m} {WD[datetime.strptime(k, "%Y-%m-%d").weekday()]}')
        else:
            wk = pd.to_datetime(t.date_dubai).dt.to_period('W-SAT').dt.start_time.dt.strftime('%d.%m')
            lines += _group(t.assign(wk=wk.values), 'wk', lambda k: f'неделя с {k}')
        lines += ['', 'По парам:'] + _group(t, 'pair', lambda k: config.DISPLAY.get(k, k))
    lines += ['', 'Сделки (время Дубая, цена входа → цена экспирации):']
    lines += [_line(r, kind != 'DAILY') for r in t.itertuples()]
    return '\n'.join(lines)


class Scheduler:
    def __init__(self, strategies, enabled):
        self.st = strategies; self.enabled = enabled
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
        names = [self.st[k]['name'] for k in self.enabled]
        # DAILY (per strategy, Mon-Fri)
        for k in self.enabled:
            ru = self.st[k]['report_utc']; h, m = divmod((ru + 240) % 1440, 60)
            sched = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if sched > now: sched -= timedelta(days=1)
            key = f'DAILY:{k}:{sched:%Y-%m-%d}'
            if sched.weekday() < 5 and key not in self.sent and now - sched < timedelta(hours=20) and k not in pending_strategies:
                log = _load_log(); day = f'{sched:%Y-%m-%d}'
                t = log[(log.date_dubai == day) & (log.strategy == k)]
                out.append(build('DAILY', t, f'{sched:%d.%m.%Y} {WD[sched.weekday()]} (Дубай)', [self.st[k]['name']]))
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
            out.append(build('WEEKLY', t, f'Неделя {start:%d.%m} — {end:%d.%m.%Y} (Дубай)', names))
            self._mark(key)
        # MONTHLY 1st 06:00
        sched = now.replace(day=1, hour=6, minute=0, second=0, microsecond=0)
        if sched > now: sched = (sched - timedelta(days=1)).replace(day=1)
        key = f'MONTHLY:{sched:%Y-%m}'
        if key not in self.sent and now - sched < timedelta(hours=20):
            prev = (sched - timedelta(days=1)).replace(day=1)
            log = _load_log()
            t = log[log.date_dubai.str.startswith(f'{prev:%Y-%m}') & log.strategy.isin(self.enabled)] if len(log) else log
            out.append(build('MONTHLY', t, f'{MONTHS[prev.month - 1].capitalize()} {prev.year} (Дубай)', names))
            self._mark(key)
        if self.silent:
            self.silent = False
            return []
        return out
