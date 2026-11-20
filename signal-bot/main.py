"""Telegram signal bot for manual binary-options trading (Pocket Option).
Run:  python main.py           -> live bot
      python main.py --check   -> setup check: Dukascopy feed + Telegram + signals of the last 5 days found on history (nothing is sent except one test message)
"""
import os, sys, time, csv, traceback
from datetime import datetime, timezone, timedelta
import pandas as pd
import config, data_feed, telegram
from strategy import scan, STRATEGIES

MIN = 60_000
DXB = timezone(timedelta(hours=4))
LVL = {1: 'фрактал', 2: 'равные хаи/лои', 3: 'Азия', 4: 'Лондон', 5: 'прошлый день', 6: 'прошлая неделя'}


def dxb(ms, fmt='%H:%M'):
    return datetime.fromtimestamp(ms / 1000, DXB).strftime(fmt)


def signal_text(s):
    st = STRATEGIES[s.strategy]
    arrow = 'CALL ⬆️ (вверх)' if s.direction == 1 else 'PUT ⬇️ (вниз)'
    return (f'{config.DISPLAY[s.pair]}  {arrow}\n'
            f'Экспирация: {st["expiry"]} мин\n'
            f'Входить сейчас, ориентир до {dxb(s.t_entry)} (Дубай)\n'
            f'Стратегия: {st["name"]}\n'
            f'Свеча 5m закрылась {dxb(s.t_close)}, цена {round(float(s.price), 5)}, уровень: {LVL.get(int(s.lvl), "-")}')


def in_any_window(t_close):
    te = t_close + MIN
    um = (te // MIN) % 1440
    return any(st['start_utc'] <= um and um + st['expiry'] <= st['end_utc']
               for k, st in STRATEGIES.items() if k in config.ENABLED)


def save(row):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    new = not os.path.exists(config.SIGNALS_CSV)
    with open(config.SIGNALS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)


class Bot:
    def __init__(self):
        self.hist = {}
        self.pending = []       # signals waiting for result
        self.sent = set()
        self.window_res = {}    # (date, strategy) -> [wins, losses, ties]

    def load(self):
        for p in config.PAIRS:
            self.hist[p] = data_feed.history(p, config.HISTORY_DAYS)
            print(f'{p}: {len(self.hist[p])} M1 candles', flush=True)
        scan(self.hist[config.PAIRS[0]], config.PAIRS[0])   # warm-up (numba compile) before the first window

    def update(self, p, need_ms=None):
        h = self.hist[p]
        start = int(h.timestamp.iloc[-1]) + MIN if len(h) else int(time.time() * 1000) - config.HISTORY_DAYS * 1440 * MIN
        d = data_feed.candles(p, start)
        if not d.empty:
            h = pd.concat([h, d], ignore_index=True).drop_duplicates('timestamp')
            cut = int(time.time() * 1000) - (config.HISTORY_DAYS + 1) * 1440 * MIN
            self.hist[p] = h[h.timestamp >= cut].reset_index(drop=True)

    def on_5m_close(self, t_close):
        for p in config.PAIRS:
            # make sure the last minute of this candle is in (if the pair had ticks)
            waited = 0
            while waited < config.CANDLE_MAX_WAIT_SEC:
                self.update(p)
                if len(self.hist[p]) and int(self.hist[p].timestamp.iloc[-1]) >= t_close - MIN:
                    break
                time.sleep(3); waited += 3
            sig = scan(self.hist[p][self.hist[p].timestamp < t_close], p)
            if sig.empty:
                continue
            sig = sig[(sig.t_close == t_close) & sig.strategy.isin(config.ENABLED)]
            for s in sig.itertuples():
                key = (s.pair, s.strategy, s.t_close)
                if key in self.sent:
                    continue
                self.sent.add(key)
                telegram.send(signal_text(s))
                self.pending.append(s)
                print('SIGNAL', key, s.direction, flush=True)

    def check_results(self):
        now = int(time.time() * 1000)
        keep = []
        for s in self.pending:
            t_exp = s.t_entry + s.expiry * MIN
            if now < t_exp + 15_000:
                keep.append(s); continue
            try:
                pe = data_feed.close_at(s.pair, s.t_entry)
                px = data_feed.close_at(s.pair, t_exp)
            except Exception as e:
                print('result error', e, flush=True); keep.append(s); continue
            if pe is None or px is None:
                res = 'NO DATA'
            elif px == pe:
                res = 'TIE'
            else:
                res = 'WIN' if (px - pe) * s.direction > 0 else 'LOSS'
            day = dxb(s.t_close, '%Y-%m-%d')
            wr = self.window_res.setdefault((day, s.strategy), [0, 0, 0])
            if res == 'WIN': wr[0] += 1
            elif res == 'LOSS': wr[1] += 1
            elif res == 'TIE': wr[2] += 1
            mark = {'WIN': '✅', 'LOSS': '❌', 'TIE': '➖'}.get(res, '?')
            arrow = 'CALL' if s.direction == 1 else 'PUT'
            telegram.send(f'{mark} {res}: {config.DISPLAY[s.pair]} {arrow}, {s.expiry} мин, вход {dxb(s.t_entry)}\n'
                          f'Цена входа {pe} → экспирация {px} (котировки Dukascopy, mid)')
            save({'date_dubai': day, 'strategy': s.strategy, 'pair': s.pair, 'direction': arrow,
                  'candle_close_dubai': dxb(s.t_close), 'entry_dubai': dxb(s.t_entry), 'expiry_min': s.expiry,
                  'signal_price': s.price, 'entry_price': pe, 'expiry_price': px, 'result': res, 'level': LVL.get(int(s.lvl), '-')})
        self.pending = keep

    def window_summaries(self, now_ms):
        um = (now_ms // MIN) % 1440
        for k, st in STRATEGIES.items():
            if k not in config.ENABLED or um != st['end_utc'] + 2:   # 2 minutes after window end
                continue
            day = dxb(now_ms, '%Y-%m-%d')
            w, l, t = self.window_res.get((day, k), [0, 0, 0])
            n = w + l
            telegram.send(f'Итог окна «{st["name"]}» {day}: {w} WIN / {l} LOSS / {t} TIE'
                          + (f', винрейт {w / n * 100:.0f}%' if n else ', сигналов не было'))

    def run(self):
        telegram.send('Бот запущен. Окна: ' + ', '.join(STRATEGIES[k]['name'] for k in config.ENABLED) + ' (время Дубая).')
        self.load()
        while True:
            try:
                now = time.time()
                nxt = (int(now // 60) + 1) * 60 + config.CANDLE_WAIT_SEC
                time.sleep(max(0, nxt - now))
                t_min = int((nxt - config.CANDLE_WAIT_SEC) * 1000)    # just-closed minute boundary
                if (t_min // MIN) % 5 == 0 and in_any_window(t_min):
                    self.on_5m_close(t_min)
                elif (t_min // MIN) % 5 == 0:
                    for p in config.PAIRS:
                        self.update(p)
                self.check_results()
                self.window_summaries(t_min)
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                telegram.send('Ошибка бота: ' + err[-500:])
                time.sleep(30)


def check():
    print('pairs:', config.PAIRS, '| strategies:', config.ENABLED)
    telegram.send('Проверка связи: бот видит этот чат.')
    rows = []
    for p in config.PAIRS:
        h = data_feed.history(p, config.HISTORY_DAYS + 5)
        last = dxb(int(h.timestamp.iloc[-1]), '%Y-%m-%d %H:%M') if len(h) else '-'
        print(f'{p}: {len(h)} M1 candles, last {last} Dubai', flush=True)
        s = scan(h, p)
        if not s.empty:
            s = s[s.t_close >= int(time.time() * 1000) - 5 * 1440 * MIN]
            rows.append(s)
    if rows:
        s = pd.concat(rows).sort_values('t_close')
        s['close_dubai'] = s.t_close.map(lambda x: dxb(x, '%Y-%m-%d %H:%M'))
        s['dir'] = s.direction.map({1: 'CALL', -1: 'PUT'})
        print('\nSignals of the last 5 days found on history:')
        print(s[['close_dubai', 'pair', 'strategy', 'dir', 'expiry', 'price']].to_string(index=False))
    else:
        print('\nNo signals in the last 5 days.')


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
    else:
        Bot().run()
