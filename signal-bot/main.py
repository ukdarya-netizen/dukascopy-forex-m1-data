"""Telegram signal bot for manual binary-options trading (Pocket Option).
Run:  python main.py           -> live bot
      python main.py --check   -> setup check: Dukascopy feed + Telegram + signals of the last 5 days found on history (nothing is sent except one test message)
"""
import os, sys, time, csv, traceback
from datetime import datetime, timezone, timedelta
import logging
import pandas as pd
import config, data_feed, telegram
logging.getLogger('dukascopy').setLevel(logging.WARNING)
from strategy import scan, STRATEGIES
from reports import Scheduler

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
            f'Свеча {st["tf"]}m закрылась {dxb(s.t_close)}, цена {config.fmt_price(s.pair, s.price)}, уровень: {LVL.get(int(s.lvl), "-")}')


def pips(pair, pe, px, direction):
    """Signed distance entry -> expiry in pips: + in favour of the trade, - against."""
    if pe is None or px is None:
        return None
    return round((px - pe) * direction / config.PIP[pair], 1)


def fmt_pips(v):
    if v is None or v != v:
        return '- п.'
    return f'{v:+.1f} п.'


def due_strategies(t_close):
    """Enabled strategies whose candle closes at t_close and whose entry/expiry fit the window."""
    te = t_close + MIN
    um = (te // MIN) % 1440
    return [k for k, st in STRATEGIES.items() if k in config.ENABLED and t_close % (st['tf'] * MIN) == 0
            and st['start_utc'] <= um and um + st['expiry'] <= st['end_utc']]


def read_log():
    if not os.path.exists(config.SIGNALS_CSV):
        return pd.DataFrame()
    return pd.read_csv(config.SIGNALS_CSV, dtype=str)


def save(row):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    new = not os.path.exists(config.SIGNALS_CSV)
    if not new:   # older log without some columns (e.g. pips) -> upgrade header in place
        old = pd.read_csv(config.SIGNALS_CSV, dtype=str)
        if list(old.columns) != list(row.keys()):
            old = old.reindex(columns=list(row.keys()))
            old.to_csv(config.SIGNALS_CSV, index=False)
    with open(config.SIGNALS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)


class Bot:
    def __init__(self):
        self.hist = {}
        self.pending = []       # signals waiting for result
        self.sent = set()
        self.reports = Scheduler(STRATEGIES, config.ENABLED)

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

    def on_close(self, t_close, due):
        for p in config.PAIRS:
            # make sure the last minute of this candle is in (if the pair had ticks)
            waited = 0
            while waited < config.CANDLE_MAX_WAIT_SEC:
                self.update(p)
                if len(self.hist[p]) and int(self.hist[p].timestamp.iloc[-1]) >= t_close - MIN:
                    break
                time.sleep(3); waited += 3
            sig = scan(self.hist[p][self.hist[p].timestamp < t_close], p, only=due)
            if sig.empty:
                continue
            sig = sig[(sig.t_close == t_close) & sig.strategy.isin(due)]
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
            mark = {'WIN': '✅', 'LOSS': '❌', 'TIE': '➖'}.get(res, '?')
            arrow = 'CALL' if s.direction == 1 else 'PUT'
            pp = pips(s.pair, pe, px, s.direction)
            telegram.send(f'{mark} {res}: {config.DISPLAY[s.pair]} {arrow}, {s.expiry} мин, вход {dxb(s.t_entry)}\n'
                          f'Цена входа {config.fmt_price(s.pair, pe)} → экспирация {config.fmt_price(s.pair, px)}, разница {fmt_pips(pp)} (котировки Dukascopy, mid)')
            save({'date_dubai': day, 'strategy': s.strategy, 'pair': s.pair, 'direction': arrow,
                  'candle_close_dubai': dxb(s.t_close), 'entry_dubai': dxb(s.t_entry), 'expiry_min': s.expiry,
                  'signal_price': s.price, 'entry_price': pe, 'expiry_price': px, 'pips': pp, 'result': res, 'level': LVL.get(int(s.lvl), '-')})
        self.pending = keep

    def status(self):
        now = int(time.time() * 1000)
        last = {p: (dxb(int(h.timestamp.iloc[-1]) + MIN, '%Y-%m-%d %H:%M') if len(h) else None) for p, h in list(self.hist.items())}
        return {'bot': 'running', 'now_dubai': dxb(now, '%Y-%m-%d %H:%M:%S'), 'strategies': config.ENABLED,
                'pairs_loaded': len(self.hist), 'last_candle_close_dubai': last, 'signals_waiting_result': len(self.pending)}

    def serve_status(self):
        """Tiny HTTP server on $PORT: lets the bot run as the project's web/API artifact (health check) and shows its status."""
        import json, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        bot = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(bot.status(), ensure_ascii=False, indent=1).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

            def log_message(self, *a):
                pass

        port = int(os.environ.get('PORT', '8080'))
        srv = ThreadingHTTPServer(('0.0.0.0', port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f'status server on port {port}', flush=True)

    def run(self):
        self.serve_status()                     # first, so the deployment health check passes while history loads
        telegram.send('Бот запущен. Окна: ' + ', '.join(STRATEGIES[k]['name'] for k in config.ENABLED) + ' (время Дубая).')
        self.load()
        while True:
            try:
                now = time.time()
                nxt = (int(now // 60) + 1) * 60 + config.CANDLE_WAIT_SEC
                time.sleep(max(0, nxt - now))
                t_min = int((nxt - config.CANDLE_WAIT_SEC) * 1000)    # just-closed minute boundary
                due = due_strategies(t_min)
                if due:
                    self.on_close(t_min, due)
                elif (t_min // MIN) % 5 == 0:
                    for p in config.PAIRS:
                        self.update(p)
                self.check_results()
                for text in self.reports.due(t_min, {x.strategy for x in self.pending}):
                    telegram.send(text)
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
        s = scan(h, p, only=config.ENABLED)
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


def chat_ids():
    """Show chats that wrote to the bot (write /start to the bot or post in the group/channel first)."""
    import requests
    r = requests.get(f'https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates', timeout=15).json()
    if not r.get('ok'):
        print('Telegram error:', r); return
    seen = {}
    for u in r.get('result', []):
        m = u.get('message') or u.get('channel_post') or u.get('my_chat_member') or {}
        c = m.get('chat')
        if c: seen[c['id']] = (c.get('type'), c.get('title') or c.get('username') or c.get('first_name'))
    if not seen:
        print('No chats yet: write /start to the bot (or post in the group/channel where the bot is admin) and run again.')
    for cid, (tp, name) in seen.items():
        print(f'TELEGRAM_CHAT_ID = {cid}   ({tp}: {name})')


if __name__ == '__main__':
    if '--chat-id' in sys.argv:
        chat_ids()
    elif '--check' in sys.argv:
        check()
    else:
        Bot().run()
