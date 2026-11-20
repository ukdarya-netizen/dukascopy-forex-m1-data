import requests
import config


def send(text):
    """Plain text only (no HTML / Markdown parse mode)."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print('[telegram disabled]\n' + text, flush=True)
        return
    url = f'https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage'
    for _ in range(3):
        try:
            r = requests.post(url, data={'chat_id': config.TELEGRAM_CHAT_ID, 'text': text,
                                         'disable_web_page_preview': True}, timeout=15)
            if r.status_code == 200:
                return
            print('telegram error', r.status_code, r.text[:200], flush=True)
        except requests.RequestException as e:
            print('telegram error', e, flush=True)
