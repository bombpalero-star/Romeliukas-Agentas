"""
Telegram pranešimų modulis.
Reikia sukurti botą per @BotFather ir gauti:
  - TELEGRAM_BOT_TOKEN
  - TELEGRAM_CHAT_ID
"""

import logging
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    def _send(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            log.warning("Telegram kredencialai nenurodyti – pranešimas neišsiųstas")
            return False
        try:
            resp = requests.post(
                TELEGRAM_API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            resp.raise_for_status()
            log.info("Telegram pranešimas išsiųstas")
            return True
        except requests.RequestException as exc:
            log.error("Telegram klaida: %s", exc)
            return False

    def send_drop_alert(self, dropped: pd.DataFrame, threshold: float) -> None:
        today = date.today().strftime("%Y-%m-%d")
        lines = [
            f"<b>📉 ETF/Akcijų kritimo ataskaita – {today}</b>",
            f"Slenkstis: {threshold}%\n",
        ]
        for row in dropped.itertuples():
            lines.append(
                f"<b>{row.ticker}</b>: {row.change_pct:+.2f}%  "
                f"(${row.prev_close} → ${row.last_close})"
            )
        self._send("\n".join(lines))
