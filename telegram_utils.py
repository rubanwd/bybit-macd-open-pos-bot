import os
import logging
import requests
from typing import Optional

TELEGRAM_API = "https://api.telegram.org"

class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_document(self, file_path: str, caption: Optional[str] = None):
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendDocument"
        with open(file_path, "rb") as f:
            files = {"document": (os.path.basename(file_path), f, "text/plain")}
            data = {"chat_id": self.chat_id}
            if caption:
                data["caption"] = caption[:1024]
            r = requests.post(url, data=data, files=files, timeout=30)
        if r.status_code != 200:
            logging.error(f"Ошибка отправки в Telegram: {r.status_code} {r.text}")
            return False
        logging.info("Отчёт отправлен в Telegram.")
        return True

    def send_message(self, text: str):
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text[:4000]}
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            logging.error(f"Ошибка sendMessage: {r.status_code} {r.text}")
            return False
        return True
