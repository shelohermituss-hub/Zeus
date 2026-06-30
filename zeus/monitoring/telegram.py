"""
Telegram notifier — sends one-way trading alerts and status messages.

Strictly send-only: this module calls only the Telegram Bot API's
sendMessage endpoint. It never polls getUpdates or processes incoming
messages, so nothing arriving in the Telegram chat can ever feed a
command back into the bot — this is a notification channel, not a
remote control surface, and it must never gain the ability to place,
modify, or cancel an order.

A network or API failure here must never interrupt trading: every
public method catches its own errors, logs them, and returns False
instead of raising. Notifications are observability, not part of the
risk-control path.
"""
from __future__ import annotations

import requests

from zeus.monitoring.alerts import Alert
from zeus.utils.logger import logger

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    """
    Send-only Telegram notifications via the Bot API.

    Disabled whenever bot_token or chat_id is empty — every method then
    becomes a no-op returning False, so callers can wire a notifier in
    unconditionally without an `if configured:` check at every call site.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        """True when both bot_token and chat_id are configured."""
        return bool(self._bot_token) and bool(self._chat_id)

    def send(self, text: str) -> bool:
        """
        Send a plain-text message to the configured chat.

        Returns True only on a confirmed 2xx response. Returns False
        (never raises) when disabled, on an empty message, on any network
        error, or on a non-2xx response — retrying transient failures
        (network errors and 5xx) up to max_retries times.
        """
        if not self.enabled or not text:
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": text}

        for attempt in range(self._max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self._timeout)
            except requests.RequestException as exc:
                logger.warning(
                    "Telegram notification network error",
                    error=str(exc), attempt=attempt,
                )
                continue

            if response.status_code == 200:
                return True
            if response.status_code < 500:
                # Client error (bad token, bad chat_id, ...) — retrying won't help.
                logger.warning(
                    "Telegram notification rejected",
                    status_code=response.status_code,
                )
                return False
            logger.warning(
                "Telegram notification server error",
                status_code=response.status_code, attempt=attempt,
            )

        return False

    def notify_alert(self, alert: Alert) -> bool:
        """Format and send a monitoring.alerts.Alert."""
        text = f"[{alert.level}] {alert.code}\n{alert.message}"
        return self.send(text)
