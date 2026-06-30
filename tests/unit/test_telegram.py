"""
Tests for zeus/monitoring/telegram.py.

All HTTP calls are mocked — no real network access. Verifies the
notifier is send-only (only ever calls sendMessage), fails closed
(no raises, returns False) on any error, and never calls the network
when disabled.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from zeus.monitoring.alerts import Alert
from zeus.monitoring.telegram import TelegramNotifier


def _response(status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


# ======================================================================
# Construction validation
# ======================================================================

class TestConstruction:
    def test_rejects_non_positive_timeout(self):
        with pytest.raises(ValueError):
            TelegramNotifier(bot_token="t", chat_id="c", timeout_seconds=0)

    def test_rejects_negative_max_retries(self):
        with pytest.raises(ValueError):
            TelegramNotifier(bot_token="t", chat_id="c", max_retries=-1)


# ======================================================================
# Enabled / disabled state
# ======================================================================

class TestEnabled:
    def test_disabled_when_token_empty(self):
        notifier = TelegramNotifier(bot_token="", chat_id="123")
        assert notifier.enabled is False

    def test_disabled_when_chat_id_empty(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="")
        assert notifier.enabled is False

    def test_disabled_when_both_empty(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        assert notifier.enabled is False

    def test_enabled_when_both_set(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        assert notifier.enabled is True

    def test_disabled_notifier_never_calls_network(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        with patch("zeus.monitoring.telegram.requests.post") as mock_post:
            result = notifier.send("hello")
        mock_post.assert_not_called()
        assert result is False


# ======================================================================
# send() — success path
# ======================================================================

class TestSendSuccess:
    def test_returns_true_on_200(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            result = notifier.send("hello")
        assert result is True
        mock_post.assert_called_once()

    def test_posts_to_send_message_endpoint(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            notifier.send("hello")
        url = mock_post.call_args.args[0]
        assert url == "https://api.telegram.org/botabc/sendMessage"

    def test_payload_contains_chat_id_and_text(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            notifier.send("hello world")
        payload = mock_post.call_args.kwargs["json"]
        assert payload == {"chat_id": "123", "text": "hello world"}

    def test_request_has_timeout(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", timeout_seconds=2.5)
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            notifier.send("hello")
        assert mock_post.call_args.kwargs["timeout"] == pytest.approx(2.5)

    def test_empty_text_returns_false_without_network_call(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post") as mock_post:
            result = notifier.send("")
        mock_post.assert_not_called()
        assert result is False


# ======================================================================
# send() — failure paths (must never raise)
# ======================================================================

class TestSendFailure:
    def test_network_exception_returns_false(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", max_retries=0)
        with patch(
            "zeus.monitoring.telegram.requests.post",
            side_effect=requests.ConnectionError("boom"),
        ):
            result = notifier.send("hello")
        assert result is False

    def test_client_error_returns_false_without_retry(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", max_retries=3)
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(400)) as mock_post:
            result = notifier.send("hello")
        assert result is False
        mock_post.assert_called_once()  # no retry on a 4xx

    def test_server_error_is_retried_up_to_max_retries(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", max_retries=2)
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(500)) as mock_post:
            result = notifier.send("hello")
        assert result is False
        assert mock_post.call_count == 3  # initial attempt + 2 retries

    def test_server_error_then_success_returns_true(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", max_retries=2)
        with patch(
            "zeus.monitoring.telegram.requests.post",
            side_effect=[_response(500), _response(200)],
        ) as mock_post:
            result = notifier.send("hello")
        assert result is True
        assert mock_post.call_count == 2

    def test_send_never_raises_on_unexpected_exception(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123", max_retries=0)
        with patch(
            "zeus.monitoring.telegram.requests.post",
            side_effect=requests.Timeout("timed out"),
        ):
            result = notifier.send("hello")  # must not raise
        assert result is False


# ======================================================================
# Read-only guarantee — never calls anything but sendMessage
# ======================================================================

class TestReadOnlyGuarantee:
    def test_send_never_calls_requests_get(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)), \
             patch("zeus.monitoring.telegram.requests.get") as mock_get:
            notifier.send("hello")
        mock_get.assert_not_called()

    def test_url_never_contains_get_updates(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            notifier.send("hello")
        url = mock_post.call_args.args[0]
        assert "getUpdates" not in url
        assert url.endswith("/sendMessage")


# ======================================================================
# notify_alert()
# ======================================================================

class TestNotifyAlert:
    def test_formats_error_alert(self):
        notifier = TelegramNotifier(bot_token="abc", chat_id="123")
        alert = Alert(level="ERROR", code="DRAWDOWN_CRITICAL", message="Drawdown 12% reached")
        with patch("zeus.monitoring.telegram.requests.post", return_value=_response(200)) as mock_post:
            result = notifier.notify_alert(alert)
        assert result is True
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "ERROR" in text
        assert "DRAWDOWN_CRITICAL" in text
        assert "Drawdown 12% reached" in text

    def test_disabled_notifier_skips_alert(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        alert = Alert(level="WARNING", code="HIGH_LATENCY", message="slow")
        with patch("zeus.monitoring.telegram.requests.post") as mock_post:
            result = notifier.notify_alert(alert)
        assert result is False
        mock_post.assert_not_called()
