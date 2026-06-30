"""Tests for zeus.utils.logger — smoke tests."""

from zeus.utils.logger import logger, setup_logger


def test_setup_logger_does_not_raise():
    setup_logger("DEBUG")


def test_logger_can_emit_messages(capsys):
    setup_logger("DEBUG")
    logger.info("test message from pytest")
    # No assertion needed — we verify no exception is raised
