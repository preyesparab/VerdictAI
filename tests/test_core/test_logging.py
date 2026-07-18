"""Tests for core.logging."""

from __future__ import annotations

import logging

from core.logging import ColorFormatter, configure_logging, get_logger


def test_get_logger_returns_logger_with_requested_name() -> None:
    logger = get_logger("repomind.test")
    assert logger.name == "repomind.test"


def test_configure_logging_attaches_console_and_file_handlers() -> None:
    configure_logging(force=True)
    root_handlers = logging.getLogger().handlers
    assert len(root_handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in root_handlers)
    assert any(isinstance(h, logging.FileHandler) for h in root_handlers)


def test_configure_logging_is_idempotent_without_force() -> None:
    configure_logging(force=True)
    handler_count = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == handler_count


def test_color_formatter_wraps_message_when_color_enabled() -> None:
    formatter = ColorFormatter(use_color=True)
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="failure", args=(), exc_info=None,
    )
    assert "\033[" in formatter.format(record)


def test_color_formatter_plain_when_color_disabled() -> None:
    formatter = ColorFormatter(use_color=False)
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="failure", args=(), exc_info=None,
    )
    assert "\033[" not in formatter.format(record)
