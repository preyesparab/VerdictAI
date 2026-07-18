"""Production logging configuration for RepoMind.

Provides one setup entry point, `configure_logging`, that attaches a
colored console handler and a timestamped file handler to the root
logger, and one lookup helper, `get_logger`, that every other module uses
instead of calling `logging.getLogger` directly. Centralizing setup here
means log format, level, and destinations are consistent across the
entire pipeline and change in exactly one place.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[41m",  # white on red
}
_RESET = "\033[0m"

_configured = False


class ColorFormatter(logging.Formatter):
    """Console formatter that colorizes each record by severity level.

    Colors are disabled automatically when the destination stream is not
    a real terminal (e.g., output redirected to a file or captured by
    CI), since ANSI escape codes would otherwise appear as garbage
    characters instead of color.
    """

    def __init__(self, *, use_color: bool) -> None:
        """Initialize the formatter.

        Args:
            use_color: Whether to wrap formatted records in ANSI color
                codes based on their severity level.
        """
        super().__init__(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record, applying color when enabled.

        Args:
            record: The log record to format.

        Returns:
            The formatted, optionally color-wrapped log line.
        """
        message = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno, "") if self._use_color else ""
        return f"{color}{message}{_RESET}" if color else message


def configure_logging(*, force: bool = False) -> None:
    """Attach console and file handlers to the root logger.

    Idempotent by default: repeated calls are no-ops so that multiple
    modules can each call this (or `get_logger`) at import time without
    duplicating handlers.

    Args:
        force: Reconfigure and replace existing handlers even if logging
            was already configured in this process.
    """
    global _configured
    if _configured and not force:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(ColorFormatter(use_color=sys.stdout.isatty()))
    root_logger.addHandler(console_handler)

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOG_DIR / f"repomind_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally the calling module's `__name__`.

    Returns:
        A `logging.Logger` that writes to both the console and the
        current run's timestamped log file.
    """
    configure_logging()
    return logging.getLogger(name)
