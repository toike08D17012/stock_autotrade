"""Logging utilities for consistent logging configuration across the project.

This module provides a centralized logging setup to ensure consistent
log formatting and configuration across all modules.
"""

import logging
import sys
from typing import TextIO


DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"


def setup_logging(
    level: str | int = logging.INFO,
    log_format: str = DEFAULT_FORMAT,
    stream: TextIO = sys.stderr,
) -> None:
    """Configure application-wide logging settings.

    This function should be called once at application startup to set up
    consistent logging across all modules.

    Args:
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', logging.INFO).
        log_format: Log message format string.
        stream: Output stream for log messages.

    Example:
        >>> from src.utils.logging_config import setup_logging
        >>> setup_logging(level='DEBUG')
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format=log_format,
        stream=stream,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    This is a convenience wrapper around logging.getLogger that ensures
    consistent logger naming conventions.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> from src.utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)
