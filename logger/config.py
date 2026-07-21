from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

LogLevel = Union[int, str]

_CONFIGURED = False
_ROOT_LOGGER_NAME = "grag"


def _coerce_level(level: LogLevel) -> int:
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if not hasattr(logging, normalized):
        raise ValueError(f"Unknown log level: {level}")
    return int(getattr(logging, normalized))


def setup_logging(
    *,
    log_dir: Union[str, Path] = "logs",
    level: LogLevel = "INFO",
    filename: str = "app.log",
    error_filename: str = "errors.log",
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure the project logger once with console and rotating file output."""
    global _CONFIGURED

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    if _CONFIGURED and not force:
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_coerce_level(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path / filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_coerce_level(level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_handler = RotatingFileHandler(
            log_path / error_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the global project logger, configuring defaults if needed."""
    if not _CONFIGURED:
        setup_logging()

    if not name:
        return logging.getLogger(_ROOT_LOGGER_NAME)

    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)

    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
