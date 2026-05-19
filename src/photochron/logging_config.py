"""
Centralized logging configuration using loguru.

Replaces stdlib logging setup. Also installs an InterceptHandler that
redirects stdlib loggers (InsightFace, httpx, ollama, etc.) into loguru
so all log lines share one format and sink.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

from photochron.config.models import ConfigLogging


class InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _format_extra(record: Any) -> str:
    """Render bound context fields (run_id, stage, etc.) compactly."""
    extra = record.get("extra") or {}
    if not extra:
        return ""
    parts = " ".join(f"{k}={v}" for k, v in extra.items())
    return f" [{parts}]"


def _console_format(record: Any) -> str:
    return (
        "<green>{time:HH:mm:ss}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>" + _format_extra(record) + "\n{exception}"
    )


def _file_format(record: Any) -> str:
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} | {message}" + _format_extra(record) + "\n{exception}"
    )


def setup_logging(
    cfg: ConfigLogging,
    level_override: str | None = None,
) -> None:
    """
    Configure loguru sinks from a ConfigLogging instance.

    Args:
        cfg: Logging configuration (level, file_path, rotation, etc.).
        level_override: Optional CLI-provided level that overrides cfg.level.
    """
    logger.remove()

    console_level = (level_override or cfg.level).upper()

    logger.add(
        sys.stderr,
        level=console_level,
        format=_console_format,
        colorize=True,
        backtrace=cfg.backtrace,
        diagnose=cfg.diagnose,
        enqueue=False,
    )

    if cfg.file_path:
        file_path = Path(cfg.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(file_path),
            level=cfg.file_level.upper(),
            format=_file_format,
            rotation=cfg.rotation,
            retention=cfg.retention,
            serialize=cfg.json_format,
            backtrace=cfg.backtrace,
            diagnose=cfg.diagnose,
            enqueue=True,
        )

    _install_stdlib_intercept(console_level)


def _install_stdlib_intercept(level: str) -> None:
    """Redirect stdlib logging into loguru."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for name in ("httpx", "httpcore", "urllib3", "insightface", "onnxruntime"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.propagate = False
        stdlib_logger.setLevel(getattr(logging, level, logging.INFO))


__all__ = ["setup_logging", "InterceptHandler"]
