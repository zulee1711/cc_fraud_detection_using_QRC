import logging
import os
from pathlib import Path
from typing import Optional

_LOGGER_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path.joinpath(PROJECT_ROOT, log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)