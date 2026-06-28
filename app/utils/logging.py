import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

from app.config.config import settings


# ANSI Color codes for console output
class ColorFormatter(logging.Formatter):
    """
    Custom console formatter that adds colors to logs based on severity levels.
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"

    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + FORMAT + reset,
        logging.INFO: blue + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for production-ready log aggregation.
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }
        # Add extra details if present
        if hasattr(record, "extra_info"):
            log_data["extra_info"] = record.extra_info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logger(name: str, log_filename: str, level: str = "INFO") -> logging.Logger:
    """
    Initializes a logger with both colored console and rotated JSON file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if already initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # 2. Daily Rotating JSON File Handler
    log_dir = settings.LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_filename)

    file_handler = TimedRotatingFileHandler(
        file_path, when="midnight", interval=1, backupCount=30, encoding="utf-8"  # Keep last 30 logs
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


# Global logger registry instances
system_logger = setup_logger("system", "system.log", settings.LOG_LEVEL)
request_logger = setup_logger("requests", "requests.log", settings.LOG_LEVEL)
inference_logger = setup_logger("inference", "inference.log", settings.LOG_LEVEL)
training_logger = setup_logger("training", "training.log", settings.LOG_LEVEL)
error_logger = setup_logger("errors", "errors.log", settings.LOG_LEVEL)


def get_system_logger():
    return system_logger


def get_request_logger():
    return request_logger


def get_inference_logger():
    return inference_logger


def get_training_logger():
    return training_logger


def get_error_logger():
    return error_logger
