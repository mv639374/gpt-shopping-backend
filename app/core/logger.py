import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str = "gpt_shopping",
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logger with console and optional file handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level from settings or parameter
    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console Handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    console_handler.setFormatter(ColoredFormatter(console_format))
    logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    
    return logger


# Create global logger instance
logger = setup_logger()


def log_data_loading(table_name: str, row_count: int, status: str = "SUCCESS"):
    """
    Log data loading information for tables
    
    Args:
        table_name: Name of the table being loaded
        row_count: Number of rows loaded
        status: Loading status (SUCCESS, FAILED, etc.)
    """
    if status == "SUCCESS":
        logger.info(f"📊 Loaded table '{table_name}' with {row_count:,} rows")
    else:
        logger.error(f"❌ Failed to load table '{table_name}'")


def log_api_request(method: str, endpoint: str, status_code: int, duration_ms: float):
    """
    Log API request information
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
    """
    logger.info(
        f"🌐 {method} {endpoint} | Status: {status_code} | Duration: {duration_ms:.2f}ms"
    )


def log_cache_hit(key: str, hit: bool):
    """
    Log cache hit/miss information
    
    Args:
        key: Cache key
        hit: Whether cache was hit (True) or missed (False)
    """
    if hit:
        logger.debug(f"✅ Cache HIT: {key}")
    else:
        logger.debug(f"❌ Cache MISS: {key}")
