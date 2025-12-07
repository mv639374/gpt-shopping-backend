import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def create_batch_logger(batch_id: str) -> logging.Logger:
    """
    Create a dedicated file logger for a batch processing session
    
    Args:
        batch_id: Unique identifier for this batch (timestamp-based)
    
    Returns:
        Logger instance that writes to file
    """
    # Create log filename
    log_filename = LOGS_DIR / f"batch_{batch_id}.log"
    
    # Create logger
    batch_logger = logging.getLogger(f"batch_{batch_id}")
    batch_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    batch_logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    batch_logger.addHandler(file_handler)
    
    # Also add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    batch_logger.addHandler(console_handler)
    
    batch_logger.info("=" * 80)
    batch_logger.info(f"📝 Batch Processing Log Started: {batch_id}")
    batch_logger.info(f"📁 Log file: {log_filename}")
    batch_logger.info("=" * 80)
    
    return batch_logger


def close_batch_logger(batch_logger: logging.Logger):
    """Close and clean up batch logger"""
    batch_logger.info("=" * 80)
    batch_logger.info("📝 Batch Processing Log Completed")
    batch_logger.info("=" * 80)
    
    # Close all handlers
    for handler in batch_logger.handlers:
        handler.close()
        batch_logger.removeHandler(handler)
