"""
FabSight Logging Configuration
Centralized logging setup
"""
import logging
import os
from src.config import LOG_LEVEL, LOG_FORMAT, LOG_DIR

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Setup a logger with both console and file handlers.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_path = os.path.join(LOG_DIR, log_file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
