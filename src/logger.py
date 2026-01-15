"""
Logging configuration for AdvanceProfit-EA.

This module provides a centralized logging setup that can be used
throughout the application. It reads configuration from config.yaml.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8}"
                f"{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str = "trading_bot",
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file_path: Optional[Path] = None,
    log_format: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file_path: Path to log file (if None, uses default)
        log_format: Custom log format string
        use_colors: Use colored output for console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # Prevent messages from propagating to root logger (avoids duplicate logs)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if use_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        else:
            console_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_file_path is None:
            # Default to logs directory
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file_path = log_dir / f"{name}.log"

        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (max 10MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))

        # File format (no colors)
        file_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """
    Get a logger instance with configuration from config.yaml.

    Args:
        name: Logger name (defaults to "trading_bot")

    Returns:
        Configured logger instance

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting trading bot")
        >>> logger.warning("Low confidence prediction")
        >>> logger.error("API connection failed")
    """
    # Check if logger already configured
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Try to load config
    try:
        from src.config import get_config
        config = get_config()

        level = config.get("logging.level", "INFO")
        log_format = config.get("logging.format")
        log_to_file = config.get("logging.file") is not None
        log_to_console = config.get("logging.console", True)

        # Get log file path from config
        log_file = config.get("logging.file")
        if log_file:
            project_root = Path(__file__).parent.parent
            log_file_path = project_root / log_file
        else:
            log_file_path = None

    except Exception:
        # Fall back to defaults if config not available
        level = "INFO"
        log_format = None
        log_to_file = True
        log_to_console = True
        log_file_path = None

    return setup_logger(
        name=name,
        level=level,
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_file_path=log_file_path,
        log_format=log_format,
        use_colors=True,
    )


# Create a default logger for the module
logger = get_logger("trading_bot")


if __name__ == "__main__":
    # Example usage
    logger = get_logger("example")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Usage in different modules
    data_logger = get_logger("trading_bot.data")
    data_logger.info("Fetching EURUSD data...")

    model_logger = get_logger("trading_bot.model")
    model_logger.info("Training model...")
    model_logger.warning("Low validation accuracy")

    api_logger = get_logger("trading_bot.api")
    api_logger.info("API server started on http://127.0.0.1:8000")
    api_logger.error("Failed to load model")
