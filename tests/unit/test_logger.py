"""
Unit tests for logging configuration.
"""

import pytest
import logging
from pathlib import Path


class TestLogger:
    """Test suite for logger setup."""

    def test_get_logger(self):
        """Test that get_logger returns a logger."""
        from src.logger import get_logger

        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_logger_levels(self):
        """Test different log levels."""
        from src.logger import get_logger

        logger = get_logger("test_levels")

        # Should have handlers
        assert len(logger.handlers) > 0

        # Should be able to log at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_logger_with_module_name(self):
        """Test logger with module naming convention."""
        from src.logger import get_logger

        logger = get_logger("trading_bot.models")
        assert logger.name == "trading_bot.models"

        logger = get_logger("trading_bot.api")
        assert logger.name == "trading_bot.api"

    def test_setup_logger_console_only(self):
        """Test logger with only console output."""
        from src.logger import setup_logger

        logger = setup_logger(
            name="console_only",
            log_to_console=True,
            log_to_file=False
        )

        # Should have only console handler
        assert len([h for h in logger.handlers if isinstance(h, logging.StreamHandler)]) >= 1

    def test_setup_logger_file_only(self, tmp_path):
        """Test logger with only file output."""
        from src.logger import setup_logger

        log_file = tmp_path / "test.log"

        logger = setup_logger(
            name="file_only",
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file
        )

        # Write a log message
        logger.info("Test message")

        # File should exist
        assert log_file.exists()

        # File should contain the message
        content = log_file.read_text()
        assert "Test message" in content
        assert "INFO" in content

    def test_setup_logger_custom_format(self):
        """Test logger with custom format."""
        from src.logger import setup_logger

        custom_format = "%(levelname)s - %(message)s"

        logger = setup_logger(
            name="custom_format",
            log_format=custom_format,
            log_to_console=True,
            log_to_file=False
        )

        assert len(logger.handlers) > 0

    def test_setup_logger_different_levels(self):
        """Test logger with different log levels."""
        from src.logger import setup_logger

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger = setup_logger(
                name=f"level_{level.lower()}",
                level=level,
                log_to_console=True,
                log_to_file=False
            )

            assert logger.level == getattr(logging, level)


class TestLoggerIntegration:
    """Integration tests for logger usage."""

    def test_logger_in_different_modules(self):
        """Test that different modules can have their own loggers."""
        from src.logger import get_logger

        data_logger = get_logger("trading_bot.data")
        model_logger = get_logger("trading_bot.model")
        api_logger = get_logger("trading_bot.api")

        assert data_logger.name == "trading_bot.data"
        assert model_logger.name == "trading_bot.model"
        assert api_logger.name == "trading_bot.api"

    def test_logger_exception_logging(self):
        """Test logging exceptions."""
        from src.logger import get_logger

        logger = get_logger("test_exceptions")

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        # Should not raise exception
        assert True
