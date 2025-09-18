"""Logging configuration for NFL Trading System."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import Config, get_config


class LoggerConfig:
    """Centralized logging configuration."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize logger configuration.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self._setup_loguru()
        self._setup_standard_logging()

    def _setup_loguru(self):
        """Setup loguru logger with configuration."""
        # Remove default logger
        logger.remove()

        # Setup console logging
        if self.config.logging.console_output:
            logger.add(
                sys.stdout,
                level=self.config.logging.level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                       "<level>{message}</level>",
                colorize=True,
                backtrace=True,
                diagnose=True
            )

        # Setup file logging with rotation
        log_file_path = Path(self.config.logging.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file_path,
            level=self.config.logging.level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=self.config.logging.max_file_size,
            retention=self.config.logging.backup_count,
            compression="zip",
            backtrace=True,
            diagnose=True,
            serialize=True if self.config.environment == 'production' else False
        )

        # Add error-specific file
        error_log_path = log_file_path.parent / "errors.log"
        logger.add(
            error_log_path,
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=self.config.logging.max_file_size,
            retention=self.config.logging.backup_count,
            compression="zip",
            backtrace=True,
            diagnose=True
        )

    def _setup_standard_logging(self):
        """Setup standard Python logging to work with loguru."""
        # Create custom handler that forwards to loguru
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # Get corresponding Loguru level if it exists
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # Find caller from where originated the logged message
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        # Set up intercept handler
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # Configure specific loggers
        for logger_name in ['urllib3', 'requests', 'httpx']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def get_logger(self, name: str):
        """Get a logger instance.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        return logger.bind(name=name)

    def configure_module_logger(self, module_name: str, level: Optional[str] = None):
        """Configure logging for a specific module.

        Args:
            module_name: Name of the module
            level: Logging level (defaults to config level)
        """
        level = level or self.config.logging.level
        logger.bind(name=module_name).level = level

    def add_file_handler(self, file_path: str, level: str = "INFO", rotation: str = "10MB"):
        """Add an additional file handler.

        Args:
            file_path: Path to log file
            level: Logging level
            rotation: Rotation policy
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            file_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=self.config.logging.backup_count,
            compression="zip"
        )

    def set_level(self, level: str):
        """Dynamically change logging level.

        Args:
            level: New logging level
        """
        logger.remove()
        self.config.logging.level = level
        self._setup_loguru()


# Global logger configuration
_logger_config = None


def setup_logging(config: Optional[Config] = None) -> LoggerConfig:
    """Setup global logging configuration.

    Args:
        config: Configuration object

    Returns:
        LoggerConfig instance
    """
    global _logger_config
    _logger_config = LoggerConfig(config)
    return _logger_config


def get_logger(name: str):
    """Get a configured logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if _logger_config is None:
        setup_logging()
    return _logger_config.get_logger(name)


def get_logger_config() -> LoggerConfig:
    """Get the global logger configuration."""
    if _logger_config is None:
        setup_logging()
    return _logger_config