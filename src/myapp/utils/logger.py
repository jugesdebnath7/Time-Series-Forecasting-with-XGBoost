# myapp/utils/logger.py
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from myapp.config.config_manager import ConfigManager


class CustomLogger:
    """Custom logger setup using configuration from ConfigManager."""

    def __init__(self):
        """Initialize logger based on configuration."""
        self.config = ConfigManager().app

        # Paths
        self.log_dir = self.config.paths.logs
        self.name = self.config.logging.app_name
        self.log_to_console = self.config.logging.log_to_console
        self.log_file_name = self.config.logging.handlers["file"].filename
        self.log_file = os.path.join(self.log_dir, self.log_file_name)

        # Handler config
        self.handler_cfg = self.config.logging.handlers["file"]
        self.max_bytes = self.handler_cfg.maxBytes
        self.backup_count = self.handler_cfg.backupCount

        # Logging level
        level_str = self.config.logging.level.upper()
        if not hasattr(logging, level_str):
            raise ValueError(f"Invalid log level in config: {level_str}")
        self.level = getattr(logging, level_str)


    def get_logger(self) -> logging.Logger:
        os.makedirs(self.log_dir, exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False

        if not logger.handlers:
            # File handler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )

            # Set file handler log level from config
            file_level_str = self.handler_cfg.level.upper()
            if not hasattr(logging, file_level_str):
                raise ValueError(f"Invalid file handler log level: {file_level_str}")
            file_handler.setLevel(getattr(logging, file_level_str))

            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Optional: Console handler
            if self.config.logging.log_to_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.level)
                console_formatter = logging.Formatter(
                    fmt="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

            logger.info(f"Logger initialized. Writing to {self.log_file}")

        return logger
