'''
    Colored Logger Configuration
    Release Date: 2025-10-07
'''

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m'
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        super().__init__(fmt, datefmt)

    def format(self, record):
        original_format = super().format(record)
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_format = (
            f"{self.COLORS['BOLD']}{record.asctime}{self.COLORS['RESET']} | "
            f"{level_color}{self.COLORS['BOLD']}{record.levelname:8}{self.COLORS['RESET']} | "
            f"{self.COLORS['UNDERLINE']}{record.name}{self.COLORS['RESET']} | "
            f"{record.getMessage()}"
        )

        return colored_format


class MLLogger:
    """ML Project Logger with colors and custom formatting"""

    def __init__(self, name: str = "MLProject", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        if self.logger.handlers:
            self.logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        colored_formatter = ColoredFormatter()
        console_handler.setFormatter(colored_formatter)


        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def get_logger(self):
        """Return the configured logger"""
        return self.logger

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


def setup_colored_logger(name: str = "MLProject", level: str = "INFO") -> logging.Logger:
    """
    Setup a colored logger for ML projects

    Args:
        name (str): Logger name
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger with colors
    """
    ml_logger = MLLogger(name, level)
    return ml_logger.get_logger()


def setup_file_logger(
        name: str = "MLProject",
        level: str = "INFO",
        log_file: str = "ml_project.log"
) -> logging.Logger:
    """
    Setup a file logger (without colors for file output)

    Args:
        name (str): Logger name
        level (str): Logging level
        log_file (str): Path to log file

    Returns:
        logging.Logger: Configured logger for file output
    """
    logger = logging.getLogger(f"{name}_file")
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def setup_dual_logger(
        name: str = "MLProject",
        level: str = "INFO",
        log_file: str = "ml_project.log"
) -> tuple[logging.Logger, logging.Logger]:
    """
    Setup both colored console logger and file logger

    Args:
        name (str): Logger name
        level (str): Logging level
        log_file (str): Path to log file

    Returns:
        tuple: (console_logger, file_logger)
    """
    console_logger = setup_colored_logger(name, level)
    file_logger = setup_file_logger(name, level, log_file)

    return console_logger, file_logger



if __name__ == "__main__":
    logger = setup_colored_logger("TestLogger", "DEBUG")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n" + "=" * 50)
    print("Testing dual logger...")


    console_logger, file_logger = setup_dual_logger("DualTest", "INFO", "test.log")
    console_logger.info("This appears in console with colors")
    file_logger.info("This appears in file without colors")
