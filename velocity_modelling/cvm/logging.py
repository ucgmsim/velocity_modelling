"""
Velocity Model Logging Module

This module provides a consistent logging interface for the velocity modeling system.
It defines the VMLogger class which wraps Python's standard logging module with
simpler methods adapted to the needs of the CVM system.
"""

import logging
import sys
from typing import Optional, Union, TextIO


class VMLogger:
    """
    A logging wrapper for the velocity model system.

    This class provides a consistent interface for logging across the entire
    CVM codebase, with configurable output destinations and log levels.

    Attributes
    ----------
    DEBUG : int
        Debug log level (10)
    INFO : int
        Info log level (20)
    WARNING : int
        Warning log level (30)
    ERROR : int
        Error log level (40)
    CRITICAL : int
        Critical log level (50)
    """

    # Define log levels as class attributes for convenient access
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(
        self,
        name: str = "velocity_model",
        level: Union[str, int] = logging.INFO,
        output_file: Optional[Union[str, TextIO]] = None,
        format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ):
        """
        Initialize a new VMLogger.

        Parameters
        ----------
        name : str, optional
            Logger name, default is "velocity_model"
        level : str or int, optional
            Log level, can be a string (DEBUG, INFO, etc.) or integer level, default is INFO
        output_file : str or file-like object, optional
            File to write logs to (in addition to console), default is None
        format_str : str, optional
            Format string for log messages
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), logging.INFO)
        else:
            numeric_level = level

        # Set up the Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates if logger already exists
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter(format_str)
        console_handler.setFormatter(formatter)

        # Add console handler to logger
        self.logger.addHandler(console_handler)

        # Add file handler if output_file is specified
        if output_file:
            if isinstance(output_file, str):
                file_handler = logging.FileHandler(output_file)
            else:
                file_handler = logging.StreamHandler(output_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Log a message at the specified level.

        This is the primary interface for logging in the velocity model system.

        Parameters
        ----------
        message : str
            The message to log
        level : int, optional
            Log level, default is INFO
        """
        self.logger.log(level, message)

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        Parameters
        ----------
        message : str
            The debug message to log
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Parameters
        ----------
        message : str
            The info message to log
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Parameters
        ----------
        message : str
            The warning message to log
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an error message.

        Parameters
        ----------
        message : str
            The error message to log
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """
        Log a critical message.

        Parameters
        ----------
        message : str
            The critical message to log
        """
        self.logger.critical(message)

    def set_level(self, level: Union[str, int]) -> None:
        """
        Set the logging level.

        Parameters
        ----------
        level : str or int
            The level to set, can be a string (DEBUG, INFO, etc.) or integer level
        """
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), logging.INFO)
        else:
            numeric_level = level

        self.logger.setLevel(numeric_level)
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)

    def get_logger(self) -> logging.Logger:
        """
        Get the underlying Python logger.

        Returns
        -------
        logging.Logger
            The underlying Python logger object
        """
        return self.logger
