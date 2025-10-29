"""
Log manage Module

Provide log management functionality categorized by timestamp and level

Usage example:
    from src.managers.log import Logger, create_logger

    # Basic Usage
    logger = Logger("logs", "my_app")
    logger.info("This is an info")
    logger.error("This is an error")
    logger.close()

    # Use the context manager
    with Logger("logs", "my_app") as logger:
        logger.info("Automatically manage resources")

    # Convenient function
    logger = create_logger("logs", "my_app")
    logger.info("Convenient create")
    logger.close()
"""

from .logger import (
    Logger,
    create_logger,
    init_global_logger,
    get_global_logger,
    set_global_logger,
)

__all__ = [
    "Logger",
    "create_logger",
    "init_global_logger",
    "get_global_logger",
    "set_global_logger",
]

__version__ = "1.0.0"
__author__ = "Tokfinity Team"
__description__ = "Timestamp:Directory and Level-Based Log Manager "
