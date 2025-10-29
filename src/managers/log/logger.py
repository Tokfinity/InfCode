"""

Log Manager
Create dir according to timestamp, store level-based log files
"""

import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


"""
Module-level Self-defined NOTICE level log (between INFO and WARNING)
"""
NOTICE_LEVEL = 25
if not hasattr(logging, "NOTICE"):
    logging.addLevelName(NOTICE_LEVEL, "NOTICE")

    def notice(self, message, *args, **kwargs):
        if self.isEnabledFor(NOTICE_LEVEL):
            self._log(NOTICE_LEVEL, message, args, **kwargs)

    logging.Logger.notice = notice  # type: ignore[attr-defined]

class ImageBuilderLogger:
    """
    ImageBuilder log manager

    Function:
    - Create sub dir based on timestamp
    - Create debug.log, info.log and error.log
    - Provide standard log format, able to print logs two levels up and the code location
    - Console and file outputs
    """
    
    def __init__(self, log_base_path: str, console_output: bool = True):
        self.log_base_path = Path(log_base_path)
        self.console_output = console_output

        self.log_dir = self._create_log_dir()

        self.file_handlers = {}

        self.logger = self._setup_logger()
    
    def _create_log_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        log_dir = self.log_base_path / f"build_images/{timestamp}"

        log_dir.mkdir(parents=True, exist_ok=True)
        
        return log_dir
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("image_builder_logger")
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        log_format = (
            "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - "
            "%(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        formatter = logging.Formatter(log_format)

        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self._add_file_handlers(logger, formatter)
        
        return logger
    
    def _add_file_handlers(self, logger: logging.Logger, formatter: logging.Formatter):
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        
        for level_name, level_value in log_levels.items():
            log_file = self.log_dir / f"{level_name}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)

            if level_name == "debug":
                file_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            elif level_name == "info":
                file_handler.addFilter(lambda record: record.levelno == logging.INFO)
            elif level_name == "warning":
                file_handler.addFilter(lambda record: record.levelno == logging.WARNING)
            elif level_name == "error":
                file_handler.addFilter(lambda record: record.levelno == logging.ERROR)
            
            logger.addHandler(file_handler)
            self.file_handlers[level_name] = file_handler

        all_log_file = self.log_dir / "all.log"
        all_file_handler = logging.FileHandler(all_log_file, encoding="utf-8")
        all_file_handler.setFormatter(formatter)
        all_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(all_file_handler)
        self.file_handlers["all"] = all_file_handler
    
    def debug(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.error(message, *args, **kwargs)
    
    @property
    def log_file(self) -> str:
        return str(self.log_dir / "all.log")
    
    def get_log_dir(self) -> str:
        return str(self.log_dir.absolute())
    
    def get_log_files(self) -> dict:
        return {
            "debug": str(self.log_dir / "debug.log"),
            "info": str(self.log_dir / "info.log"),
            "warning": str(self.log_dir / "warning.log"),
            "error": str(self.log_dir / "error.log"),
            "all": str(self.log_dir / "all.log"),
        }
    
    def close(self):
        for handler in self.file_handlers.values():
            handler.close()

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __str__(self) -> str:
        return f"ImageBuilderLogger(dir={self.log_dir})"
    
    def __repr__(self) -> str:
        return (
            f"ImageBuilderLogger("
            f"base_path='{self.log_base_path}', "
            f"log_dir='{self.log_dir}', "
            f"console_output={self.console_output})"
        )

class Logger:
    """
    Log manager

    Function:
    - Create sub dir based on timestamp
    - Create debug.log, info.log, notice.log, warning.log and error.log
    - Provide standard log format
    - Console and file outputs
    """

    def __init__(
        self,
        log_base_path: str,
        logger_name: str = "tokfinity_logger",
        console_output: bool = True,
        log_format: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        self.log_base_path = Path(log_base_path)
        self.logger_name = logger_name
        self.console_output = console_output
        self.instance_id = instance_id

        self.log_format = log_format or (
            "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - "
            "%(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
        )

        self.log_dir = self._create_log_dir()

        self.file_handlers = {}

        self.logger = self._setup_logger()

    def _create_log_dir(self) -> Path:
        if self.instance_id:
            log_dir = self.log_base_path / self.instance_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            log_dir = self.log_base_path / timestamp

        log_dir.mkdir(parents=True, exist_ok=True)

        return log_dir

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)

        logger.handlers.clear()

        formatter = logging.Formatter(self.log_format)

        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self._add_file_handlers(logger, formatter)

        return logger

    def _add_file_handlers(self, logger: logging.Logger, formatter: logging.Formatter):
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "notice": NOTICE_LEVEL,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }

        for level_name, level_value in log_levels.items():
            log_file = self.log_dir / f"{level_name}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)

            if level_name == "debug":
                file_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            elif level_name == "info":
                file_handler.addFilter(lambda record: record.levelno == logging.INFO)
            elif level_name == "notice":
                file_handler.addFilter(lambda record: record.levelno == NOTICE_LEVEL)
            elif level_name == "warning":
                file_handler.addFilter(lambda record: record.levelno == logging.WARNING)
            elif level_name == "error":
                file_handler.addFilter(lambda record: record.levelno == logging.ERROR)

            logger.addHandler(file_handler)
            self.file_handlers[level_name] = file_handler

        all_log_file = self.log_dir / "all.log"
        all_file_handler = logging.FileHandler(all_log_file, encoding="utf-8")
        all_file_handler.setFormatter(formatter)
        all_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(all_file_handler)
        self.file_handlers["all"] = all_file_handler

    def debug(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.info(message, *args, **kwargs)

    def notice(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.log(NOTICE_LEVEL, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        self.logger.error(message, *args, **kwargs)

    def get_log_dir(self) -> str:
        return str(self.log_dir.absolute())

    def get_log_files(self) -> dict:
        return {
            "debug": str(self.log_dir / "debug.log"),
            "info": str(self.log_dir / "info.log"),
            "notice": str(self.log_dir / "notice.log"),
            "warning": str(self.log_dir / "warning.log"),
            "error": str(self.log_dir / "error.log"),
            "all": str(self.log_dir / "all.log"),
        }

    @property
    def log_file(self) -> str:
        return str(self.log_dir / "all.log")

    def close(self):
        for handler in self.file_handlers.values():
            handler.close()
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        return f"Logger(name={self.logger_name}, dir={self.log_dir})"

    def __repr__(self) -> str:
        return (
            f"Logger("
            f"name='{self.logger_name}', "
            f"base_path='{self.log_base_path}', "
            f"log_dir='{self.log_dir}', "
            f"console_output={self.console_output})"
        )


def create_logger(
    log_base_path: str,
    logger_name: str = "tokfinity_logger",
    console_output: bool = True,
) -> Logger:
    return Logger(
        log_base_path=log_base_path,
        logger_name=logger_name,
        console_output=console_output,
    )


# Global log manager instance (optional)
_global_logger: Optional[Logger] = None


def get_global_logger() -> Optional[Logger]:
    return _global_logger


def set_global_logger(logger: Logger):
    global _global_logger
    _global_logger = logger


def init_global_logger(
    log_base_path: str, logger_name: str = "global_logger"
) -> Logger:
    global _global_logger
    _global_logger = Logger(log_base_path, logger_name)
    return _global_logger
