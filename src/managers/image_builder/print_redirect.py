"""
Redirect print output of a third party repo to a specific log
"""
import sys
import logging
import re
from typing import Union, Set, Pattern
from typing import Optional, TextIO
from pathlib import Path
from src.managers.log.logger import Logger as CustomLogger


class StreamWrapper:
    """Simple stream wrapper，for redirecting stdout/stderr"""
    
    def __init__(self, original_stream, write_func):
        self.original_stream = original_stream
        self.write_func = write_func
        self.buffer = ""
        
    def write(self, text):
        self.buffer += text

        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():
                self.write_func(line + '\n')
        
        return len(text)
        
    def flush(self):
        if self.buffer:
            if self.buffer.strip():
                self.write_func(self.buffer)
            self.buffer = ""
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()
            
    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class PrintRedirector:
    """Redirect print output of a third party repo to a specific log"""

    TRACEBACK_START_PATTERN = re.compile(r'Traceback\s*\(most recent call last\):', re.IGNORECASE)
    EXCEPTION_END_PATTERN = re.compile(
        r'\b(Error|Exception|KeyError|ValueError|TypeError|AttributeError|'
        r'ImportError|BuildError|DockerError|IndentationError|SyntaxError|'
        r'RuntimeError|OSError|FileNotFoundError|PermissionError)\s*:',
        re.IGNORECASE
    )

    ERROR_KEYWORDS = {'error', 'failed', 'exception', 'fatal', 'critical'}
    WARNING_KEYWORDS = {'warning', 'warn', 'deprecated'}
    SKIP_KEYWORDS = {'skipping', 'skip', 'ignoring', 'ignore'}
    
    def __init__(self, logger: Union[logging.Logger, CustomLogger]):
        self.logger = logger
        self.original_print = None
        self.original_stdout = None
        self.original_stderr = None

        self.traceback_buffer = []
        self.in_traceback = False
        
    def __enter__(self):
        self.start_redirect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_redirect()
        
    def start_redirect(self):
        if isinstance(__builtins__, dict):
            self.original_print = __builtins__['print']
        else:
            self.original_print = getattr(__builtins__, 'print')
        if isinstance(__builtins__, dict):
            __builtins__['print'] = self._redirected_print
        else:
            setattr(__builtins__, 'print', self._redirected_print)
        
    def stop_redirect(self):
        if self.original_print:
            if isinstance(__builtins__, dict):
                __builtins__['print'] = self.original_print
            else:
                setattr(__builtins__, 'print', self.original_print)
            
    def _redirected_print(self, *args, **kwargs):
        if not args:
            return
        output = ' '.join(str(arg) for arg in args)

        if self.TRACEBACK_START_PATTERN.search(output):
            self.in_traceback = True
            self.traceback_buffer = [output]
            return

        if self.in_traceback:
            self.traceback_buffer.append(output)

            if self.EXCEPTION_END_PATTERN.search(output):
                self._log_traceback_and_reset()
            return

        self._log_by_level(output)
    
    def _log_traceback_and_reset(self):
        full_traceback = '\n'.join(self.traceback_buffer)
        self.logger.error(f"[Third party repo exception stack]\n{full_traceback}")

        self.in_traceback = False
        self.traceback_buffer.clear()
    
    def _log_by_level(self, output: str):
        output_lower = output.lower()

        if any(keyword in output_lower for keyword in self.ERROR_KEYWORDS):
            self.logger.error(f"[Third party] {output}")
        elif any(keyword in output_lower for keyword in self.WARNING_KEYWORDS):
            self.logger.warning(f"[Third party] {output}")
        elif any(keyword in output_lower for keyword in self.SKIP_KEYWORDS):
            self.logger.info(f"[Third party] {output}")
        else:
            self.logger.info(f"[Third party] {output}")


def redirect_swebench_prints(logger: Union[logging.Logger, CustomLogger]):
    return PrintRedirector(logger)
    