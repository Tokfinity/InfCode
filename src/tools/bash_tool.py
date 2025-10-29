# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# Copyright (c) 2025 Beijing Tokens Infinity Technology Co., Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by Beijing Tokens Infinity Technology Co., Ltd. and/or its affiliates. on 27 Oct 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
# and https://github.com/bytedance/trae-agent/blob/main/LICENSE
#
# This modified file is released under the same license.


import asyncio
import os
from typing import override

from src.tools.base import (
    Tool,
    ToolCallArguments,
    ToolError,
    ToolExecResult,
    ToolParameter,
    BASH_TOOL_NAME,
)
from src.tools.executor import Executor
from src.managers.log.logger import Logger
from typing import Dict, Any
from traceback import format_exc


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = (
        ",,,,bash-command-exit-__ERROR_CODE__-banner,,,,"  # `__ERROR_CODE__` will be replaced by `$?` or `!errorlevel!` later
    )

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._started:
            return

        # Windows compatibility: os.setsid not available

        if os.name != "nt":  # Unix-like systems
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /v:on",  # enable delayed expansion to allow `echo !errorlevel!`
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

    async def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process is None:
            return
        if self._process.returncode is not None:
            return
        self._process.terminate()

        # Wait until the process has truly terminated.
        stdout, stderr = await self._process.communicate()

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the bash shell."""
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        error_code = 0

        sentinel_before, pivot, sentinel_after = self._sentinel.partition(
            "__ERROR_CODE__"
        )
        assert pivot == "__ERROR_CODE__"

        errcode_retriever = "!errorlevel!" if os.name == "nt" else "$?"
        command_sep = "&" if os.name == "nt" else ";"

        # send command to the process
        self._process.stdin.write(
            b"(\n"
            + command.encode()
            + f"\n){command_sep} echo {self._sentinel.replace('__ERROR_CODE__', errcode_retriever)}\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output: str = self._process.stdout._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
                    if sentinel_before in output:
                        # strip the sentinel from output
                        output, pivot, exit_banner = output.rpartition(sentinel_before)
                        assert pivot

                        # get error code inside banner
                        error_code_str, pivot, _ = exit_banner.partition(sentinel_after)
                        if not pivot or not error_code_str.isdecimal():
                            continue

                        error_code = int(error_code_str)
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            output = output[:-1]  # pyright: ignore[reportUnknownVariableType]

        error: str = self._process.stderr._buffer.decode()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        if error.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            error = error[:-1]  # pyright: ignore[reportUnknownVariableType]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined] # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        return ToolExecResult(
            output=output, error=error, error_code=error_code
        )  # pyright: ignore[reportUnknownArgumentType]


class BashTool(Tool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    def __init__(
        self,
        model_provider: str | None = None,
        executor: Executor | None = None,
        logger: Logger | None = None,
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(model_provider, logger, config)
        self._session: _BashSession | None = None
        self.executor = executor

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return BASH_TOOL_NAME

    @override
    def get_description(self) -> str:
        return """Execute commands within a bash shell environment, either on the local system or inside a container.
* When providing the "command" parameter, its contents must be provided as-is without any XML escaping.
* You have access to a mirrored repository of common Linux (via apt) and Python (via pip) packages for installation.
* State is persisted across all command executions and throughout our conversation session.
* Avoid executing commands that are likely to generate excessively large outputs.
* Avoid executing interactive commands that require user input (e.g., password prompts, confirmation messages).
* For Git commands, always prefer non-interactive forms. For example, use git --no-pager diff instead of git diff to prevent opening a pager.
* To inspect a specific range of lines in a file (e.g., lines 5-10), you can use a command like: sed -n '5,10p' /path/to/file
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="command",
                type="string",
                description="The exact bash command string to be executed.",
                required=True,
            ),
            ToolParameter(
                name="restart",
                type="boolean",
                description="If true, terminates the current shell session and starts a new one before executing the command. This clears the session state.",
                required=restart_required,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if arguments.get("restart"):
            if self._session:
                await self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolExecResult(output="tool has been restarted.")

        if self._session is None:
            try:
                self._session = _BashSession()
                await self._session.start()
            except Exception as e:
                return ToolExecResult(
                    error=f"Error starting bash session: {e}",
                    error_code=-1,
                )

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        try:
            return await self._session.run(command)
        except Exception as e:
            return ToolExecResult(
                error=f"Error running bash command: {e}",
                error_code=-1,
            )

    async def container_execute(
        self, arguments: ToolCallArguments, session_id: str = "0"
    ) -> ToolExecResult:
        """Execute a command in a container bash shell."""
        if not self.executor:
            return ToolExecResult(
                error="Container execution requires an executor to be provided during tool initialization",
                error_code=-1,
            )

        if arguments.get("restart"):
            # Close the existing session if it exists
            self.executor.close_session("0")
            # The executor will automatically recreate session '0' when needed
            return ToolExecResult(output="Container session has been restarted.")

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for container execution",
                error_code=-1,
            )
        # command_with_init = f"source /opt/miniconda3/bin/activate && conda activate testbed && {command}"
        # Check if the session is alive before executing the command
        if not self.executor.check_session():
            return ToolExecResult(
                error="Container session is not alive and could not be restarted",
                error_code=-1,
            )

        try:
            return_code, output = self.executor.execute(session_id, command)
            # return_code, output = self.executor.execute_once(command_with_init)

            # The executor returns (return_code, output) tuple
            # We'll treat any non-zero return code as an error
            error = None
            if return_code != 0:
                error = f"Command failed with exit code {return_code}, output: {output}"

            return ToolExecResult(output=output, error=error, error_code=return_code)
        except Exception as e:
            return ToolExecResult(
                error=f"Error running container bash command: {e}", error_code=-1
            )

    @override
    async def close(self):
        """Properly close self._process."""
        if self._session:
            await self._session.stop()
            self._session = None
