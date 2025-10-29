"""Search tool for finding files based on text content using ripgrep (rg)."""

import asyncio
import json
from logging import Logger
import re
from pathlib import Path
from typing import Any, Dict, override
from traceback import format_exc

from src.tools.base import (
    Tool,
    ToolCallArguments,
    ToolError,
    ToolExecResult,
    ToolParameter,
    SubmitToolResult,
    SUBMIT_RESULT_TOOL_NAME,
)
from src.tools.run import run
from src.tools.executor import Executor


class SubmitResultTool(Tool):
    """Tool for git diff, not for model to invoke"""

    def __init__(
        self,
        model_provider: str | None = None,
        executor: Executor | None = None,
        logger: Logger | None = None,
        config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_provider, logger, config)
        self._executor = executor

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return SUBMIT_RESULT_TOOL_NAME

    @override
    def get_description(self) -> str:
        return """
Submit the final result to complete the task.

This tool should be called when you are confident that the issue has been resolved. Simply indicate that you are ready to submit the result - the system will automatically capture the git diff and generate the final patch.

You don't need to provide the actual patch content manually. Just call this tool to signal completion, and the system will handle the rest.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        params = [
            ToolParameter(
                name="is_task_done",
                type="boolean",
                description="Whether the task is done",
                required=True,
            ),
            ToolParameter(
                name="test_status",
                type="string",
                description="The status of test execution after applying the patch",
                required=True,
                enum=["passed", "failed", "skipped", "error"],
            ),
            ToolParameter(
                name="reasoning",
                type="string",
                description="Detailed explanation of the logic behind the patch, including root cause analysis and solution approach",
                required=True,
            ),
        ]

        return params

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the tool locally (not supported for submit_result tool)."""
        return ToolExecResult(
            error="SubmitResultTool only supports container execution", error_code=-1
        )

    @override
    async def container_execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if not self._executor:
            return ToolExecResult(
                error="No executor provided for git diff tool", error_code=-1
            )
        try:
            is_task_done = arguments.get("is_task_done", False)
            test_status = arguments.get("test_status", "error")
            reasoning = arguments.get("reasoning", "")
            root_path = self.config.get("builder", {}).get("repo_root_path", "/")
            cmd_parts = ["cd", str(root_path), "&&", "git", "--no-pager", "diff"]
            command = " ".join(cmd_parts)
            self.logger.debug(
                f"DEBUG: GitDiffTool executing command: {command}"
            )  # Debug output
            return_code, output = self._executor.execute_once(command)
            self.logger.debug(
                f"DEBUG: GitDiffTool result - Return code: {return_code}, Output: \n{output}"
            )  # Debug output

            if return_code == 0:
                submit_result = SubmitToolResult(
                    return_code=return_code,
                    output=output,
                    is_task_done=is_task_done,
                    test_status=test_status,
                    reasoning=reasoning,
                )
                return ToolExecResult(output=str(submit_result))
            else:
                return ToolExecResult(
                    error=f"GitDiffTool exited with code {return_code}. Output: {output}",
                    error_code=return_code,
                )

        except Exception as e:
            return ToolExecResult(
                error=f"Container search error: {str(e)}", error_code=-1
            )
