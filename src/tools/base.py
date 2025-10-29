# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# Copyright (c) 2025 Beijing Tokens Infinity Technology Co., Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by Beijing Tokens Infinity Technology Co., Ltd. and/or its affiliates. on 27 Oct 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/bytedance/trae-agent/blob/main/LICENSE
#
# This modified file is released under the same license.


"""Base classes for tools and tool calling."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import override
from src.managers.log.logger import Logger
from typing import Dict, Any
from traceback import format_exc
from pathlib import Path

ParamSchemaValue = str | list[str] | bool | dict[str, object]
Property = dict[str, ParamSchemaValue]

BASH_TOOL_NAME = "bash"
STR_REPLACE_BASED_EDIT_TOOL_NAME = "str_replace_based_edit_tool"
SEARCH_TOOL_NAME = "search_tool"
SUBMIT_RESULT_TOOL_NAME = "submit_result"


class ToolError(Exception):
    """Base class for tool errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message: str = message


@dataclass
class ToolExecResult:
    """Intermediate result of a tool execution."""

    output: str | None = None
    error: str | None = None
    error_code: int = 0


@dataclass
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    name: str  # Gemini specific field
    success: bool
    result: str | None = None
    error: str | None = None
    id: str | None = None  # OpenAI-specific field


@dataclass
class SubmitToolResult:
    """Structured result for submit_result tool."""

    return_code: int
    output: str
    is_task_done: bool
    test_status: str
    reasoning: str

    def __str__(self) -> str:
        """Convert to JSON string for output."""
        import json

        return json.dumps(
            {
                "return_code": self.return_code,
                "output": self.output,
                "is_task_done": self.is_task_done,
                "test_status": self.test_status,
                "reasoning": self.reasoning,
            }
        )

    @classmethod
    def from_string(cls, json_str: str) -> "SubmitToolResult":
        """Create SubmitToolResult from JSON string."""
        import json

        data = json.loads(json_str)
        return cls(
            return_code=data.get("return_code", 0),
            output=data.get("output", ""),
            is_task_done=data.get("is_task_done", False),
            test_status=data.get("test_status", "error"),
            reasoning=data.get("reasoning", ""),
        )


ToolCallArguments = dict[
    str, str | int | float | dict[str, object] | list[object] | None
]


@dataclass
class ToolCall:
    """Represents a parsed tool call."""

    name: str
    call_id: str
    arguments: ToolCallArguments = field(default_factory=dict)
    id: str | None = None

    @override
    def __str__(self) -> str:
        return f"ToolCall(name={self.name}, arguments={self.arguments}, call_id={self.call_id}, id={self.id})"


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    type: str | list[str]
    description: str
    enum: list[str] | None = None
    items: dict[str, object] | None = None
    required: bool = True


class Tool(ABC):
    """Base class for all tools."""

    def __init__(
        self,
        model_provider: str | None = None,
        logger: Logger | None = None,
        config: Dict[str, Any] | None = None,
    ):
        self._model_provider = model_provider
        self.logger = logger
        self.config = config

    @cached_property
    def model_provider(self) -> str | None:
        return self.get_model_provider()

    @cached_property
    def name(self) -> str:
        return self.get_name()

    @cached_property
    def description(self) -> str:
        return self.get_description()

    @cached_property
    def parameters(self) -> list[ToolParameter]:
        return self.get_parameters()

    def get_model_provider(self) -> str | None:
        """Get the model provider."""
        return self._model_provider

    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get the tool parameters."""
        pass

    @abstractmethod
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the tool with given parameters."""
        pass

    # Optional container execution hooks (to be overridden by tools that support containers)
    async def container_execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the tool inside a container shell (optional)."""
        raise ToolError(
            f"Tool '{self.get_name()}' does not support container execution"
        )

    def container_search(
        self, arguments: ToolCallArguments, session_id: str = "0"
    ) -> ToolExecResult:
        """Execute a search-like operation inside container (optional)."""
        raise ToolError(f"Tool '{self.get_name()}' does not support container search")

    # Optional container file editing hooks used by edit tools
    def container_read_file(self, path) -> str:
        """Read a file inside container (optional)."""
        raise ToolError(
            f"Tool '{self.get_name()}' does not support container_read_file"
        )

    def container_write_file(self, path, content: str) -> None:
        """Write a file inside container (optional)."""
        raise ToolError(
            f"Tool '{self.get_name()}' does not support container_write_file"
        )

    def container_str_replace(
        self, path, old_str: str, new_str: str | None
    ) -> ToolExecResult:
        """String replace inside a file in container (optional)."""
        raise ToolError(
            f"Tool '{self.get_name()}' does not support container_str_replace"
        )

    def container_insert(self, path, insert_line: int, new_str: str) -> ToolExecResult:
        """Insert text into a file in container (optional)."""
        raise ToolError(f"Tool '{self.get_name()}' does not support container_insert")
    
    def view_handler_container(
        self, arguments: ToolCallArguments, path: Path
    ) -> ToolExecResult:
        """View handler in container (optional)."""
        raise ToolError(f"Tool '{self.get_name()}' does not support view_handler_container")


    def json_definition(self) -> dict[str, object]:
        """Default return Claude format (backward compatibility)"""
        return self._definition_for_claude_fmt()

    def _definition_for_claude_fmt(self) -> dict[str, object]:
        """Return Claude format tool definition (Anthropic Messages API)"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema(),
        }

    def _definition_for_openai_fmt(self) -> dict[str, object]:
        """Return OpenAI format tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_input_schema(),
            },
        }

    def get_input_schema(self) -> dict[str, object]:
        """Get the input schema for the tool."""
        schema: dict[str, object] = {
            "type": "object",
        }

        properties: dict[str, Property] = {}
        required: list[str] = []

        for param in self.parameters:
            param_schema: Property = {
                "type": param.type,
                "description": param.description,
            }

            # For OpenAI strict mode, all params must be in 'required'.
            # Optional params are made "nullable" to be compliant.
            if self.model_provider == "openai":
                required.append(param.name)
                if not param.required:
                    current_type = param_schema["type"]
                    if isinstance(current_type, str):
                        param_schema["type"] = [current_type, "null"]
                    elif isinstance(current_type, list) and "null" not in current_type:
                        param_schema["type"] = list(current_type) + ["null"]
            elif param.required:
                required.append(param.name)

            if param.enum:
                param_schema["enum"] = param.enum

            if param.items:
                param_schema["items"] = param.items

            # For OpenAI, nested objects also need additionalProperties: false
            if self.model_provider == "openai" and param.type == "object":
                param_schema["additionalProperties"] = False

            properties[param.name] = param_schema

        schema["properties"] = properties
        if len(required) > 0:
            schema["required"] = required

        # For OpenAI, the top-level schema needs additionalProperties: false
        if self.model_provider == "openai":
            schema["additionalProperties"] = False

        return schema

    async def close(self):
        """Ensure proper tool resource deallocation before task completion."""
        return None  # Using "pass" will trigger a Ruff check error: B027


class ToolExecutor:
    """Tool executor that manages tool execution."""

    def __init__(self, tools: list[Tool], logger: Logger | None = None):
        self._tools = tools
        self._tool_map: dict[str, Tool] | None = None
        self.logger = logger

    async def close_tools(self):
        """Ensure all tool resources are properly released."""
        tasks = [tool.close() for tool in self._tools if hasattr(tool, "close")]
        res = await asyncio.gather(*tasks)
        return res

    def _normalize_name(self, name: str) -> str:
        """Normalize tool name by making it lowercase and removing underscores."""
        return name.lower().replace("_", "")

    @property
    def tools(self) -> dict[str, Tool]:
        if self._tool_map is None:
            self._tool_map = {
                self._normalize_name(tool.name): tool for tool in self._tools
            }
        return self._tool_map

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call locally."""
        normalized_name = self._normalize_name(tool_call.name)
        if normalized_name not in self.tools:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available tools: {[tool.name for tool in self._tools]}",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

        tool = self.tools[normalized_name]

        try:
            tool_exec_result = await tool.execute(tool_call.arguments)
            return ToolResult(
                name=tool_call.name,
                success=tool_exec_result.error_code == 0,
                result=tool_exec_result.output,
                error=tool_exec_result.error,
                call_id=tool_call.call_id,
                id=tool_call.id,
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Error executing tool '{tool_call.name}': {str(e)}, traceback: {format_exc()}.",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

    async def container_execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call in container."""
        normalized_name = self._normalize_name(tool_call.name)
        if normalized_name not in self.tools:
            self.logger.warning(
                f"[ToolExecutor] '{tool_call.name}' not found. Available tools: {[tool.name for tool in self._tools]}"
            )
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Tool '{tool_call.name}' not found. Available tools: {[tool.name for tool in self._tools]}",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

        tool = self.tools[normalized_name]

        try:
            tool_exec_result = await self._container_execute_tool_by_name(
                tool, tool_call
            )
            return ToolResult(
                name=tool_call.name,
                success=tool_exec_result.error_code == 0,
                result=tool_exec_result.output,
                error=tool_exec_result.error,
                call_id=tool_call.call_id,
                id=tool_call.id,
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                success=False,
                error=f"Error executing tool '{tool_call.name}': {str(e)}, traceback: {format_exc()}.",
                call_id=tool_call.call_id,
                id=tool_call.id,
            )

    async def _container_execute_tool_by_name(
        self, tool: Tool, tool_call: ToolCall
    ) -> ToolExecResult:
        tool_name = tool.get_name()

        if tool_name == BASH_TOOL_NAME:
            # BashTool: execute through container
            if hasattr(tool, "container_execute"):
                return await tool.container_execute(tool_call.arguments)
            else:
                raise ToolError(
                    f"Tool '{tool_name}' does not support container execution"
                )

        elif tool_name == STR_REPLACE_BASED_EDIT_TOOL_NAME:
            # TextEditorTool: execute through container
            if hasattr(tool, "container_read_file"):
                return await self._execute_edit_tool_in_container(
                    tool, tool_call.arguments
                )
            else:
                raise ToolError(
                    f"Tool '{tool_name}' does not support container execution"
                )

        elif tool_name == SEARCH_TOOL_NAME:
            # SearchTool: execute through container
            if hasattr(tool, "container_search"):
                return tool.container_search(tool_call.arguments)
            else:
                raise ToolError(
                    f"Tool '{tool_name}' does not support container execution"
                )
        elif tool_name == SUBMIT_RESULT_TOOL_NAME:
            # SubmitResultTool: execute through container
            if hasattr(tool, "container_execute"):
                return await tool.container_execute(tool_call.arguments)
            else:
                raise ToolError(
                    f"Tool '{tool_name}' does not support container execution"
                )
        else:
            # Other tools：container execution not supported
            raise ToolError(f"Tool '{tool_name}' does not support container execution")

    async def _execute_edit_tool_in_container(
        self, tool: Tool, arguments: ToolCallArguments
    ) -> ToolExecResult:
        command = str(arguments.get("command", ""))
        path_str = str(arguments.get("path", ""))

        if not path_str:
            return ToolExecResult(
                error="No path provided for the edit tool", error_code=-1
            )

        from pathlib import Path

        path = Path(path_str)

        try:
            if command == "view":
                return tool.view_handler_container(arguments, path)
                #return ToolExecResult(output=tool._make_output(content, str(path)))

            elif command == "create":
                file_text = str(arguments.get("file_text", ""))
                tool.container_write_file(path, file_text)
                return ToolExecResult(output=f"File created successfully at: {path}")

            elif command == "str_replace":
                old_str = str(arguments.get("old_str", ""))
                new_str = arguments.get("new_str")
                if new_str is not None:
                    new_str = str(new_str)
                return tool.container_str_replace(path, old_str, new_str)

            elif command == "insert":
                insert_line = int(arguments.get("insert_line", 0))
                new_str = str(arguments.get("new_str", ""))
                return tool.container_insert(path, insert_line, new_str)

            else:
                return ToolExecResult(
                    error=f"Unsupported command '{command}' for container execution",
                    error_code=-1,
                )

        except Exception as e:
            return ToolExecResult(
                error=f"Container edit tool error: {str(e)}.", error_code=-1
            )

    async def parallel_tool_call(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in parallel locally"""
        return await asyncio.gather(
            *[self.execute_tool_call(call) for call in tool_calls]
        )

    async def sequential_tool_call(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResult]:
        """Execute tool calls in sequential locally"""
        return [await self.execute_tool_call(call) for call in tool_calls]

    async def container_parallel_tool_call(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResult]:
        """Execute tool calls in parallel in container"""
        return await asyncio.gather(
            *[self.container_execute_tool_call(call) for call in tool_calls]
        )

    async def container_sequential_tool_call(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolResult]:
        """Execute tool calls in sequential in container"""
        return [await self.container_execute_tool_call(call) for call in tool_calls]
