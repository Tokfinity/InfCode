"""Tools module for Code Agent."""

from src.tools.base import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolResult,
    BASH_TOOL_NAME,
    STR_REPLACE_BASED_EDIT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SUBMIT_RESULT_TOOL_NAME,
)
from src.tools.bash_tool import BashTool
from src.tools.edit_tool import TextEditorTool
from src.tools.search_tool import SearchTool
from src.tools.submit_result_tool import SubmitResultTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "TextEditorTool",
    "JSONEditTool",
    "SearchTool",
    "SubmitResultTool",
]

tools_registry: dict[str, type[Tool]] = {
    BASH_TOOL_NAME: BashTool,
    STR_REPLACE_BASED_EDIT_TOOL_NAME: TextEditorTool,
    SEARCH_TOOL_NAME: SearchTool,
    SUBMIT_RESULT_TOOL_NAME: SubmitResultTool,
}
