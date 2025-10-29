from typing import Any, Dict, List
import json
from traceback import format_exc
from src.managers.log.logger import Logger
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.prompts.prompts_manager import PromptsManager
from src.tools.base import (
    ToolExecutor,
    BASH_TOOL_NAME,
    STR_REPLACE_BASED_EDIT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SUBMIT_RESULT_TOOL_NAME,
)
from src.managers.loop.types import ToolStats, LLMUsage



class BaseLoop:
    def __init__(self, instance_id: str, instance_data: Dict[str, Any], logger: Logger, prompts_manager: PromptsManager | None, llm_manager: LLMAPIManager | None, tool_executor: ToolExecutor, config: Dict[str, Any] | None = None):
        self.instance_id = instance_id
        self.instance_data = instance_data
        self.logger = logger
        self.prompts_manager = prompts_manager
        self.llm_manager = llm_manager
        self.tool_executor = tool_executor
        self.config = config or {}
        self.component_name = self.__class__.__name__
    

    def _make_assistant(
        self, content: str | None, tool_calls: Any, messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Construct an assistant message based on the current content and tool calls, and append it to the messages.
        """
        safe_content = content or ""
        if not safe_content and not tool_calls:
            self.logger.warning(
                f"[{self.component_name}] Assistant returned an empty message with no tool calls; skipping this message and prompting to continue"
            )
            messages.append(
                {"role": "user", "content": "请继续分析问题并使用工具来解决问题。"}
            )
            return False
        assistant_message: Dict[str, Any] = {"role": "assistant"}
        if tool_calls and not safe_content:
            assistant_message["content"] = ""
        elif safe_content:
            assistant_message["content"] = safe_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        return True
    
    def _make_tool_response(
        self, tool_results: List[Any], messages: List[Dict[str, Any]]
    ) -> None:
        """Convert tool execution results into standard tool messages (role=tool) and append them to the messages.

        - Generate content per result: use prompts_manager.tool_response_prompts([{...}]) to produce the content
        - Set tool_call_id: prefer ToolResult.id; fallback to ToolResult.call_id
        """
        if not tool_results:
            return
        for result in tool_results:
            single_dict = [
                {
                    "name": getattr(result, "name", "unknown"),
                    "success": getattr(result, "success", False),
                    "result": getattr(result, "result", None) or "",
                    "error": getattr(result, "error", None) or "",
                }
            ]
            content_text = (
                self.prompts_manager.tool_response_prompts(single_dict)
                if self.prompts_manager
                else ""
            )
            tool_call_id = getattr(result, "id", None) or getattr(
                result, "call_id", None
            )
            messages.append(
                {
                    "role": "tool",
                    "content": content_text,
                    "tool_call_id": tool_call_id,
                }
            )
    
    def _response_log(
        self, response: Any, first_content: str, first_tool_calls: Any, total_turns: int
    ) -> None:
        """notice log for the current turn's LLM output"""
        try:
            response_log: Dict[str, Any] = {}
            if hasattr(response, "usage") and response.usage:
                response_log["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            if hasattr(response, "choices") and response.choices:
                response_log["choice"] = {
                    "message": {
                        "content": first_content,
                        "tool_calls": first_tool_calls,
                    }
                }
            if response_log:
                self.logger.notice(
                    f"[{self.component_name}] The {total_turns}th turn output: {json.dumps(response_log, ensure_ascii=False)}"
                )
            else:
                self.logger.notice(
                    f"[{self.component_name}] The {total_turns}th turn output: {str(response)}"
                )
        except Exception:
            self.logger.notice(
                f"[{self.component_name}] 第 {total_turns} 轮: LLM 输出序列化失败，使用字符串表示: {str(response)}, traceback: {format_exc()}."
            )
    
    def _debug_messages(
        self, turn: int, messages: List[Dict[str, Any]], prefix_len: int = 300
    ) -> None:
        """debug log for the messages to be sent to the model"""
        try:
            self.logger.debug(f"[{self.component_name}] msg:")
            recent_messages = messages[-2:] if len(messages) > 2 else messages
            base_index = len(messages) - len(recent_messages)
            for offset, msg in enumerate[Dict[str, Any]](recent_messages):
                idx = base_index + offset
                role = msg.get("role")
                content = msg.get("content")
                content_str = content if isinstance(content, str) else ""
                preview = content_str[:prefix_len]
                content_len = len(content_str)
                extra = ""
                if role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    has_tool = tool_calls is not None and tool_calls != []
                    try:
                        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False)
                    except Exception:
                        self.logger.warning(
                            f"[{self.component_name}] In debug_messages function, fail: {format_exc()}, tool calls: {tool_calls}."
                        )
                        tool_calls_json = str(tool_calls)
                    extra = f", has_tool_calls={has_tool}, tool_calls={tool_calls_json}"
                elif role == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    extra = f", tool_call_id={tool_call_id}"
                self.logger.debug(
                    f"[{self.component_name}] {turn+1}th, msg#{idx}: role={role}, content_len={content_len}, content_preview={json.dumps(preview, ensure_ascii=False)}{extra}"
                )
        except Exception:
            self.logger.warning(
                f"[{self.component_name}] In debug_messages function, fail msg: {format_exc()}."
            )

    def _debug_last_message(
        self, turn: int, messages: List[Dict[str, Any]], prefix_len: int = 300
    ) -> None:
        """debug last turn msg"""
        try:
            if not messages:
                return
            last_assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break
            if last_assistant_idx is None:
                return
            msg = messages[last_assistant_idx]
            content = msg.get("content")
            content_str = content if isinstance(content, str) else ""
            preview = content_str[:prefix_len]
            content_len = len(content_str)
            tool_calls = msg.get("tool_calls")
            has_tool = tool_calls is not None and tool_calls != []
            try:
                tool_calls_json = json.dumps(tool_calls, ensure_ascii=False)
            except Exception:
                self.logger.warning(
                    f"[{self.component_name}] In debug_last_message function, fail: {format_exc()}, tool calls: {tool_calls}."
                )
                tool_calls_json = str(tool_calls)
            self.logger.debug(
                f"[{self.component_name}] {turn+1}th turn, output_preview: role=assistant, content_len={content_len}, content_preview={json.dumps(preview, ensure_ascii=False)}, has_tool_calls={has_tool}, tool_calls={tool_calls_json}"
            )
        except Exception:
            self.logger.warning(
                f"[{self.component_name}] In debug_last_message function, last turn fail: {format_exc()}."
            )
    
    def _debug_tools(self, tools: List[Dict[str, Any]]) -> None:
        """debug tools msg"""
        try:
            self.logger.debug(f"[{self.component_name}] tools num: {len(tools)}")
            for i, tool in enumerate(tools):
                try:
                    tool_json = json.dumps(tool, ensure_ascii=False)
                    self.logger.debug(f"[{self.component_name}] tool #{i+1}: {tool_json}")
                except Exception:
                    self.logger.debug(
                        f"[{self.component_name}] tool #{i+1} fail: {format_exc()}, string: {str(tool)}."
                    )
        except Exception:
            try:
                self.logger.warning(
                    f"[{self.component_name}] fail; traceback: {format_exc()}."
                )
                self.logger.warning(f"[{self.component_name}] tools string: {str(tools)}")
            except Exception:
                pass
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        pass
    
    def _is_bash_tool(self, tool_name: str) -> bool:
        return BASH_TOOL_NAME in tool_name
    
    def _is_edit_tool(self, tool_name: str) -> bool:
        return "edit" in tool_name or "str_replace" in tool_name or STR_REPLACE_BASED_EDIT_TOOL_NAME in tool_name
    
    def _is_search_tool(self, tool_name: str) -> bool:
        return SEARCH_TOOL_NAME in tool_name or "search" in tool_name
    
    def _is_submit_result_tool(self, tool_name: str) -> bool:
        return SUBMIT_RESULT_TOOL_NAME in tool_name

    def _update_usage(self, response: Any, usage_stats: LLMUsage) -> None:
        if hasattr(response, "usage") and response.usage:
            usage_stats.prompt_tokens += int(getattr(response.usage, "prompt_tokens", 0) or 0)
            usage_stats.completion_tokens += int(
                getattr(response.usage, "completion_tokens", 0) or 0
            )
            usage_stats.total_tokens += int(getattr(response.usage, "total_tokens", 0) or 0)

    def _init_usage_stats(self) -> LLMUsage:
        return LLMUsage()

    def _init_tools_stats(self) -> ToolStats:
        return ToolStats()

    def _update_tool_call_statistic(
        self, tool_results: List[Any], tool_stats: ToolStats
    ) -> None:
        for result in tool_results:
            try:
                tool_name = getattr(result, "name", "")
                tool_name = tool_name.lower() if isinstance(tool_name, str) else ""
                success = bool(getattr(result, "success", False))

                if self._is_bash_tool(tool_name):
                    tool_stats.bash["count"] += 1
                    if not success:
                        tool_stats.bash["failed"] += 1

                elif self._is_edit_tool(tool_name):
                    tool_stats.edit["count"] += 1
                    if not success:
                        tool_stats.edit["failed"] += 1

                elif self._is_search_tool(tool_name):
                    tool_stats.search["count"] += 1
                    if not success:
                        tool_stats.search["failed"] += 1

                elif self._is_submit_result_tool(tool_name):
                    tool_stats.submit_result["count"] += 1
                    if not success:
                        tool_stats.submit_result["failed"] += 1
            except Exception:
                continue