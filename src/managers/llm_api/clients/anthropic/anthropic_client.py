"""
Anthropic Claude Client Module
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from src.managers.log.logger import Logger

from src.managers.llm_api.base_client import (
    BaseLLMAPI,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    MessageRole,
    Choice,
    Usage,
)


class AnthropicClient(BaseLLMAPI):
    """
    Anthropic Claude API Client
    Anthropic Claude models supported
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        anthropic_version: str = "2023-06-01",
        logger: Optional[Logger] = None,
        **kwargs
    ):
        """
        Initialize Anthropic client

        Args:
            api_key: Anthropic API key
            base_url: API basic URL，default Anthropic API
            timeout: timeout in seconds
            max_retries: max retries
            retry_delay: retry delay in seconds
            anthropic_version: API version
            **kwargs: other args
        """
        self.anthropic_version = anthropic_version

        if base_url is None:
            base_url = "https://api.anthropic.com"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            **kwargs
        )

    def _initialize_client(self) -> None:
        """Initialize HTTP client"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "User-Agent": "tokfinity-llm-client/1.0",
        }

        self._create_http_clients(headers)

    def _get_chat_endpoint(self) -> str:
        return "/v1/messages"

    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Build Anthropic API request payload"""
        messages, system_prompt = self._convert_messages_to_anthropic_format(
            request.messages
        )

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if request.stop is not None:
            payload["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        if request.tools is not None:
            payload["tools"] = self._convert_tools_to_anthropic_format(request.tools)

        if request.tool_choice is not None:
            payload["tool_choice"] = self._convert_tool_choice_to_anthropic_format(
                request.tool_choice
            )

        return payload

    def _convert_messages_to_anthropic_format(
        self, messages: List[ChatMessage]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert messages to anthropic format"""
        anthropic_messages = []
        system_prompt = None

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                system_prompt = message.content
            elif message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                anthropic_messages.append(
                    {"role": message.role.value, "content": message.content}
                )

        return anthropic_messages, system_prompt

    def _convert_tools_to_anthropic_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI format tools to anthropic format"""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function_def = tool.get("function", {})
                anthropic_tool = {
                    "name": function_def.get("name", ""),
                    "description": function_def.get("description", ""),
                    "input_schema": function_def.get("parameters", {}),
                }
                anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _convert_tool_choice_to_anthropic_format(
        self, tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """Convert OpenAI format tool choice to anthropic format"""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return "auto"
            elif tool_choice == "none":
                return "none"
            else:
                return {"type": "tool", "name": tool_choice}
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                return {
                    "type": "tool",
                    "name": tool_choice.get("function", {}).get("name", ""),
                }
            elif tool_choice.get("type") == "tool":
                return tool_choice

        return "auto"

    def _convert_anthropic_tool_calls(
        self, content_list: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert anthropic tool calls to OpenAI format"""
        tool_calls = []

        for item in content_list:
            if item.get("type") == "tool_use":
                tool_call = {
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("input", {}),
                    },
                }
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None

    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        """Convert anthropic API response to OpenAI format"""
        content = ""
        tool_calls = None

        if response_data.get("content"):
            content_data = (
                response_data["content"][0] if response_data["content"] else {}
            )
            content = content_data.get("text", "")

            if content_data.get("type") == "tool_use":
                tool_calls = self._convert_anthropic_tool_calls(
                    response_data.get("content", [])
                )

        message = ChatMessage(
            role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=self._convert_stop_reason(response_data.get("stop_reason")),
        )

        usage_data = response_data.get("usage", {})
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )

        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            object="chat.completion",
            created=int(time.time()),
            model=response_data.get("model", ""),
            choices=[choice],
            usage=usage,
        )

    def _convert_stop_reason(self, stop_reason: Optional[str]) -> Optional[str]:
        """Convert anthropic stop reason to OpenAI format"""
        if stop_reason == "end_turn":
            return "stop"
        elif stop_reason == "max_tokens":
            return "length"
        elif stop_reason == "stop_sequence":
            return "stop"
        else:
            return stop_reason

    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        """Convert anthropic steam response to OpenAI format"""
        event_type = chunk_data.get("type")

        if event_type == "content_block_delta":
            delta = chunk_data.get("delta", {})
            text = delta.get("text", "")

            if text:
                choices = [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ]

                return ChatCompletionChunk(
                    id=chunk_data.get("message", {}).get("id", ""),
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=chunk_data.get("message", {}).get("model", ""),
                    choices=choices,
                )

        elif event_type == "message_stop":
            choices = [{"index": 0, "delta": {}, "finish_reason": "stop"}]

            return ChatCompletionChunk(
                id=chunk_data.get("message", {}).get("id", ""),
                object="chat.completion.chunk",
                created=int(time.time()),
                model=chunk_data.get("message", {}).get("model", ""),
                choices=choices,
            )

        return None

    def _parse_stream_line(self, line: str) -> Optional[ChatCompletionChunk]:
        try:
            chunk_data = json.loads(line)
            return self._parse_stream_chunk(chunk_data)
        except json.JSONDecodeError:
            # if not json, try SSE
            return super()._parse_stream_line(line)
