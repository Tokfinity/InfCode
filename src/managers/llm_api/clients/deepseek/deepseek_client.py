"""
DeepSeek Client Module
"""

import time
from typing import Dict, List, Any, Optional
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


class DeepSeekClient(BaseLLMAPI):
    """
    DeepSeek API Client

    DeepSeek models supported, including DeepSeek-Coder、DeepSeek-Chat etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        if base_url is None:
            base_url = "https://api.deepseek.com/v1"

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            **kwargs,
        )

    def _initialize_client(self) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "tokfinity-llm-client/1.0",
        }

        self._create_http_clients(headers)

    def _get_chat_endpoint(self) -> str:
        return "/chat/completions"

    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }

        # optional
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.stop is not None:
            payload["stop"] = request.stop

        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty

        if request.tools is not None:
            payload["tools"] = request.tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.response_format is not None:
            payload["response_format"] = request.response_format

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.user is not None:
            payload["user"] = request.user

        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})

            tool_calls = message_data.get("tool_calls")

            message = ChatMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
                tool_calls=tool_calls,
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)

        usage_data = response_data.get("usage", {})
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=response_data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=response_data.get("system_fingerprint"),
        )

    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        """Parse stream chunk"""
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", ""),
            choices=chunk_data.get("choices", []),
            system_fingerprint=chunk_data.get("system_fingerprint"),
        )

    def list_models(self) -> Dict[str, Any]:
        """Get available models"""
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()

    async def alist_models(self) -> Dict[str, Any]:
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()
