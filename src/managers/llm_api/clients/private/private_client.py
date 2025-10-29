"""
Private model client
Implement of private model API based on the BaseLLMAPI
vLLM, Text Generation Inference, Ollama supported
"""

import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Generator
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


class PrivateModelClient(BaseLLMAPI):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        deployment_type: str = "vllm",
        custom_headers: Optional[Dict[str, str]] = None,
        supports_tools: bool = True,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        self.deployment_type = deployment_type.lower()
        self.custom_headers = custom_headers or {}
        self.supports_tools = supports_tools
        super().__init__(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            logger=logger,
            **kwargs,
        )

    def _initialize_client(self) -> None:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "tokfinity-llm-client/1.0",
        }

        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        headers.update(self.custom_headers)


        if self.deployment_type == "ollama":
            pass
        elif self.deployment_type == "tgi":
            pass

        self._create_http_clients(headers)

    def _get_chat_endpoint(self) -> str:
        if self.deployment_type == "ollama":
            return "/api/chat"
        elif self.deployment_type == "tgi":
            return "/generate"
        else:
            return "/chat/completions"

    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        if self.deployment_type == "ollama":
            return self._build_ollama_payload(request)
        elif self.deployment_type == "tgi":
            return self._build_tgi_payload(request)
        else:
            return self._build_openai_payload(request)

    def _build_openai_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }

        # Optional
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.stop is not None:
            payload["stop"] = request.stop

        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty

        if self.supports_tools and request.tools is not None:
            payload["tools"] = request.tools

        if self.supports_tools and request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.response_format is not None:
            payload["response_format"] = request.response_format

        return payload

    def _build_ollama_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        }

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        if request.stop is not None:
            payload["options"]["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        return payload

    def _build_tgi_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        prompt = self._messages_to_prompt(request.messages)

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": True,
                "stream": request.stream,
            },
        }

        if request.max_tokens is not None:
            payload["parameters"]["max_new_tokens"] = request.max_tokens

        if request.stop is not None:
            payload["parameters"]["stop_sequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        return payload

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        prompt_parts = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        if self.deployment_type == "ollama":
            return self._parse_ollama_response(response_data)
        elif self.deployment_type == "tgi":
            return self._parse_tgi_response(response_data)
        else:
            return self._parse_openai_response(response_data)

    def _parse_openai_response(
        self, response_data: Dict[str, Any]
    ) -> ChatCompletionResponse:
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})

            message = ChatMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
                tool_calls=message_data.get("tool_calls"),
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
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
        )

    def _parse_ollama_response(
        self, response_data: Dict[str, Any]
    ) -> ChatCompletionResponse:
        content = response_data.get("message", {}).get("content", "")

        message = ChatMessage(role=MessageRole.ASSISTANT, content=content)

        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop" if response_data.get("done", False) else None,
        )

        return ChatCompletionResponse(
            id=f"ollama-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=response_data.get("model", ""),
            choices=[choice],
        )

    def _parse_tgi_response(
        self, response_data: Dict[str, Any]
    ) -> ChatCompletionResponse:
        if isinstance(response_data, list) and len(response_data) > 0:
            response_data = response_data[0]

        content = response_data.get("generated_text", "")

        message = ChatMessage(role=MessageRole.ASSISTANT, content=content)

        choice = Choice(
            index=0,
            message=message,
            finish_reason=response_data.get("finish_reason", "stop"),
        )

        return ChatCompletionResponse(
            id=f"tgi-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model="tgi-model",
            choices=[choice],
        )

    def _parse_stream_line(self, line: str) -> Optional[ChatCompletionChunk]:
        if self.deployment_type == "ollama":
            try:
                chunk_data = json.loads(line)
                return self._parse_ollama_chunk(chunk_data)
            except json.JSONDecodeError:
                return None
        else:
            # Standard SSE format
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    return None

                try:
                    chunk_data = json.loads(data)
                    return self._parse_stream_chunk(chunk_data)
                except json.JSONDecodeError:
                    return None

        return None

    def _parse_ollama_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        content = chunk_data.get("message", {}).get("content", "")
        done = chunk_data.get("done", False)

        choices = [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": "stop" if done else None,
            }
        ]

        return ChatCompletionChunk(
            id=f"ollama-{int(time.time())}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=chunk_data.get("model", ""),
            choices=choices,
        )

    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", ""),
            choices=chunk_data.get("choices", []),
        )
