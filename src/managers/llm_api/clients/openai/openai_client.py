"""
OpenAI Client
"""

import time
from typing import Dict, List, Any, Optional, Union
from traceback import format_exc
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
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)


class OpenAIClient(BaseLLMAPI):
    """
    OpenAI API Client

    OpenAI API supported, and other API services compatible with the OpenAI format
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        self.organization = organization

        if base_url is None:
            base_url = "https://api.openai.com/v1"

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

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        self._create_http_clients(headers)

        return "/chat/completions"

    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload = {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": request.stream,
        }

        # Optional
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.stop is not None:
            payload["stop"] = request.stop

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
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", ""),
            choices=chunk_data.get("choices", []),
            system_fingerprint=chunk_data.get("system_fingerprint"),
        )

    def list_models(self) -> Dict[str, Any]:
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()

    async def alist_models(self) -> Dict[str, Any]:
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()

    def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> EmbeddingResponse:

        actual_timeout = timeout if timeout is not None else self.timeout
        actual_max_retries = (
            max_retries if max_retries is not None else self.max_retries
        )
        actual_retry_delay = (
            retry_delay if retry_delay is not None else self.retry_delay
        )

        request = EmbeddingRequest(
            input=input_text,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        payload = self._build_embedding_request_payload(request)

        for attempt in range(actual_max_retries + 1):
            try:
                print(f"Debug: Payload: {payload}")

                response = self.session.post(
                    f"{self.base_url}/embeddings", json=payload, timeout=actual_timeout
                )


                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_embedding_response(response_data)
                else:
                    error_msg = f"embedding failed (try {attempt + 1}): HTTP {response.status_code}"
                    if hasattr(response, "text"):
                        error_msg += f" - {response.text}"
                    print(f"Debug: {error_msg}")

                    if attempt < actual_max_retries:
                        print(f"Debug: wait {actual_retry_delay} and retry...")
                        time.sleep(actual_retry_delay)
                        continue
                    else:
                        raise Exception(f"All retries failed: {error_msg}")

            except Exception as e:
                error_msg = f"Embedding request failed (try {attempt + 1}): {str(e)}, traceback: {format_exc()}."
                print(f"Debug: {error_msg}")

                if attempt < actual_max_retries:
                    print(f"Debug: wait {actual_retry_delay} and retry...")
                    time.sleep(actual_retry_delay)
                    continue
                else:
                    raise Exception(f"All retries failed: {str(e)}, traceback: {format_exc()}.")

        raise Exception("Unknown error")

    def _build_embedding_request_payload(
        self, request: EmbeddingRequest
    ) -> Dict[str, Any]:

        payload = {
            "input": request.input,
            "model": request.model,
            "encoding_format": request.encoding_format,
        }

        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions

        if request.user is not None:
            payload["user"] = request.user

        return payload

    def _parse_embedding_response(
        self, response_data: Dict[str, Any]
    ) -> EmbeddingResponse:
        embedding_data_list = []
        for data_item in response_data.get("data", []):
            embedding_data = EmbeddingData(
                object=data_item.get("object", "embedding"),
                embedding=data_item.get("embedding", []),
                index=data_item.get("index", 0),
            )
            embedding_data_list.append(embedding_data)

        usage_data = response_data.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=embedding_data_list,
            model=response_data.get("model", ""),
            usage=usage,
        )
