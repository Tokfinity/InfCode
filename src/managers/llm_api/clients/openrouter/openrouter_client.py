"""
OpenRouter Client
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


class OpenRouterClient(BaseLLMAPI):

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        app_name: Optional[str] = None,
        site_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        """
        Initialize OpenRouter Client

        Args:
            api_key: OpenRouter API key
            base_url: API base URL, default to OpenRouter official API
            app_name: Application name (optional, for statistics)
            site_url: Website URL (optional, for statistics)
            timeout: Request timeout
            max_retries: Maximum number of retries
            retry_delay: Retry delay
            **kwargs: Other configuration parameters
        """
        self.app_name = app_name or "tokfinity-llm-client"
        self.site_url = site_url

        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"

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
            "X-Title": self.app_name,
        }

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

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
        """
        Parse OpenRouter API response

        Args:
            response_data: API response data

        Returns:
            ChatCompletionResponse: Parsed response object
        """
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
        """
        Parse stream chunk
        Args:
            chunk_data: Raw chunk data

        Returns:
            Optional[ChatCompletionChunk]: Parsed chunk
        """
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", ""),
            choices=chunk_data.get("choices", []),
            system_fingerprint=chunk_data.get("system_fingerprint"),
        )

    def list_models(self) -> Dict[str, Any]:
        """
        Get available models

        Returns:
            Dict[str, Any]: Model list response
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()

    async def alist_models(self) -> Dict[str, Any]:
        """
        Async get available models

        Returns:
            Dict[str, Any]: Model list response
        """
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()

    def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        Get specific generation request details

        Args:
            generation_id: Generation request ID

        Returns:
            Dict[str, Any]: Generation information
        """
        response = self.client.get(f"/generation?id={generation_id}")
        response.raise_for_status()
        return response.json()

    async def aget_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        Async get specific generation request details

        Args:
            generation_id: Generation request ID

        Returns:
            Dict[str, Any]: Generation information
        """
        response = await self.async_client.get(f"/generation?id={generation_id}")
        response.raise_for_status()
        return response.json()

    def get_account_credits(self) -> Dict[str, Any]:
        """
        Get account credits

        Returns:
            Dict[str, Any]: Account credits information
        """
        response = self.client.get("/auth/key")
        response.raise_for_status()
        return response.json()

    async def aget_account_credits(self) -> Dict[str, Any]:
        """
        Async get account credits

        Returns:
            Dict[str, Any]: Account credits information
        """
        response = await self.async_client.get("/auth/key")
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_popular_models() -> List[str]:
        """
        Get popular models

        Returns:
            List[str]: Popular model names
        """
        return [
            # OpenAI
            "openai/gpt-4-turbo-preview",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            # Anthropic
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            # Google
            "google/gemini-pro",
            "google/gemini-pro-vision",
            # Meta
            "meta-llama/llama-2-70b-chat",
            "meta-llama/llama-2-13b-chat",
            # Mistral
            "mistralai/mixtral-8x7b-instruct",
            "mistralai/mistral-7b-instruct",
            # Open Source
            "microsoft/wizardlm-2-8x22b",
            "databricks/dbrx-instruct",
            "cohere/command-r-plus",
        ]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get specific model details

        Args:
            model_name: Model name

        Returns:
            Dict[str, Any]: Model information
        """
        models_response = self.list_models()
        models = models_response.get("data", [])

        for model in models:
            if model.get("id") == model_name:
                return model

        raise ValueError(f"Model {model_name} not found")

    async def aget_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Async get specific model details

        Args:
            model_name: Model name

        Returns:
            Dict[str, Any]: Model information
        """
        models_response = await self.alist_models()
        models = models_response.get("data", [])

        for model in models:
            if model.get("id") == model_name:
                return model

        raise ValueError(f"Model {model_name} not found")
