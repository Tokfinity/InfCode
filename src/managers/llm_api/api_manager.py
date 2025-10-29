"""
LLM API manager
"""

import os
import time
from typing import Dict, List, Any, Optional, Union, Generator
from dotenv import load_dotenv
import yaml
from traceback import format_exc
from .base_client import (
    BaseLLMAPI,
    ChatMessage,
    MessageRole,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)

# 导入所有客户端
from .clients.openai.openai_client import OpenAIClient
from .clients.anthropic.anthropic_client import AnthropicClient
from .clients.deepseek.deepseek_client import DeepSeekClient
from .clients.openrouter.openrouter_client import OpenRouterClient
from .clients.private.private_client import PrivateModelClient


class LLMAPIManager:
    SUPPORTED_CLIENTS = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "deepseek": DeepSeekClient,
        "openrouter": OpenRouterClient,
        "private": PrivateModelClient,
    }

    DEFAULT_CONFIGS = {
        "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
        "anthropic": {"base_url": None, "api_key_env": "ANTHROPIC_API_KEY"},
        "deepseek": {"base_url": None, "api_key_env": "DEEPSEEK_API_KEY"},
        "openrouter": {
            "base_url": None,
            "api_key_env": "OPENROUTER_API_KEY",
            "extra_config": {
                "app_name": "tokfinity-llm-client",
                "site_url": "https://github.com/your-repo",
            },
        },
        "private": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": "PRIVATE_API_KEY",
            "extra_config": {"deployment_type": "vllm"},
        },
    }

    def __init__(
        self,
        client_name: Optional[str] = None,
        stream: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        auto_load_env: bool = True,
        logger: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # if client not provided, get first provider from default providers in config
        if client_name is None:
            default_client, default_model = (
                self._load_default_client_and_model_from_config()
            )
            self.client_name = default_client
            self.default_model = default_model
        else:
            self.client_name = client_name.lower()
            self.default_model = None
        
        self.stream = stream
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger
        self.logger.info(f"[LLMAPIManager]: Using client: {self.client_name}, model: {self.default_model}.")
        self.config = config

        if auto_load_env:
            self._load_environment()

        if self.client_name not in self.SUPPORTED_CLIENTS:
            raise ValueError(
                f"Unsupported client: {client_name}。"
                f"Support client: {list(self.SUPPORTED_CLIENTS.keys())}"
            )

        self.client = self._create_client(api_key, base_url, logger, **kwargs)


    def _load_environment(self) -> None:
        """load environment variables"""
        env_paths = [".env", "../.env", "../../.env", "../../../.env"]

        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                break

    def _create_client(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, logger: Optional[Any] = None, **kwargs
    ) -> BaseLLMAPI:
        client_class = self.SUPPORTED_CLIENTS[self.client_name]
        config = self.DEFAULT_CONFIGS[self.client_name]

        if api_key is None:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                if self.client_name == "private":
                    api_key = "EMPTY"  # private mode may not need a key
                else:
                    raise ValueError(
                        f"Fail to find env variable, please set: {config['api_key_env']} "
                        f"or upload ai_key parameter when initialize"
                    )

        if base_url is None:
            env_key = f"{self.client_name.upper()}_BASE_URL"
            if self.client_name == "private":
                env_key = "PRIVATE_URL"

            base_url = os.getenv(env_key)
            if base_url is None:
                base_url = config.get("base_url")

        client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        if logger is not None:
            client_kwargs["logger"] = logger

        extra_config = config.get("extra_config", {})
        client_kwargs.update(extra_config)
        client_kwargs.update(kwargs)

        if self.client_name == "openrouter":
            client_kwargs.setdefault("app_name", "tokfinity-llm-client")
        elif self.client_name == "private":
            client_kwargs.setdefault("deployment_type", "vllm")

        return client_class(**client_kwargs)

    def _load_default_client_and_model_from_config(self) -> (str, Optional[str]):
        """
        Get first item from providers from config/config.yaml as the default client
        And take the first model as default model
        """
        # Parse the root of config file （relative to the project root）
        # This file is in src/managers/llm_api/api_manager.py
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        config_path = os.path.join(base_dir, "config", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        providers = cfg.get("providers")
        if not isinstance(providers, dict) or len(providers) == 0:
            raise ValueError("config.yaml lack of providers config or format error")

        first_provider_name = next(iter(providers.keys()))
        models = providers.get(first_provider_name) or []
        first_model = (
            models[0] if isinstance(models, list) and len(models) > 0 else None
        )

        client_key = first_provider_name.strip().lower()
        if client_key not in self.SUPPORTED_CLIENTS:
            raise ValueError(
                f"Default provider '{first_provider_name}' in config not registered in SUPPORTED_CLIENTS"
            )

        return client_key, first_model

    def chat(
        self,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[
        ChatCompletionResponse, Generator[ChatCompletionChunk, None, None], None
    ]:
        """
        Send chat message and get response

        Args:
            model: model name (optional, default self.default_model)
            messages: Concatenated message list.
                - If list of dict: [{"role": "system|user|assistant|tool", "content": "..."}]
                - Or `ChatMessage` list
            temperature: Temperature
            max_tokens: Max tokens
            timeout: Request timeout, use the value from initialization if not specified
            retry: Max retries, use the value from initialization if not specified
            tools: Tools description list (OpenAI tools format)
            tool_choice: Tool choice strategy（"auto" | "none" | {"type":..., ...}）
            **kwargs: Other request args

        Returns:
            Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None], None]:
                - Non-streaming: Return the complete ChatCompletionResponse object
                - Stream: Return a streaming response generator
                - Fail: Return None
        """
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_retry = retry if retry is not None else self.max_retries

        actual_model = model if model is not None else self.default_model
        if not actual_model:
            raise ValueError("Model unprovided, and cannot get default model from config")

        normalized_messages: List[ChatMessage] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                if msg.content is None:
                    msg.content = ""
                normalized_messages.append(msg)
            else:
                role_value = msg.get("role")
                content_value = msg.get("content")

                if content_value is None:
                    content_value = ""
                
                role_enum = MessageRole(role_value)
                normalized_messages.append(
                    ChatMessage(
                        role=role_enum,
                        content=content_value,
                        name=msg.get("name"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id"),
                    )
                )

        last_exception = None
        for attempt in range(actual_retry + 1):
            try:
                request = self.client.create_request(
                    messages=normalized_messages,
                    model=actual_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=self.stream,
                    tools=tools,
                    tool_choice=tool_choice,
                    config=self.config,
                    **kwargs,
                )

                response = self.client.chat_completions_create(request)

                if self.stream:
                    # Streaming response: has not processed
                    return response
                else:
                    # Non-streaming response: return complete ChatCompletionResponse
                    return response

            except Exception as e:
                last_exception = e
                if attempt < actual_retry:
                    delay = min(2**attempt, 30)
                    if self.logger:
                        self.logger.warning(
                            f"{attempt + 1}th try failed，retry after {delay} seconds: {str(e)}, traceback: {format_exc()}."
                        )
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(
                            f"All {actual_retry + 1} tries failed: {str(e)}"
                        )

        return None

    def chat_simple(
        self,
        model: str,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Optional[str]:
        # Ban stream
        original_stream = self.stream
        self.stream = False

        try:
            response = self.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                retry=retry,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            if response and hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            return None

        finally:
            # Recover original stream settings
            self.stream = original_stream

    def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        **kwargs,
    ) -> Optional[EmbeddingResponse]:
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_retry = retry if retry is not None else self.max_retries

        if isinstance(input_text, str):
            text_list = [input_text]
            single_input = True
        else:
            text_list = input_text
            single_input = False

        if self.logger:
            self.logger.debug(
                f"Start embedding request - client: {self.client_name}, model: {model}, text num: {len(text_list)}"
            )

        if not hasattr(self.client, "create_embeddings"):
            error_msg = f"Client {self.client_name} not support embedding"
            if self.logger:
                self.logger.error(error_msg)
            return None

        last_exception = None
        for attempt in range(actual_retry + 1):
            try:
                if self.logger:
                    self.logger.debug(f"{attempt + 1}th try to create embedding vector")

                response = self.client.create_embeddings(
                    input_text=text_list,
                    model=model,
                    encoding_format=encoding_format,
                    dimensions=dimensions,
                    user=user,
                    timeout=actual_timeout,
                    max_retries=1,
                    **kwargs,
                )

                if response:
                    return response
                else:
                    raise Exception("Client return empty response")

            except Exception as e:
                last_exception = e
                if attempt < actual_retry:
                    delay = min(2**attempt, 30)
                    if self.logger:
                        self.logger.warning(
                            f"{attempt + 1}th embedding request failed，retry after {delay}s: {str(e)}, traceback: {format_exc()}."
                        )
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(
                            f"All {actual_retry + 1} embedding tries failed: {str(e)}, traceback: {format_exc()}."
                        )

        return None

    def get_client_info(self) -> Dict[str, Any]:
        return {
            "client_name": self.client_name,
            "stream": self.stream,
            "client_info": (
                self.client.get_model_info()
                if hasattr(self.client, "get_model_info")
                else {}
            ),
        }

    def get_model_name(self) -> str:
        return self.default_model or "unknown"

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        return f"LLMAPIManager(client={self.client_name}, stream={self.stream})"

    def __repr__(self) -> str:
        return (
            f"LLMAPIManager("
            f"client_name='{self.client_name}', "
            f"stream={self.stream})"
        )


# 便捷函数
def create_manager(
    client_name: str, stream: bool = False, logger: Optional[Any] = None, **kwargs
) -> LLMAPIManager:
    return LLMAPIManager(
        client_name=client_name, stream=stream, logger=logger, **kwargs
    )


COMMON_CONFIGS = {
    "openai_gpt4": {"client_name": "openai", "model_name": "gpt-4o"},
    "openai_gpt35": {"client_name": "openai", "model_name": "gpt-3.5-turbo"},
    "claude_sonnet": {
        "client_name": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022",
    },
    "claude_haiku": {
        "client_name": "anthropic",
        "model_name": "claude-3-haiku-20240307",
    },
    "deepseek_chat": {"client_name": "deepseek", "model_name": "deepseek-chat"},
    "deepseek_coder": {"client_name": "deepseek", "model_name": "deepseek-coder"},
}


def create_common_manager(
    config_name: str, stream: bool = False, logger: Optional[Any] = None, **kwargs
) -> LLMAPIManager:
    if config_name not in COMMON_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available configs: {list(COMMON_CONFIGS.keys())}"
        )

    config = COMMON_CONFIGS[config_name]
    return LLMAPIManager(
        client_name=config["client_name"], stream=stream, logger=logger, **kwargs
    )
