"""
LLM API base class - standard OpenAI format
Multi LLM providers' universal API supported
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
import requests
import asyncio
from traceback import format_exc
from src.managers.log.logger import Logger


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: MessageRole
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    messages: List[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice:
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    system_fingerprint: Optional[str] = None


@dataclass
class EmbeddingRequest:
    input: Union[str, List[str]]
    model: str
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


@dataclass
class EmbeddingData:
    object: str
    embedding: List[float]
    index: int


@dataclass
class EmbeddingUsage:
    prompt_tokens: int
    total_tokens: int


@dataclass
class EmbeddingResponse:
    object: str
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class BaseLLMAPI(ABC):
    """
    LLM API base class

    Provide standard OpenAI format API, support:
    - Synchronous/Asynchronous Chat Completions
    - Streaming Responses
    - Tool Calling
    - Error handling and Retry
    - Usage Statics
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
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = kwargs

        self.logger = logger

        self.session: Optional[requests.Session] = None

        self._initialize_client()

    def _create_http_clients(self, headers: Dict[str, str]) -> None:
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.session.timeout = self.timeout

    @abstractmethod
    def _initialize_client(self) -> None:
        pass

    @abstractmethod
    def _get_chat_endpoint(self) -> str:
        pass

    @abstractmethod
    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Optional[ChatCompletionChunk]:
        pass

    def chat_completions_create(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
        self.validate_request(request)

        def _make_request():
            payload = self._build_request_payload(request)
            endpoint = self._get_chat_endpoint()
            if request.stream:
                return self._stream_chat_completion(payload, endpoint)
            else:

                full_url = self.base_url.rstrip("/") + endpoint

                headers = {}
                if self.session and self.session.headers:
                    headers.update(self.session.headers)
                else:

                    headers = {
                        "Content-Type": "application/json",
                    }
                    if self.api_key and self.api_key != "EMPTY":
                        headers["Authorization"] = f"Bearer {self.api_key}"

                response = requests.post(
                    full_url, json=payload, headers=headers, timeout=self.timeout
                )

                if response.status_code != 200:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(error_msg)
                    raise requests.exceptions.HTTPError(error_msg, response=response)
                response.raise_for_status()
                return self._parse_response(response.json())

        return self._retry_with_backoff(_make_request)

    async def achat_completions_create(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        self.validate_request(request)

        def _make_async_request():
            payload = self._build_request_payload(request)
            endpoint = self._get_chat_endpoint()

            if request.stream:
                return self._stream_chat_completion(
                    payload, endpoint
                )
            else:
                full_url = self.base_url.rstrip("/") + endpoint

                headers = {}
                if self.session and self.session.headers:
                    headers.update(self.session.headers)
                else:
                    headers = {
                        "Content-Type": "application/json",
                    }
                    if self.api_key and self.api_key != "EMPTY":
                        headers["Authorization"] = f"Bearer {self.api_key}"

                response = requests.post(
                    full_url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_response(response.json())

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _make_async_request)

    def create_message(self, role: MessageRole, content: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> ChatMessage:
        return ChatMessage(role=role, content=content, **kwargs)

    def _make_token_cache_request(self, messages: List[ChatMessage], model: str, config: Optional[Dict[str, Any]] = None) -> list[ChatMessage]:
        """
        Insert cache_control block to Claude/Anthropic model according to config, forward compatibility

        - Only activate when model is "Claude/Anthropic"
        - Preserve the content as is for other providers to avoid breaking compatibility.
        - If token_cache.role_turns is configured, inject it matching the turn and role; otherwise, only inject it into the first system message.
        """
        role_turns = self._load_role_turns(config)
        #if self.logger:
        #    self.logger.debug(f"In _make_token_cache_request, got role_turns: {role_turns}.")
        if not self._is_claude_like(model):
            return messages

        turn_to_roles: Dict[int, List[str]] = {}
        try:
            for rt in role_turns or []:
                try:
                    turn = int(rt.get("turn"))
                    role = (rt.get("role") or "").strip().lower()
                    if role:
                        if turn not in turn_to_roles:
                            turn_to_roles[turn] = []
                        turn_to_roles[turn].append(role)
                except Exception:
                    continue
        except Exception:
            turn_to_roles = {}

        current_turn = 0
        result_messages: List[ChatMessage] = []

        for msg in messages:
            msg_blocks = self._to_claude_blocks(msg.content)

            # 0th turn rule: If met user, only add cache to its own content and put it into result.
            if current_turn == 0 and msg.role == MessageRole.USER:
                cached_blocks = self._add_cache_flag(msg_blocks)
                result_messages.append(
                    ChatMessage(
                        role=msg.role,
                        content=cached_blocks,
                        name=msg.name,
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                    )
                )
                #if self.logger:
                #    self.logger.debug(
                #        f"Applied cache to initial user message at turn {current_turn}."
                #    )
                continue

            # Other messages: just add it to result
            result_messages.append(
                ChatMessage(
                    role=msg.role,
                    content=msg_blocks,
                    name=msg.name,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                )
            )

            # Hit anchor point: When the current turn's configuration includes the assistant and the current message is from the assistant
            # add an extra system turn.
            roles_for_turn = turn_to_roles.get(current_turn, [])
            if msg.role == MessageRole.ASSISTANT and roles_for_turn and ("assistant" in roles_for_turn):
                refresh_text = f"refresh cache tokens."
                refresh_blocks = self._to_claude_blocks(refresh_text)
                refresh_blocks = self._add_cache_flag(refresh_blocks)
                result_messages.append(
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=refresh_blocks,
                    )
                )
                #if self.logger:
                #    self.logger.debug(
                #        f"Appended system refresh after assistant at turn {current_turn}, refresh_blocks: {refresh_blocks}."
                #    )

            # assistant message make the round +1
            if msg.role == MessageRole.ASSISTANT:
                current_turn += 1

        return result_messages

    def _to_claude_blocks(self, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            return content
        return [{"type": "text", "text": content}]

    def _add_cache_flag(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        added = False
        for blk in blocks:
            if not added and isinstance(blk, dict) and blk.get("type") == "text":
                out.append({**blk, "cache_control": {"type": "ephemeral"}})
                added = True
            else:
                out.append(blk)
        return out

    def _is_claude_like(self, model: str) -> bool:
        try:
            model_lc = (model or "").lower()
            return ("claude" in model_lc) or ("anthropic" in model_lc)
        except Exception:
            return False

    def _load_role_turns(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Load token_cache.role_turns，Priority：Invoker config > default value"""
        try:
            role_turns = None

            # 1) Read from invoker's config
            if config and isinstance(config, dict):
                token_cache_cfg = config.get("token_cache")
                if isinstance(token_cache_cfg, dict):
                    role_turns = token_cache_cfg.get("role_turns")

            # 2) If still lack, use default value in current example config
            if not role_turns:
                role_turns = [
                    {"turn": 0, "role": "user"},
                    {"turn": 25, "role": "assistant"},
                    {"turn": 50, "role": "assistant"},
                    {"turn": 75, "role": "assistant"},
                ]

            return role_turns
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Read token_cache.role_turns fail，use default value: {e}")
            return [
                {"turn": 0, "role": "user"},
                {"turn": 25, "role": "assistant"},
                {"turn": 50, "role": "assistant"},
                {"turn": 75, "role": "assistant"},
            ]

    def create_request(
        self, messages: List[ChatMessage], model: str, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ChatCompletionRequest:

        messages = self._make_token_cache_request(messages, model, config)

        return ChatCompletionRequest(messages=messages, model=model, **kwargs)

    def _handle_error(self, error: Exception) -> None:
        if self.logger:
            self.logger.error(f"API request failed: {error}")
        else:
            print(f"API request failed: {error}")
        raise error

    def _retry_with_backoff(self, func, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)  # 指数退避
                    if self.logger:
                        self.logger.warning(
                            f"{attempt + 1}th try failed，retry after {delay}s: {e}, traceback: {format_exc()}."
                        )
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"All retries failed: {e}, traceback: {format_exc()}.")

        raise last_exception

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.__class__.__name__,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def validate_request(self, request: ChatCompletionRequest) -> bool:
        if not request.messages:
            raise ValueError("Message list is empty")

        if not request.model:
            raise ValueError("Model name is empty")

        for idx, message in enumerate(request.messages):
            if not message.content and not message.tool_calls:
                try:
                    msg_info = {
                        "index": idx,
                        "role": getattr(message.role, "value", str(message.role)),
                        "content_len": (
                            len(message.content)
                            if isinstance(message.content, str)
                            else (len(message.content) if isinstance(message.content, list) else 0)
                        ),
                        "has_tool_calls": bool(message.tool_calls),
                        "tool_calls": message.tool_calls,
                    }
                    if self.logger:
                        self.logger.warning(
                        f"Request validation failed: Invalid message exists (lacking both content and tool_calls): {json.dumps(msg_info, ensure_ascii=False)}"
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Request validation failed: Invalid message exists (lacking both content and tool_calls), index={idx}, error: {e}, traceback: {format_exc()}."
                        )
                raise ValueError("Cannot lacking both content and tool_calls")

        return True

    def format_messages_for_api(
        self, messages: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        formatted_messages = []

        for message in messages:
            msg_dict = {"role": message.role.value, "content": message.content}

            if message.name:
                msg_dict["name"] = message.name

            if message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls

            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            formatted_messages.append(msg_dict)

        return formatted_messages

    def _stream_chat_completion(
        self, payload: Dict[str, Any], endpoint: str
    ) -> Generator[ChatCompletionChunk, None, None]:
        full_url = self.base_url.rstrip("/") + endpoint

        headers = {}
        if self.session and self.session.headers:
            headers.update(self.session.headers)
        else:
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key and self.api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            full_url, json=payload, headers=headers, timeout=self.timeout, stream=True
        )
        response.raise_for_status()

        try:
            line_count = 0
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    line_str = line.decode("utf-8")
                    chunk = self._parse_stream_line(line_str)
                    if chunk:
                        yield chunk
                    else:
                        print(f"Jump invalid line")
            print(f"Streaming request process done, {line_count} processed")
        finally:
            response.close()

    async def _astream_chat_completion(
        self, payload: Dict[str, Any], endpoint: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        import asyncio

        def _sync_stream():
            return list(self._stream_chat_completion(payload, endpoint))

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, _sync_stream)

        for chunk in chunks:
            yield chunk

    def _parse_stream_line(self, line: str) -> Optional[ChatCompletionChunk]:
        # Process standard SSE format
        if line.startswith("data: "):
            data = line[6:]

            if data.strip() == "[DONE]":
                return None

            try:
                chunk_data = json.loads(data)
                return self._parse_stream_chunk(chunk_data)
            except json.JSONDecodeError:
                self.logger.warning(f"Unable to parse streaming data: {data}")
                return None

        return None

    def close(self) -> None:
        if self.session:
            self.session.close()

    async def aclose(self) -> None:
        if self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(base_url={self.base_url})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"base_url={self.base_url}, "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )
