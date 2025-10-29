"""
LLM API Management Module

Provide a standard OpenAI format LLM API interface that supports:
- Unified Chat Completions endpoint
- Synchronous and asynchronous operations
- Streaming responses
- Tool Calling
- Error handling and retry mechanism(s)

Supported providers:
- OpenAI: OpenAI API and compatible services
- Anthropic: Claude models
- DeepSeek: DeepSeek models
- Private: Private models（vLLM、TGI、Ollama etc.）

Usage example::
    # 1: use general manager (suggested)
    from llm_api import LLMAPIManager

    # create manager
    manager = LLMAPIManager(
        client_name="openai",
        model_name="gpt-3.5-turbo",
        stream=False
    )

    # sendf messages
    response = manager.chat("Hello world!")
    print(response)

    # 2: Use client directly
    from llm_api import OpenAIClient, ChatMessage, MessageRole

    # creat client
    client = OpenAIClient(api_key="your-api-key")

    # create4 message
    messages = [
        ChatMessage(role=MessageRole.USER, content="你好，世界！")
    ]

    # send request
    request = client.create_request(messages=messages, model="gpt-3.5-turbo")
    response = client.chat_completions_create(request)

    print(response.choices[0].message.content)
"""

from src.managers.llm_api.base_client import (
    BaseLLMAPI,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    MessageRole,
    Choice,
    Usage,
)


from src.managers.llm_api.clients.openai.openai_client import OpenAIClient
from src.managers.llm_api.clients.anthropic.anthropic_client import AnthropicClient
from src.managers.llm_api.clients.deepseek.deepseek_client import DeepSeekClient
from src.managers.llm_api.clients.openrouter.openrouter_client import OpenRouterClient
from src.managers.llm_api.clients.private.private_client import PrivateModelClient


from src.managers.llm_api.api_manager import (
    LLMAPIManager,
    create_manager,
    create_common_manager,
    COMMON_CONFIGS,
)

__all__ = [
    "BaseLLMAPI",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "MessageRole",
    "Choice",
    "Usage",
    "OpenAIClient",
    "AnthropicClient",
    "DeepSeekClient",
    "OpenRouterClient",
    "PrivateModelClient",
    "LLMAPIManager",
    "create_manager",
    "create_common_manager",
    "COMMON_CONFIGS",
]

__version__ = "1.0.0"
__author__ = "Tokfinity Team"
__description__ = "标准 OpenAI 格式的 LLM API 基类库"
