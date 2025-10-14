"""Test examples for LLM port documentation."""

import pytest

from portico.ports.llm import (
    ChatCompletionProvider,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Usage,
)


class MockLLMProvider(ChatCompletionProvider):
    """Mock LLM provider for testing."""

    def __init__(self):
        self.completions = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = ChatCompletionResponse(
            id="test-123",
            model=request.model or "gpt-3.5-turbo",
            message=Message(
                role=MessageRole.ASSISTANT, content="Hello! I'm a test response."
            ),
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )
        self.completions.append(request)
        return response

    async def list_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4"]


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_llm_usage():
    """Basic chat completion."""
    provider = MockLLMProvider()

    # Create a simple chat request
    request = ChatCompletionRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="What is 2+2?"),
        ],
        model="gpt-3.5-turbo",
    )

    # Get completion
    response = await provider.complete(request)

    assert response.message.role == MessageRole.ASSISTANT
    assert response.model == "gpt-3.5-turbo"
    assert response.usage.total_tokens > 0


# --8<-- [end:basic-usage]


# --8<-- [start:temperature-control]
@pytest.mark.asyncio
async def test_temperature_control():
    """Control response randomness with temperature."""
    provider = MockLLMProvider()

    # Lower temperature = more focused/deterministic
    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Explain quantum physics")],
        model="gpt-3.5-turbo",
        temperature=0.2,  # Low temperature for factual responses
    )

    response = await provider.complete(request)
    assert response.message is not None


# --8<-- [end:temperature-control]


# --8<-- [start:max-tokens]
@pytest.mark.asyncio
async def test_max_tokens():
    """Limit response length with max_tokens."""
    provider = MockLLMProvider()

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Write a story")],
        max_tokens=100,  # Limit to ~100 tokens
    )

    response = await provider.complete(request)
    assert response.message is not None


# --8<-- [end:max-tokens]


# --8<-- [start:conversation-history]
@pytest.mark.asyncio
async def test_conversation_history():
    """Maintain conversation context."""
    provider = MockLLMProvider()

    # Multi-turn conversation
    request = ChatCompletionRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="My name is Alice."),
            Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alice!"),
            Message(role=MessageRole.USER, content="What's my name?"),
        ]
    )

    response = await provider.complete(request)
    # Assistant has context from previous messages
    assert response.message is not None


# --8<-- [end:conversation-history]


# --8<-- [start:system-prompts]
@pytest.mark.asyncio
async def test_system_prompts():
    """Guide model behavior with system prompts."""
    provider = MockLLMProvider()

    request = ChatCompletionRequest(
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content="You are a Shakespearean poet. Respond in iambic pentameter.",
            ),
            Message(role=MessageRole.USER, content="Tell me about the weather."),
        ]
    )

    response = await provider.complete(request)
    assert response.message.role == MessageRole.ASSISTANT


# --8<-- [end:system-prompts]


# --8<-- [start:list-models]
@pytest.mark.asyncio
async def test_list_available_models():
    """List available models from provider."""
    provider = MockLLMProvider()

    models = await provider.list_models()

    assert "gpt-3.5-turbo" in models
    assert "gpt-4" in models
    assert len(models) > 0


# --8<-- [end:list-models]


# --8<-- [start:token-usage]
@pytest.mark.asyncio
async def test_track_token_usage():
    """Monitor token consumption."""
    provider = MockLLMProvider()

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )

    response = await provider.complete(request)

    # Track usage for billing/monitoring
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


# --8<-- [end:token-usage]


# --8<-- [start:stop-sequences]
@pytest.mark.asyncio
async def test_stop_sequences():
    """Stop generation at specific sequences."""
    provider = MockLLMProvider()

    request = ChatCompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Count to 10")],
        stop=["5", "six"],  # Stop when these appear
    )

    response = await provider.complete(request)
    assert response.message is not None


# --8<-- [end:stop-sequences]


# --8<-- [start:error-handling]
@pytest.mark.asyncio
async def test_llm_error_handling():
    """Handle LLM API errors gracefully."""
    from portico.exceptions import LLMError

    provider = MockLLMProvider()

    try:
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        response = await provider.complete(request)
        assert response is not None
    except LLMError as e:
        # Handle rate limits, API errors, etc.
        print(f"LLM error: {e}")
        # Implement retry logic or fallback


# --8<-- [end:error-handling]


# ========== OpenAI Adapter Examples ==========


# --8<-- [start:openai-provider-config]
def test_openai_provider_configuration():
    """Configure OpenAI provider."""
    # Mock configuration (doesn't actually connect)
    config = {
        "api_key": "sk-test-key",
        "default_model": "gpt-3.5-turbo",
        "timeout": 30.0,
        "max_retries": 3,
        "organization": None,
    }

    # Would create provider like:
    # provider = OpenAIProvider(
    #     api_key=config["api_key"],
    #     default_model=config["default_model"],
    #     timeout=config["timeout"],
    #     max_retries=config["max_retries"],
    # )

    assert config["default_model"] == "gpt-3.5-turbo"
    assert config["timeout"] == 30.0
    assert config["max_retries"] == 3


# --8<-- [end:openai-provider-config]


# --8<-- [start:openai-model-selection]
def test_openai_model_selection():
    """Select OpenAI model based on task."""
    # GPT-3.5 Turbo - Fast and cost-effective
    gpt35_model = "gpt-3.5-turbo"
    assert "gpt-3.5" in gpt35_model

    # GPT-4 - Most capable
    gpt4_model = "gpt-4"
    assert "gpt-4" in gpt4_model

    # GPT-4 Turbo - Long context
    gpt4_turbo_model = "gpt-4-turbo"
    assert "turbo" in gpt4_turbo_model


# --8<-- [end:openai-model-selection]


# --8<-- [start:openai-environment-config]
def test_openai_environment_configuration():
    """Load OpenAI configuration from environment."""
    # Simulate environment variables
    test_env = {
        "OPENAI_API_KEY": "sk-test-key-123",
    }

    # Load from environment
    api_key = test_env.get("OPENAI_API_KEY")

    assert api_key is not None
    assert api_key.startswith("sk-")


# --8<-- [end:openai-environment-config]


# --8<-- [start:openai-cost-calculation]
def test_openai_cost_calculation():
    """Calculate OpenAI API costs."""
    # GPT-3.5 pricing
    gpt35_prompt_cost = 0.0015 / 1000  # $0.0015 per 1K tokens
    gpt35_completion_cost = 0.002 / 1000  # $0.002 per 1K tokens

    # Example usage
    prompt_tokens = 100
    completion_tokens = 50

    cost = prompt_tokens * gpt35_prompt_cost + completion_tokens * gpt35_completion_cost

    assert cost > 0
    assert cost < 1.0  # Should be very small for 150 tokens


# --8<-- [end:openai-cost-calculation]


# --8<-- [start:openai-rate-limit-handling]
@pytest.mark.asyncio
async def test_openai_rate_limit_handling():
    """Handle OpenAI rate limits with exponential backoff."""
    max_retries = 3
    wait_times = []

    for attempt in range(max_retries):
        wait = 2**attempt  # Exponential backoff: 1s, 2s, 4s
        wait_times.append(wait)

    assert wait_times == [1, 2, 4]
    assert len(wait_times) == max_retries


# --8<-- [end:openai-rate-limit-handling]


# --8<-- [start:openai-context-length]
def test_openai_context_length_limits():
    """OpenAI model context length limits."""
    context_limits = {
        "gpt-3.5-turbo": 16_000,  # 16K tokens
        "gpt-4": 8_000,  # 8K tokens
        "gpt-4-turbo": 128_000,  # 128K tokens
    }

    assert context_limits["gpt-3.5-turbo"] == 16_000
    assert context_limits["gpt-4"] == 8_000
    assert context_limits["gpt-4-turbo"] == 128_000


# --8<-- [end:openai-context-length]


# --8<-- [start:openai-system-prompts]
@pytest.mark.asyncio
async def test_openai_system_prompts():
    """Use system prompts to guide OpenAI model behavior."""
    from portico.ports.llm import Message, MessageRole

    # System prompt for Python expert
    system_prompt = Message(
        role=MessageRole.SYSTEM,
        content="You are a Python expert. Provide concise, working code examples.",
    )

    assert system_prompt.role == MessageRole.SYSTEM
    assert "Python expert" in system_prompt.content


# --8<-- [end:openai-system-prompts]


# --8<-- [start:openai-conversation-history]
def test_openai_conversation_history_management():
    """Manage conversation history to stay within context limits."""
    # Simulate long conversation
    conversation_history = [f"Message {i}" for i in range(100)]

    # Keep only last 10 messages
    max_history = 10
    trimmed_history = conversation_history[-max_history:]

    assert len(trimmed_history) == 10
    assert trimmed_history[0] == "Message 90"
    assert trimmed_history[-1] == "Message 99"


# --8<-- [end:openai-conversation-history]


# --8<-- [start:openai-json-mode]
@pytest.mark.asyncio
async def test_openai_json_mode():
    """Request JSON output from OpenAI."""
    from portico.ports.llm import Message, MessageRole

    # System prompt for JSON output
    json_system = Message(
        role=MessageRole.SYSTEM,
        content="You are a helpful assistant. Always respond with valid JSON.",
    )

    user_message = Message(role=MessageRole.USER, content="List 3 colors")

    assert "JSON" in json_system.content
    assert json_system.role == MessageRole.SYSTEM


# --8<-- [end:openai-json-mode]


# --8<-- [start:openai-error-types]
def test_openai_error_types():
    """OpenAI error types for exception handling."""
    from portico.exceptions import (
        LLMError,
        LLMProviderError,
        LLMRateLimitError,
        OpenAIProviderError,
    )

    # All specific errors inherit from LLMError
    assert issubclass(LLMProviderError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(OpenAIProviderError, LLMProviderError)


# --8<-- [end:openai-error-types]


# ========== Anthropic Adapter Examples ==========


# --8<-- [start:anthropic-provider-config]
def test_anthropic_provider_configuration():
    """Configure Anthropic provider."""
    # Mock configuration (doesn't actually connect)
    config = {
        "api_key": "sk-ant-test-key",
        "default_model": "claude-3-sonnet-20240229",
        "timeout": 60.0,
        "max_retries": 3,
    }

    # Would create provider like:
    # provider = AnthropicProvider(
    #     api_key=config["api_key"],
    #     default_model=config["default_model"],
    #     timeout=config["timeout"],
    #     max_retries=config["max_retries"],
    # )

    assert config["default_model"] == "claude-3-sonnet-20240229"
    assert config["timeout"] == 60.0
    assert config["max_retries"] == 3


# --8<-- [end:anthropic-provider-config]


# --8<-- [start:anthropic-model-selection]
def test_anthropic_model_selection():
    """Select Claude model based on task."""
    # Claude 3 Opus - Most capable
    opus_model = "claude-3-opus-20240229"
    assert "opus" in opus_model

    # Claude 3 Sonnet - Balanced
    sonnet_model = "claude-3-sonnet-20240229"
    assert "sonnet" in sonnet_model

    # Claude 3 Haiku - Fast and affordable
    haiku_model = "claude-3-haiku-20240307"
    assert "haiku" in haiku_model


# --8<-- [end:anthropic-model-selection]


# --8<-- [start:anthropic-environment-config]
def test_anthropic_environment_configuration():
    """Load Anthropic configuration from environment."""
    # Simulate environment variables
    test_env = {
        "ANTHROPIC_API_KEY": "sk-ant-test-key-123",
    }

    # Load from environment
    api_key = test_env.get("ANTHROPIC_API_KEY")

    assert api_key is not None
    assert api_key.startswith("sk-ant-")


# --8<-- [end:anthropic-environment-config]


# --8<-- [start:anthropic-cost-calculation]
def test_anthropic_cost_calculation():
    """Calculate Anthropic API costs."""
    # Claude 3 Haiku pricing (most affordable)
    haiku_input_cost = 0.25 / 1_000_000  # $0.25 per 1M tokens
    haiku_output_cost = 1.25 / 1_000_000  # $1.25 per 1M tokens

    # Claude 3 Sonnet pricing
    sonnet_input_cost = 3.0 / 1_000_000  # $3 per 1M tokens
    sonnet_output_cost = 15.0 / 1_000_000  # $15 per 1M tokens

    # Example usage
    input_tokens = 1000
    output_tokens = 500

    haiku_cost = input_tokens * haiku_input_cost + output_tokens * haiku_output_cost

    sonnet_cost = input_tokens * sonnet_input_cost + output_tokens * sonnet_output_cost

    assert haiku_cost > 0
    assert sonnet_cost > haiku_cost  # Sonnet more expensive


# --8<-- [end:anthropic-cost-calculation]


# --8<-- [start:anthropic-long-context]
def test_anthropic_long_context_support():
    """Claude's long context window (200K tokens)."""
    context_limits = {
        "claude-3-opus-20240229": 200_000,
        "claude-3-sonnet-20240229": 200_000,
        "claude-3-haiku-20240307": 200_000,
    }

    # All Claude 3 models support 200K tokens
    for model, limit in context_limits.items():
        assert limit == 200_000


# --8<-- [end:anthropic-long-context]


# --8<-- [start:anthropic-error-handling]
def test_anthropic_error_types():
    """Anthropic error types for exception handling."""
    from portico.exceptions import (
        AnthropicProviderError,
        LLMError,
        LLMProviderError,
        LLMRateLimitError,
    )

    # All specific errors inherit from LLMError
    assert issubclass(LLMProviderError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(AnthropicProviderError, LLMProviderError)


# --8<-- [end:anthropic-error-handling]


# --8<-- [start:anthropic-vs-openai]
def test_anthropic_vs_openai_comparison():
    """Compare Claude vs GPT for different tasks."""
    # Context window comparison
    openai_context = {
        "gpt-3.5-turbo": 16_000,
        "gpt-4": 8_000,
    }

    anthropic_context = {
        "claude-3-haiku": 200_000,
        "claude-3-sonnet": 200_000,
    }

    # Claude has much larger context window
    assert anthropic_context["claude-3-haiku"] > openai_context["gpt-3.5-turbo"]
    assert anthropic_context["claude-3-sonnet"] > openai_context["gpt-4"]


# --8<-- [end:anthropic-vs-openai]
