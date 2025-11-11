# LLM Kit

## Overview

The LLM Kit provides multi-provider Large Language Model integration with conversation management, prompt templating, and variable collection. It enables applications to interact with LLMs (OpenAI, Anthropic) through a unified interface with support for stored prompts, conversation threads, and dynamic variable substitution.

**Purpose**: Unified LLM access with conversation management and template-based prompt engineering.

**Domain**: Chat completions, conversation management, prompt engineering, template rendering

**Capabilities**:

- Multi-provider support (OpenAI GPT models, Anthropic Claude models)
- Chat completions with message history
- Conversation thread management with persistence
- Prompt template storage and rendering
- Variable collection and validation for prompts
- Template-based conversation initialization
- RAG context injection support
- Model listing and configuration
- Token usage tracking

**Architecture Type**: Stateless kit (no database models for core LLM operations, optional database for conversations/prompts)

**When to Use**:

- AI-powered chatbots and assistants
- Content generation and summarization
- Code generation and analysis
- Question answering systems
- Multi-turn conversations with context
- Template-based prompt engineering
- Applications requiring multiple LLM providers

## Quick Start

### Basic Chat Completion

```python
from portico import compose
from portico.ports.llm import Message, MessageRole

app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.llm(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o-mini",
            temperature=0.7,
        ),
    ],
)

await app.initialize()

# Get LLM service
llm_service = app.kits["llm"].service

# Simple completion
response_text = await llm_service.complete_simple(
    content="Explain quantum computing in one sentence.",
    temperature=0.5
)
print(response_text)

# Multi-turn conversation
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful coding assistant."),
    Message(role=MessageRole.USER, content="How do I reverse a string in Python?"),
]

response = await llm_service.complete_from_messages(
    messages=messages,
    model="gpt-4o-mini",
    temperature=0.3
)
print(response.message.content)
```

## Core Concepts

### LLMService

The `LLMService` is the main service for LLM operations. It provides methods for chat completions, prompt management, and template rendering.

**Key Methods**:

- `complete_chat()` - Generate completion from ChatCompletionRequest
- `complete_from_messages()` - Generate completion from message list
- `complete_from_prompt()` - Generate completion using stored prompt template
- `complete_simple()` - Simple completion returning just text
- `list_models()` - List available models for the provider
- `create_prompt()`, `get_prompt()`, `update_prompt()`, `delete_prompt()` - Prompt CRUD operations
- `validate_prompt_template()` - Validate Jinja2 template syntax

### ConversationService

The `ConversationService` manages multi-turn conversation threads with persistent storage and LLM integration.

**Key Features**:

- Persistent conversation threads with database storage
- Automatic message history management
- Template-based conversation initialization
- Variable substitution in conversation prompts
- Message pagination and retrieval

**Key Methods**:

- `create_conversation()` - Create new conversation thread
- `send_message_and_get_response()` - Send user message and get LLM response
- `send_message_with_prompt()` - Start conversation with prompt template
- `send_message_with_variables()` - Send message with variable substitution
- `get_conversation_messages()` - Retrieve conversation history
- `clear_conversation()` - Reset conversation messages

### VariableService

The `VariableService` manages variable definitions for prompt templates with type validation and user input collection.

**Supported Variable Types**:

- `TEXT` - Free-form text input
- `NUMBER` - Numeric values with validation
- `BOOLEAN` - True/false values
- `SELECT` - Choice from predefined options

**Key Methods**:

- `create_variable_definition()` - Define variable with type and validation
- `validate_variable_values()` - Validate values against definitions
- `set_conversation_variables()` - Store variable values for conversation
- `get_prompt_variables_info()` - Get variable definitions for prompt

### ChatCompletionProvider

The `ChatCompletionProvider` is the port interface implemented by LLM adapters (OpenAI, Anthropic). Applications interact with the abstraction, not specific providers.

**Available Providers**:

- `OpenAIProvider` - OpenAI GPT models (gpt-4, gpt-4o-mini, gpt-3.5-turbo, etc.)
- `AnthropicProvider` - Anthropic Claude models (claude-3-5-sonnet, claude-3-opus, etc.)

### Prompt Templates

Prompts are stored templates using Jinja2 syntax for variable substitution. They enable reusable, versioned prompt engineering with type-safe variable inputs.

**Template Features**:

- Jinja2 variable substitution: `{ { user_name } }`, `{ { topic } }`
- Variable extraction and validation
- Default model/temperature/max_tokens in metadata
- Tags for organization and filtering
- Version tracking (when used with VersionKit)

### Message Roles

Chat messages have three roles:

- `SYSTEM` - System instructions that set assistant behavior
- `USER` - User input messages
- `ASSISTANT` - LLM-generated responses

## Configuration

### LlmKitConfig

```python
from dataclasses import dataclass

@dataclass
class LlmKitConfig:
    provider: Literal["openai", "anthropic"]  # Required
    api_key: str                              # Required
    model: Optional[str] = None               # Uses provider default if not set
    temperature: float = 0.7                  # 0.0-2.0
    max_tokens: Optional[int] = None          # Provider default if None
    enable_prompt_registry: bool = False      # Enable database prompt storage
```

### Composing the LLM Kit

```python
from portico import compose

# OpenAI configuration
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.llm(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
            enable_prompt_registry=True,  # Enable prompt storage
        ),
    ],
)

# Anthropic configuration
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.llm(
            provider="anthropic",
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,
        ),
    ],
)
```

### Environment-Based Configuration

```python
import os

app = compose.webapp(
    database_url=os.getenv("DATABASE_URL"),
    kits=[
        compose.llm(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        ),
    ],
)
```

## Usage Examples

### 1. Multi-Turn Conversation with History

```python
from portico.ports.llm import Message, MessageRole

messages = []

# System message (optional, sets behavior)
messages.append(Message(
    role=MessageRole.SYSTEM,
    content="You are a Python expert who explains concepts clearly and concisely."
))

# First user message
messages.append(Message(
    role=MessageRole.USER,
    content="What are decorators in Python?"
))

# Get first response
response = await llm_service.complete_from_messages(messages)
messages.append(response.message)

print(f"Assistant: {response.message.content}")
print(f"Tokens used: {response.usage.total_tokens}")

# Follow-up question
messages.append(Message(
    role=MessageRole.USER,
    content="Can you show me an example?"
))

# Get second response (with full history)
response = await llm_service.complete_from_messages(messages)
messages.append(response.message)

print(f"Assistant: {response.message.content}")
```

### 2. Persistent Conversations with ConversationService

```python
from portico.ports.llm import CreateConversationRequest

# Requires conversation_repository in LlmKit initialization
conversation_service = ConversationService(
    conversation_repository=conversation_repo,
    llm_service=llm_service
)

# Create conversation
conversation = await conversation_service.create_conversation(
    CreateConversationRequest(
        title="Python Learning Session",
        user_id=user.id
    )
)

# Send message and get response
conversation = await conversation_service.send_message_and_get_response(
    conversation_id=conversation.id,
    user_message="What is list comprehension?",
    temperature=0.5
)

# All messages are stored in database
messages = await conversation_service.get_conversation_messages(
    conversation_id=conversation.id
)

for msg in messages:
    print(f"{msg.role}: {msg.content}")
```

### 3. Prompt Templates with Variables

```python
from portico.ports.llm import create_prompt_request

# Create prompt template
prompt_request = create_prompt_request(
    name="email_generator",
    template="""Generate a professional email with the following details:

To: { { recipient_name } }
Subject: { { subject } }
Tone: { { tone } }

Please write an appropriate email body.""",
    description="Generate professional emails",
    variables=["recipient_name", "subject", "tone"],
    default_model="gpt-4o-mini",
    default_temperature=0.7,
    tags=["email", "communication"]
)

prompt = await llm_service.create_prompt(prompt_request)

# Use prompt with variables
response = await llm_service.complete_from_prompt(
    prompt_name_or_id="email_generator",
    variables={
        "recipient_name": "Dr. Smith",
        "subject": "Research Collaboration Proposal",
        "tone": "formal and respectful"
    }
)

print(response.message.content)
```

### 4. Variable Collection and Validation

```python
from portico.ports.llm import (
    CreateVariableDefinitionRequest,
    VariableType
)

variable_service = VariableService(variable_repository=variable_repo)

# Define variables with types and validation
await variable_service.create_variable_definition(
    CreateVariableDefinitionRequest(
        name="tone",
        description="Email tone",
        variable_type=VariableType.SELECT,
        options=["formal", "casual", "friendly"],
        is_required=True
    )
)

await variable_service.create_variable_definition(
    CreateVariableDefinitionRequest(
        name="word_count",
        description="Approximate word count",
        variable_type=VariableType.NUMBER,
        is_required=False
    )
)

# Validate user input
variable_defs = await variable_service.list_variable_definitions()
user_values = {
    "tone": "professional",  # Invalid - not in options
    "word_count": "many"     # Invalid - not a number
}

errors = await variable_service.validate_variable_values(
    variable_defs,
    user_values
)

if errors:
    print(f"Validation errors: {errors}")
    # Output: {'tone': ['tone must be one of: formal, casual, friendly'],
    #          'word_count': ['word_count must be a valid number']}
```

### 5. RAG Context Injection

```python
# Fetch relevant context from vector store or database
context = """
Product: CloudSync Pro
Features: Real-time sync, 256-bit encryption, 1TB storage
Price: $9.99/month
"""

# Inject context into completion
response = await llm_service.complete_from_messages(
    messages=[
        Message(
            role=MessageRole.USER,
            content="What are the key features of CloudSync Pro?"
        )
    ],
    rag_context=context,  # Injected as additional context
    temperature=0.3
)

print(response.message.content)
# LLM response will be grounded in the provided context
```

## Domain Models

### Message

Represents a single message in a conversation.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `MessageRole` | Message role (system, user, assistant) |
| `content` | `str` | Message text content |

### ChatCompletionRequest

Request for chat completion.

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Message]` | Conversation messages |
| `model` | `Optional[str]` | Model name override |
| `temperature` | `Optional[float]` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `Optional[int]` | Maximum tokens to generate |
| `top_p` | `Optional[float]` | Nucleus sampling (0.0-1.0) |
| `frequency_penalty` | `Optional[float]` | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | `Optional[float]` | Presence penalty (-2.0 to 2.0) |
| `stop` | `Optional[str \| List[str]]` | Stop sequences |
| `stream` | `bool` | Whether to stream response |
| `rag_context` | `Optional[str]` | RAG context to inject |

### ChatCompletionResponse

Response from chat completion.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Response identifier |
| `model` | `str` | Model used for completion |
| `message` | `Message` | Generated message |
| `usage` | `Optional[Usage]` | Token usage information |
| `created_at` | `datetime` | Response creation timestamp |

### Usage

Token usage statistics.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Tokens in prompt |
| `completion_tokens` | `int` | Tokens in completion |
| `total_tokens` | `int` | Total tokens used |

### Conversation

Persistent conversation thread.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Conversation identifier |
| `title` | `str` | Conversation title |
| `user_id` | `Optional[UUID]` | Owner of conversation |
| `is_public` | `bool` | Whether visible to all users |
| `prompt_id` | `Optional[UUID]` | Template used to initialize |
| `template_version_id` | `Optional[UUID]` | Template version used |
| `system_prompt` | `Optional[str]` | Rendered system prompt |
| `variable_values` | `Optional[Dict[str, str]]` | Variable values used |
| `messages` | `List[Message]` | Conversation messages (loaded) |
| `message_count` | `int` | Cached message count |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

### Prompt (Template Alias)

Stored prompt template (alias for Template domain model).

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Prompt identifier |
| `name` | `str` | Unique prompt name |
| `content` | `str` | Jinja2 template content |
| `description` | `Optional[str]` | Prompt description |
| `variables` | `List[str]` | Variable names in template |
| `metadata` | `Dict[str, Any]` | LLM config (model, temperature, max_tokens) |
| `tags` | `List[str]` | Tags for filtering |
| `user_id` | `Optional[UUID]` | Prompt owner |
| `is_public` | `bool` | Whether visible to all users |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

### VariableDefinition

Variable definition with type and validation.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Variable name |
| `description` | `str` | Human-readable description |
| `variable_type` | `VariableType` | Type (text, number, boolean, select) |
| `is_required` | `bool` | Whether variable is required |
| `default_value` | `Optional[str]` | Default value |
| `options` | `Optional[List[str]]` | Valid options (for SELECT type) |
| `validation_pattern` | `Optional[str]` | Regex validation pattern |

### MessageRole Enum

| Value | Description |
|-------|-------------|
| `SYSTEM` | System instructions (sets behavior) |
| `USER` | User input messages |
| `ASSISTANT` | LLM-generated responses |

### VariableType Enum

| Value | Description |
|-------|-------------|
| `TEXT` | Free-form text input |
| `NUMBER` | Numeric value with validation |
| `BOOLEAN` | True/false value |
| `SELECT` | Choice from predefined options |

## Best Practices

### 1. Use System Messages for Consistent Behavior

Always start conversations with a system message to define assistant behavior.

```python
# GOOD - Clear system instructions
messages = [
    Message(
        role=MessageRole.SYSTEM,
        content="You are a technical support assistant. Be concise, "
                "provide step-by-step instructions, and always ask for "
                "clarification if needed."
    ),
    Message(role=MessageRole.USER, content="My printer won't connect."),
]

# BAD - No system message, inconsistent responses
messages = [
    Message(role=MessageRole.USER, content="My printer won't connect."),
]
```

**Why**: System messages set consistent behavior across all interactions, improving response quality and predictability.

### 2. Control Token Usage with max_tokens

Set explicit `max_tokens` limits to control costs and response length.

```python
# GOOD - Explicit token limit
response = await llm_service.complete_simple(
    content="Summarize this article in 100 words",
    max_tokens=150,  # Allows for ~100 word response
    temperature=0.5
)

# BAD - No token limit, potentially expensive responses
response = await llm_service.complete_simple(
    content="Tell me everything about quantum physics",
    # No max_tokens - could generate thousands of tokens
)
```

**Why**: Unbounded token generation can lead to unexpected costs and unnecessarily long responses.

### 3. Use Lower Temperature for Factual Responses

Adjust temperature based on use case: low for factual, high for creative.

```python
# GOOD - Low temperature for factual Q&A
factual_response = await llm_service.complete_simple(
    content="What is the capital of France?",
    temperature=0.2  # Deterministic, factual
)

# GOOD - Higher temperature for creative writing
creative_response = await llm_service.complete_simple(
    content="Write a short poem about autumn",
    temperature=0.9  # Creative, varied
)

# BAD - High temperature for factual questions
bad_response = await llm_service.complete_simple(
    content="What is 2 + 2?",
    temperature=0.9  # May produce incorrect or creative answers
)
```

**Why**: Temperature controls randomness. Low values produce consistent, factual responses; high values enable creativity but reduce reliability.

### 4. Store Prompts as Templates, Not Hardcoded Strings

Use the prompt registry for reusable, versioned prompts.

```python
# GOOD - Stored prompt template
prompt_request = create_prompt_request(
    name="support_response",
    template="Respond to this support ticket: { { ticket_content } }\n\n"
             "Customer tier: { { tier } }",
    variables=["ticket_content", "tier"],
    default_temperature=0.5,
    tags=["support"]
)
await llm_service.create_prompt(prompt_request)

# Later usage
response = await llm_service.complete_from_prompt(
    prompt_name_or_id="support_response",
    variables={"ticket_content": ticket, "tier": "premium"}
)

# BAD - Hardcoded prompt strings
prompt = f"Respond to this support ticket: {ticket}\n\nCustomer tier: {tier}"
response = await llm_service.complete_simple(prompt)
```

**Why**: Stored templates enable version control, A/B testing, non-developer editing, and centralized management of prompts.

### 5. Validate Variables Before Rendering

Always validate variable values against definitions before template rendering.

```python
# GOOD - Validation before rendering
variable_defs = await variable_service.get_prompt_variables_info(prompt)
errors = await variable_service.validate_variable_values(variable_defs, user_input)

if errors:
    return {"error": "Invalid input", "details": errors}

response = await llm_service.complete_from_prompt(
    prompt_name_or_id=prompt.id,
    variables=user_input
)

# BAD - No validation, may cause rendering errors
response = await llm_service.complete_from_prompt(
    prompt_name_or_id=prompt.id,
    variables=user_input  # May be missing required variables or wrong types
)
```

**Why**: Early validation provides clear error messages and prevents template rendering failures at runtime.

### 6. Use Conversation Threads for Multi-Turn Interactions

For applications with ongoing conversations, use ConversationService instead of manual message management.

```python
# GOOD - Persistent conversation with automatic history management
conversation = await conversation_service.create_conversation(
    CreateConversationRequest(title="Support Session", user_id=user.id)
)

# First message
await conversation_service.send_message_and_get_response(
    conversation_id=conversation.id,
    user_message="I need help with my account"
)

# Follow-up (history automatically included)
await conversation_service.send_message_and_get_response(
    conversation_id=conversation.id,
    user_message="I can't reset my password"
)

# BAD - Manual message management without persistence
messages = []
messages.append(Message(role=MessageRole.USER, content="I need help"))
response1 = await llm_service.complete_from_messages(messages)
messages.append(response1.message)

# Loses history if user refreshes page or session ends
messages.append(Message(role=MessageRole.USER, content="Can't reset password"))
response2 = await llm_service.complete_from_messages(messages)
```

**Why**: ConversationService provides persistence, pagination, and automatic history management, essential for real user interactions.

### 7. Handle LLM Errors Gracefully

Wrap LLM calls in try-except blocks and provide fallback behavior.

```python
# GOOD - Error handling with fallback
try:
    response = await llm_service.complete_simple(
        content=user_question,
        temperature=0.5,
        max_tokens=500
    )
    return {"answer": response}

except LLMError as e:
    logger.error(f"LLM error: {e}")
    return {
        "answer": "I'm having trouble processing your request. Please try again.",
        "error": "llm_unavailable"
    }

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {
        "answer": "An unexpected error occurred. Please contact support.",
        "error": "internal_error"
    }

# BAD - No error handling
response = await llm_service.complete_simple(content=user_question)
return {"answer": response}  # May crash on API errors, rate limits, etc.
```

**Why**: LLM APIs can fail due to rate limits, network issues, or service outages. Graceful degradation maintains user experience.

## Security Considerations

### 1. API Key Protection

Never expose API keys in code, logs, or error messages.

```python
# GOOD - API key from environment
import os

app = compose.webapp(
    database_url=os.getenv("DATABASE_URL"),
    kits=[
        compose.llm(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),  # From environment
        ),
    ],
)

# BAD - Hardcoded API key
app = compose.webapp(
    kits=[
        compose.llm(
            provider="openai",
            api_key="sk-proj-abc123...",  # Exposed in code
        ),
    ],
)
```

### 2. Input Sanitization

Sanitize user inputs before passing to LLM to prevent prompt injection.

```python
def sanitize_user_input(text: str) -> str:
    """Remove potentially harmful instruction patterns."""
    # Remove instruction-like patterns
    text = re.sub(r"ignore (previous|above|all) instructions?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"system message:", "", text, flags=re.IGNORECASE)

    # Limit length
    return text[:5000]

# GOOD - Sanitized input
user_question = sanitize_user_input(request.form["question"])
response = await llm_service.complete_simple(user_question)

# BAD - Raw user input
response = await llm_service.complete_simple(request.form["question"])
```

### 3. Access Control for Prompts and Conversations

Verify ownership before accessing conversations or prompts.

```python
# GOOD - Access control check
conversation = await conversation_service.get_conversation(conversation_id)

if conversation.user_id != current_user.id and not current_user.is_admin:
    raise AuthorizationError("Cannot access this conversation")

# BAD - No access control
conversation = await conversation_service.get_conversation(conversation_id)
# Anyone can access any conversation
```

### 4. Rate Limiting

Implement rate limiting to prevent abuse and control costs.

```python
from portico.exceptions import RateLimitError

# GOOD - Rate limiting
if not await rate_limiter.check_limit(user_id, "llm_calls", max_per_hour=100):
    raise RateLimitError("LLM rate limit exceeded. Try again in an hour.")

response = await llm_service.complete_simple(user_question)

# BAD - No rate limiting
# Users can make unlimited expensive LLM calls
response = await llm_service.complete_simple(user_question)
```

## FAQs

### 1. How do I choose between OpenAI and Anthropic?

Both providers offer high-quality models with different strengths:

**OpenAI (GPT models)**:

- **Best for**: Function calling, JSON mode, structured outputs, vision tasks
- **Models**: gpt-4o (latest), gpt-4o-mini (fast/cheap), gpt-3.5-turbo (legacy)
- **Strengths**: Excellent at following instructions, strong code generation

**Anthropic (Claude models)**:

- **Best for**: Long-form content, analysis, creative writing, safety-critical applications
- **Models**: claude-3-5-sonnet (balanced), claude-3-opus (most capable), claude-3-haiku (fast)
- **Strengths**: Larger context windows (200k tokens), strong reasoning, more "thoughtful" responses

```python
# Switch providers by changing config
compose.llm(provider="openai", api_key=openai_key, model="gpt-4o-mini")
compose.llm(provider="anthropic", api_key=anthropic_key, model="claude-3-5-sonnet-20241022")
```

### 2. How do I handle long conversations that exceed context limits?

Use conversation summarization or sliding window approaches:

```python
async def handle_long_conversation(conversation_id: UUID) -> ChatCompletionResponse:
    messages = await conversation_service.get_conversation_messages(conversation_id)

    # If conversation is too long, summarize older messages
    if len(messages) > 20:
        # Keep system message and recent messages
        recent_messages = [messages[0]] + messages[-10:]

        # Summarize middle messages
        middle_content = "\n".join([m.content for m in messages[1:-10]])
        summary_response = await llm_service.complete_simple(
            content=f"Summarize this conversation: {middle_content}",
            max_tokens=200
        )

        # Insert summary as system message
        summary_msg = Message(role=MessageRole.SYSTEM, content=f"Previous conversation summary: {summary_response}")
        messages = [messages[0], summary_msg] + recent_messages

    return await llm_service.complete_from_messages(messages)
```

### 3. How do I test code that uses LLMService?

Use mock providers or create test fixtures:

```python
import pytest
from unittest.mock import AsyncMock
from portico.kits.llm import LLMService
from portico.ports.llm import ChatCompletionResponse, Message, MessageRole, Usage

@pytest.fixture
def mock_llm_service():
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = ChatCompletionResponse(
        id="test_123",
        model="gpt-4o-mini",
        message=Message(role=MessageRole.ASSISTANT, content="Test response"),
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )

    return LLMService(completion_provider=mock_provider)

@pytest.mark.asyncio
async def test_user_query_handler(mock_llm_service):
    result = await handle_user_query("What is Python?", mock_llm_service)

    assert result["answer"] == "Test response"
    assert result["tokens_used"] == 15
```

### 4. How do I implement streaming responses?

Streaming is supported via the `stream=True` parameter in `ChatCompletionRequest`:

```python
request = ChatCompletionRequest(
    messages=[Message(role=MessageRole.USER, content="Write a story")],
    stream=True
)

# For streaming, use provider directly (not yet fully supported in service layer)
async for chunk in llm_service.completion_provider.stream(request):
    print(chunk.message.content, end="", flush=True)
```

Note: Full streaming support in LLMService is under development. Current implementation returns complete responses.

### 5. How do I track token usage and costs?

Monitor token usage from `ChatCompletionResponse.usage`:

```python
response = await llm_service.complete_from_messages(messages)

# Track usage
tokens_used = response.usage.total_tokens
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens

# Calculate cost (example pricing for gpt-4o-mini)
cost_per_1k_prompt = 0.00015  # $0.15 per 1M tokens
cost_per_1k_completion = 0.0006  # $0.60 per 1M tokens

cost = (prompt_tokens / 1000 * cost_per_1k_prompt +
        completion_tokens / 1000 * cost_per_1k_completion)

# Store in database for billing
await analytics.track_llm_usage(
    user_id=user.id,
    model=response.model,
    tokens=tokens_used,
    cost=cost
)
```

### 6. How do I implement RAG (Retrieval-Augmented Generation)?

Inject retrieved context using the `rag_context` parameter:

```python
# Retrieve relevant context from vector store
from portico.kits.rag import RAGService

rag_service = app.kits["rag"].service
search_results = await rag_service.search(query="Portico architecture", top_k=3)

# Format context
context = "\n\n".join([result.content for result in search_results])

# Inject into LLM completion
response = await llm_service.complete_from_messages(
    messages=[
        Message(role=MessageRole.USER, content="How does Portico implement hexagonal architecture?")
    ],
    rag_context=context,  # Injected as additional context
    temperature=0.3
)
```

For full RAG capabilities, use the RAG Kit which handles embedding, vector storage, and context retrieval automatically.

### 7. How do I version prompts for A/B testing?

Use tags and metadata to track prompt versions:

```python
# Version 1
prompt_v1 = await llm_service.create_prompt(
    create_prompt_request(
        name="welcome_message_v1",
        template="Welcome to our service, { { user_name } }!",
        tags=["welcome", "v1", "control"],
        metadata={"version": 1, "experiment": "welcome_msg_test"}
    )
)

# Version 2
prompt_v2 = await llm_service.create_prompt(
    create_prompt_request(
        name="welcome_message_v2",
        template="Hey { { user_name } }, great to see you here!",
        tags=["welcome", "v2", "variant"],
        metadata={"version": 2, "experiment": "welcome_msg_test"}
    )
)

# Randomly select version for user
prompt_name = random.choice(["welcome_message_v1", "welcome_message_v2"])
response = await llm_service.complete_from_prompt(
    prompt_name_or_id=prompt_name,
    variables={"user_name": user.name}
)

# Track which version was used
await analytics.track_prompt_usage(
    user_id=user.id,
    prompt_name=prompt_name,
    response_quality=user_feedback
)
```

### 8. How do I handle multi-language conversations?

Specify language in system message or prompt template:

```python
# Language-specific system message
system_message = Message(
    role=MessageRole.SYSTEM,
    content=f"You are a helpful assistant. Respond in {user_language}."
)

messages = [system_message, user_message]
response = await llm_service.complete_from_messages(messages)

# Or use prompt templates with language variable
prompt_request = create_prompt_request(
    name="multilingual_support",
    template="Respond to the user in { { language } }: { { user_question } }",
    variables=["language", "user_question"]
)

response = await llm_service.complete_from_prompt(
    prompt_name_or_id="multilingual_support",
    variables={"language": "Spanish", "user_question": question}
)
```

### 9. How do I implement function calling (tool use)?

Function calling is provider-specific and handled through provider extensions:

```python
# OpenAI function calling
from openai import OpenAI

# Access underlying provider for advanced features
provider = llm_service.completion_provider

if isinstance(provider, OpenAIProvider):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]

    # Make completion with tools
    response = await provider.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in SF?"}],
        tools=tools
    )

    # Handle tool calls
    if response.choices[0].message.tool_calls:
        # Execute function and return result
        pass
```

Note: Function calling is an advanced feature requiring direct provider access.

### 10. How do I monitor LLM performance and quality?

Implement logging and metrics collection:

```python
import time
from portico.kits.logging import get_logger

logger = get_logger(__name__)

async def monitored_llm_call(user_question: str) -> str:
    start_time = time.time()

    try:
        response = await llm_service.complete_simple(
            content=user_question,
            temperature=0.7,
            max_tokens=500
        )

        # Log success metrics
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "llm_completion_success",
            duration_ms=duration_ms,
            tokens=response.usage.total_tokens if response.usage else None,
            model=response.model,
            question_length=len(user_question)
        )

        return response.message.content

    except Exception as e:
        # Log failure
        logger.error(
            "llm_completion_failed",
            error=str(e),
            question_length=len(user_question)
        )
        raise

# Set up dashboards to track:
# - Average response time
# - Token usage trends
# - Error rates by model
# - User satisfaction scores
```

## Related Ports

- **Template Port** - Prompt storage and rendering (used by LLMService)
- **Cache Port** - Cache LLM responses to reduce costs
- **RAG Port** - Full RAG pipeline with embedding and vector search
- **Audit Port** - Track LLM usage for compliance

## Architecture Notes

The LLM Kit is a **stateless kit** with optional database storage for conversations and prompts. The core LLM completion functionality does not require database access.

**Key Architectural Decisions**:

- **Port-based provider abstraction**: Applications depend on `ChatCompletionProvider` interface, not specific providers (OpenAI, Anthropic)
- **Composition root pattern**: Providers are instantiated only in `compose.py`, maintaining hexagonal architecture
- **Template integration**: Reuses Template Port for prompt storage, avoiding duplication
- **Conversation persistence**: Optional ConversationService provides stateful conversation management
- **Variable validation**: VariableService enables type-safe prompt inputs with runtime validation

The LLM Kit follows hexagonal architecture by depending on ports rather than concrete implementations, enabling flexible provider swapping and comprehensive testing with mock providers.
