# LLM Port

## Overview

The LLM Port defines the contract for Large Language Model chat completion operations, conversation management, and prompt template integration in Portico applications.

**Purpose**: Abstract LLM provider operations to enable pluggable LLM backends (OpenAI, Anthropic, local models) with consistent conversation storage and prompt template management.

**Domain**: Large Language Model integration, chat completions, conversation management, prompt engineering

**Key Capabilities**:

- Chat completion with streaming support
- Multiple LLM provider support (OpenAI, Anthropic)
- Conversation thread storage and retrieval
- User ownership and access control for conversations
- Template-based prompt management (via Template Port)
- Conversation variable tracking and reproduction
- Message history management
- Token usage tracking
- RAG context injection

**Port Type**: Adapter + Repository (multi-interface)

**When to Use**:

- Chat applications with LLM integration
- AI assistants and chatbots
- Template-based prompt management systems
- Multi-user conversation platforms
- RAG (Retrieval-Augmented Generation) applications
- Applications requiring conversation history
- Prompt engineering and experimentation tools

## Domain Models

### Message

A single message in a conversation. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `role` | `MessageRole` | Yes | - | Message role (SYSTEM, USER, or ASSISTANT) |
| `content` | `str` | Yes | - | Message text content |

**Example**:

```python
from portico.ports.llm import Message, MessageRole

# System message
system_msg = Message(
    role=MessageRole.SYSTEM,
    content="You are a helpful assistant specialized in Python programming."
)

# User message
user_msg = Message(
    role=MessageRole.USER,
    content="How do I read a CSV file in Python?"
)

# Assistant message
assistant_msg = Message(
    role=MessageRole.ASSISTANT,
    content="You can use the csv module or pandas library..."
)
```

### ChatCompletionRequest

Request for chat completion with configuration parameters. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `messages` | `List[Message]` | Yes | - | Conversation messages |
| `model` | `Optional[str]` | No | `None` | Model name override (provider default if None) |
| `temperature` | `Optional[float]` | No | `None` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `Optional[int]` | No | `None` | Maximum tokens to generate |
| `top_p` | `Optional[float]` | No | `None` | Nucleus sampling (0.0-1.0) |
| `frequency_penalty` | `Optional[float]` | No | `None` | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | `Optional[float]` | No | `None` | Presence penalty (-2.0 to 2.0) |
| `stop` | `Optional[str \| List[str]]` | No | `None` | Stop sequences |
| `stream` | `bool` | No | `False` | Whether to stream response |
| `rag_context` | `Optional[str]` | No | `None` | RAG context to inject (OpenAI only) |

**Example**:

```python
from portico.ports.llm import ChatCompletionRequest, Message, MessageRole

request = ChatCompletionRequest(
    messages=[
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is Python?")
    ],
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)

response = await chat_provider.complete(request)
```

### ChatCompletionResponse

Response from chat completion. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | - | Unique response identifier |
| `model` | `str` | Yes | - | Model used for generation |
| `message` | `Message` | Yes | - | Generated assistant message |
| `usage` | `Optional[Usage]` | No | `None` | Token usage statistics |
| `created_at` | `datetime` | No | Current UTC time | Response timestamp |

**Example**:

```python
response = await chat_provider.complete(request)

print(f"Response: {response.message.content}")
print(f"Model: {response.model}")
if response.usage:
    print(f"Tokens used: {response.usage.total_tokens}")
```

### Usage

Token usage information for cost tracking. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt_tokens` | `int` | Yes | - | Tokens in input prompt |
| `completion_tokens` | `int` | Yes | - | Tokens in generated response |
| `total_tokens` | `int` | Yes | - | Total tokens consumed |

### Conversation

A conversation thread with user ownership and template integration. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | `uuid4()` | Unique conversation identifier |
| `title` | `str` | Yes | - | Conversation title |
| `user_id` | `Optional[UUID]` | No | `None` | Owner user ID (None = system) |
| `is_public` | `bool` | No | `False` | Public visibility |
| `prompt_id` | `Optional[UUID]` | No | `None` | Template used to start conversation |
| `template_version_id` | `Optional[UUID]` | No | `None` | Template version used |
| `system_prompt` | `Optional[str]` | No | `None` | Rendered system prompt |
| `variable_values` | `Optional[Dict[str, str]]` | No | `None` | Variables used in rendering |
| `messages` | `List[Message]` | No | `[]` | Conversation messages (loaded separately) |
| `message_count` | `int` | No | `0` | Cached message count |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |
| `updated_at` | `datetime` | No | Current UTC time | Last update timestamp |

**Methods**:

- `is_owned_by(user_id: UUID) -> bool` - Check if conversation is owned by user
- `is_accessible_by(user_id: Optional[UUID]) -> bool` - Check if conversation is accessible (public, owned, or system)
- `was_created_from_template() -> bool` - Check if conversation has associated template
- `can_reproduce() -> bool` - Check if conversation can be reproduced (has template + variables)

**Example**:

```python
from portico.ports.llm import Conversation, Message, MessageRole

conversation = Conversation(
    title="Python Help Session",
    user_id=user_id,
    is_public=False,
    prompt_id=template_id,
    system_prompt="You are a Python expert assistant.",
    variable_values={"expertise": "Python"},
    messages=[
        Message(role=MessageRole.SYSTEM, content="You are a Python expert."),
        Message(role=MessageRole.USER, content="How do I install packages?")
    ]
)

# Check if reproducible
if conversation.can_reproduce():
    print("This conversation can be reproduced from the template")
```

### CreateConversationRequest

Request for creating a new conversation.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `title` | `str` | Yes | - | Conversation title |
| `user_id` | `Optional[UUID]` | No | `None` | Owner user ID |
| `is_public` | `bool` | No | `False` | Public visibility |
| `prompt_id` | `Optional[UUID]` | No | `None` | Template ID for template-based conversation |
| `template_version_id` | `Optional[UUID]` | No | `None` | Template version ID |
| `system_prompt` | `Optional[str]` | No | `None` | Rendered system prompt |
| `variable_values` | `Optional[Dict[str, str]]` | No | `None` | Variable values used |
| `initial_message` | `Optional[str]` | No | `None` | First user message |

**Example**:

```python
from portico.ports.llm import CreateConversationRequest

request = CreateConversationRequest(
    title="Code Review Session",
    user_id=current_user_id,
    prompt_id=code_review_template_id,
    variable_values={"language": "Python", "focus": "best practices"},
    initial_message="Please review this code: ..."
)

conversation = await conversation_repo.create(request)
```

### UpdateConversationRequest

Request for updating conversation metadata. All fields optional.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `title` | `Optional[str]` | No | `None` | New title |
| `is_public` | `Optional[bool]` | No | `None` | New public status |

### VariableDefinition

Definition of a variable for prompt templates. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | `uuid4()` | Unique identifier |
| `name` | `str` | Yes | - | Variable name |
| `description` | `Optional[str]` | No | `None` | Variable description |
| `variable_type` | `VariableType` | No | `TEXT` | Variable type (TEXT, NUMBER, BOOLEAN, SELECT) |
| `default_value` | `Optional[str]` | No | `None` | Default value |
| `options` | `List[str]` | No | `[]` | Options for SELECT type |
| `is_required` | `bool` | No | `True` | Whether variable is required |
| `validation_pattern` | `Optional[str]` | No | `None` | Regex validation pattern |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |
| `updated_at` | `datetime` | No | Current UTC time | Update timestamp |

**Example**:

```python
from portico.ports.llm import VariableDefinition, VariableType

# Text variable
var_def = VariableDefinition(
    name="language",
    description="Programming language for code examples",
    variable_type=VariableType.TEXT,
    is_required=True
)

# Select variable with options
tone_var = VariableDefinition(
    name="tone",
    description="Conversation tone",
    variable_type=VariableType.SELECT,
    options=["professional", "casual", "friendly"],
    default_value="professional"
)
```

### ConversationVariable

A variable value associated with a conversation. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | `uuid4()` | Unique identifier |
| `conversation_id` | `UUID` | Yes | - | Associated conversation ID |
| `variable_name` | `str` | Yes | - | Variable name |
| `variable_value` | `str` | Yes | - | Variable value |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

## Enumerations

### MessageRole

Valid roles for chat messages.

| Value | Description |
|-------|-------------|
| `SYSTEM` | System instruction message |
| `USER` | User input message |
| `ASSISTANT` | AI-generated response message |

**Example**:

```python
from portico.ports.llm import MessageRole

# Use enum for type safety
Message(role=MessageRole.SYSTEM, content="You are helpful.")
Message(role=MessageRole.USER, content="Hello!")
Message(role=MessageRole.ASSISTANT, content="Hi there!")
```

### VariableType

Valid types for prompt variables.

| Value | Description |
|-------|-------------|
| `TEXT` | Free-form text input |
| `NUMBER` | Numeric input |
| `BOOLEAN` | True/false value |
| `SELECT` | Selection from predefined options |

## Port Interfaces

### ChatCompletionProvider

The `ChatCompletionProvider` abstract base class defines the contract for LLM chat completion operations.

**Location**: `portico.ports.llm.ChatCompletionProvider`

#### Key Methods

##### complete

```python
async def complete(request: ChatCompletionRequest) -> ChatCompletionResponse
```

Generate a chat completion. Primary method for LLM interactions.

**Parameters**:

- `request`: Chat completion request with messages and configuration

**Returns**: ChatCompletionResponse containing generated message and usage stats.

**Example**:

```python
from portico.ports.llm import ChatCompletionRequest, Message, MessageRole

# Simple completion
request = ChatCompletionRequest(
    messages=[
        Message(role=MessageRole.SYSTEM, content="You are a helpful coding assistant."),
        Message(role=MessageRole.USER, content="Write a Python function to reverse a string.")
    ],
    model="gpt-4",
    temperature=0.7,
    max_tokens=200
)

response = await chat_provider.complete(request)
print(response.message.content)

# Multi-turn conversation
messages = [
    Message(role=MessageRole.SYSTEM, content="You are a math tutor."),
    Message(role=MessageRole.USER, content="What is 5 + 3?"),
    Message(role=MessageRole.ASSISTANT, content="5 + 3 equals 8."),
    Message(role=MessageRole.USER, content="What about 8 * 2?")
]

request = ChatCompletionRequest(messages=messages, temperature=0.3)
response = await chat_provider.complete(request)
```

##### list_models

```python
async def list_models() -> List[str]
```

List available models from the provider. Returns list of model names.

**Example**:

```python
models = await chat_provider.list_models()
print(f"Available models: {models}")
# ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
```

### ConversationRepository

The `ConversationRepository` abstract base class defines the contract for conversation storage and retrieval.

**Location**: `portico.ports.llm.ConversationRepository`

#### Key Methods

##### create

```python
async def create(conversation_data: CreateConversationRequest) -> Conversation
```

Create a new conversation. Primary method for starting conversations.

**Parameters**:

- `conversation_data`: Request with conversation creation data

**Returns**: Created Conversation object.

**Example**:

```python
from portico.ports.llm import CreateConversationRequest

# Simple conversation
request = CreateConversationRequest(
    title="Python Q&A",
    user_id=current_user_id
)
conversation = await conversation_repo.create(request)

# Template-based conversation
request = CreateConversationRequest(
    title="Code Review: auth.py",
    user_id=current_user_id,
    prompt_id=code_review_template_id,
    system_prompt="You are an expert Python code reviewer focusing on security.",
    variable_values={"language": "Python", "focus": "security"}
)
conversation = await conversation_repo.create(request)
```

##### add_message

```python
async def add_message(conversation_id: UUID, message: Message) -> Optional[Conversation]
```

Add a message to a conversation. Primary method for conversation interaction.

**Parameters**:

- `conversation_id`: Conversation identifier
- `message`: Message to add

**Returns**: Updated Conversation object if found, None otherwise.

**Example**:

```python
from portico.ports.llm import Message, MessageRole

# Add user message
user_message = Message(
    role=MessageRole.USER,
    content="How do I handle exceptions in Python?"
)
updated = await conversation_repo.add_message(conversation_id, user_message)

# Add assistant response
assistant_message = Message(
    role=MessageRole.ASSISTANT,
    content="In Python, you use try-except blocks..."
)
updated = await conversation_repo.add_message(conversation_id, assistant_message)
```

#### Other Methods

##### get_by_id

```python
async def get_by_id(conversation_id: UUID, user_id: Optional[UUID] = None) -> Optional[Conversation]
```

Retrieve a conversation by ID with access control. Returns Conversation if found and accessible, None otherwise.

##### update

```python
async def update(
    conversation_id: UUID,
    update_data: UpdateConversationRequest,
    user_id: Optional[UUID] = None
) -> Optional[Conversation]
```

Update conversation metadata. Verifies ownership if user_id provided. Returns updated Conversation or None.

##### delete

```python
async def delete(conversation_id: UUID, user_id: Optional[UUID] = None) -> bool
```

Delete a conversation. Verifies ownership if user_id provided. Returns True if deleted, False if not found.

##### list_conversations

```python
async def list_conversations(
    user_id: Optional[UUID] = None,
    include_public: bool = True,
    prompt_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Conversation]
```

List conversations with filtering and pagination. Returns user's conversations + public + system if user_id provided.

##### get_messages

```python
async def get_messages(conversation_id: UUID, limit: int = 100, offset: int = 0) -> List[Message]
```

Get messages from a conversation with pagination. Returns list of Message objects.

##### increment_message_count

```python
async def increment_message_count(conversation_id: UUID, increment: int = 1) -> None
```

Increment the message count for a conversation. Used for caching message counts.

### VariableRepository

The `VariableRepository` abstract base class defines the contract for prompt variable management.

**Location**: `portico.ports.llm.VariableRepository`

#### Key Methods

##### create_definition

```python
async def create_definition(definition_data: CreateVariableDefinitionRequest) -> VariableDefinition
```

Create a new variable definition. Returns created VariableDefinition.

##### set_conversation_variables

```python
async def set_conversation_variables(conversation_id: UUID, variables: Dict[str, str]) -> List[ConversationVariable]
```

Set variable values for a conversation. Returns list of ConversationVariable objects.

#### Other Methods

##### get_definition_by_name

```python
async def get_definition_by_name(name: str) -> Optional[VariableDefinition]
```

Retrieve a variable definition by name. Returns VariableDefinition or None.

##### list_definitions

```python
async def list_definitions(limit: int = 100, offset: int = 0) -> List[VariableDefinition]
```

List variable definitions with pagination.

##### get_conversation_variables

```python
async def get_conversation_variables(conversation_id: UUID) -> Dict[str, str]
```

Get variable values for a conversation. Returns dictionary mapping variable names to values.

##### delete_conversation_variables

```python
async def delete_conversation_variables(conversation_id: UUID) -> bool
```

Delete all variable values for a conversation. Returns True if deleted, False if none found.

## Helper Functions

### create_prompt_request

```python
def create_prompt_request(
    name: str,
    template: str,
    description: Optional[str] = None,
    variables: Optional[List[str]] = None,
    default_model: Optional[str] = None,
    default_temperature: Optional[float] = None,
    default_max_tokens: Optional[int] = None,
    tags: Optional[List[str]] = None
) -> CreateTemplateRequest
```

Helper to create LLM prompt templates with proper metadata. Returns CreateTemplateRequest configured for LLM prompts.

**Example**:

```python
from portico.ports.llm import create_prompt_request

request = create_prompt_request(
    name="code_explainer",
    template="Explain this {{ language }} code:\n\n{{ code }}",
    description="Code explanation assistant",
    variables=["language", "code"],
    default_model="gpt-4",
    default_temperature=0.3,
    default_max_tokens=1000,
    tags=["code", "education"]
)

template = await template_registry.create(request)
```

## Common Patterns

### Multi-Turn Chat Conversation

```python
from portico.ports.llm import (
    ChatCompletionProvider,
    ConversationRepository,
    Message,
    MessageRole,
    ChatCompletionRequest,
    CreateConversationRequest
)

async def chat_session(
    chat_provider: ChatCompletionProvider,
    conversation_repo: ConversationRepository,
    user_id: UUID
):
    """Interactive multi-turn chat session."""

    # Create conversation
    conversation = await conversation_repo.create(
        CreateConversationRequest(
            title="Python Help",
            user_id=user_id
        )
    )

    # Add system message
    system_msg = Message(
        role=MessageRole.SYSTEM,
        content="You are a helpful Python programming assistant."
    )
    await conversation_repo.add_message(conversation.id, system_msg)

    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Add user message
        user_msg = Message(role=MessageRole.USER, content=user_input)
        await conversation_repo.add_message(conversation.id, user_msg)

        # Get all messages for context
        messages = await conversation_repo.get_messages(conversation.id)

        # Generate response
        request = ChatCompletionRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7
        )
        response = await chat_provider.complete(request)

        # Add assistant response
        await conversation_repo.add_message(conversation.id, response.message)

        print(f"Assistant: {response.message.content}\n")
```

### Template-Based Conversation with RAG

```python
from portico.ports.llm import create_prompt_request
from portico.ports.template import TemplateRegistry
from portico.adapters.template import Jinja2TemplateRenderer

async def create_rag_conversation(
    template_registry: TemplateRegistry,
    template_renderer: Jinja2TemplateRenderer,
    conversation_repo: ConversationRepository,
    chat_provider: ChatCompletionProvider,
    user_query: str,
    rag_context: str,
    user_id: UUID
):
    """Create conversation using template with RAG context."""

    # Create prompt template
    prompt_request = create_prompt_request(
        name="rag_assistant",
        template="""You are a helpful assistant. Use the following context to answer questions accurately.

Context:
{{ context }}

Answer the user's question based on the context above.""",
        variables=["context"],
        default_model="gpt-4",
        default_temperature=0.3
    )
    template = await template_registry.create(prompt_request)

    # Render system prompt
    system_prompt = template_renderer.render(
        template.content,
        {"context": rag_context}
    )

    # Create conversation
    conversation = await conversation_repo.create(
        CreateConversationRequest(
            title="RAG Q&A",
            user_id=user_id,
            prompt_id=template.id,
            system_prompt=system_prompt,
            variable_values={"context": rag_context[:100] + "..."}  # Truncate for storage
        )
    )

    # Add messages
    await conversation_repo.add_message(
        conversation.id,
        Message(role=MessageRole.SYSTEM, content=system_prompt)
    )
    await conversation_repo.add_message(
        conversation.id,
        Message(role=MessageRole.USER, content=user_query)
    )

    # Generate response
    messages = await conversation_repo.get_messages(conversation.id)
    response = await chat_provider.complete(
        ChatCompletionRequest(messages=messages, model="gpt-4")
    )

    await conversation_repo.add_message(conversation.id, response.message)

    return conversation, response.message.content
```

## Integration with Kits

The LLM Port is used by the **LLM Kit** to provide high-level chat completion and conversation management services.

```python
from portico import compose

# Configure LLM kit with OpenAI
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.llm(
            provider="openai",
            api_key="sk-...",
            default_model="gpt-4"
        )
    ]
)

await app.initialize()

# Access LLM service
llm_service = app.kits["llm"].service

# Simple completion
response = await llm_service.complete_simple(
    content="What is Python?",
    system_message="You are a programming expert.",
    temperature=0.7
)
print(response)

# Chat with messages
from portico.ports.llm import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are helpful."),
    Message(role=MessageRole.USER, content="Hello!")
]
response = await llm_service.complete_from_messages(messages)
print(response.message.content)

# Use stored prompt template
response = await llm_service.complete_from_prompt(
    prompt_name_or_id="code_explainer",
    variables={"language": "Python", "code": "def hello(): pass"}
)
```

The LLM Kit provides:

- OpenAI and Anthropic provider adapters
- Conversation storage with SQLAlchemy
- Template integration for prompt management
- Token usage tracking
- Streaming support (provider-dependent)

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Use System Messages for Instructions**: Set behavior with system messages, not user messages

   ```python
   # ✅ GOOD: System message for instructions
   messages = [
       Message(role=MessageRole.SYSTEM, content="You are a helpful Python tutor."),
       Message(role=MessageRole.USER, content="How do I use list comprehensions?")
   ]

   # ❌ BAD: Instructions in user message
   messages = [
       Message(role=MessageRole.USER, content="Act as a Python tutor. How do I use list comprehensions?")
   ]
   ```

2. **Track Conversations for Context**: Use ConversationRepository to maintain multi-turn context

   ```python
   # ✅ GOOD: Load full conversation history
   messages = await conversation_repo.get_messages(conversation_id)
   response = await chat_provider.complete(ChatCompletionRequest(messages=messages))

   # ❌ BAD: No context (each message independent)
   response = await chat_provider.complete(
       ChatCompletionRequest(messages=[Message(role=MessageRole.USER, content=user_input)])
   )
   ```

3. **Set Temperature Based on Use Case**: Lower for factual, higher for creative

   ```python
   # ✅ GOOD: Match temperature to task
   # Factual question answering
   request = ChatCompletionRequest(messages=messages, temperature=0.3)

   # Creative writing
   request = ChatCompletionRequest(messages=messages, temperature=0.9)

   # ❌ BAD: High temperature for factual tasks
   request = ChatCompletionRequest(messages=messages, temperature=1.5)
   # Will produce inconsistent/creative answers
   ```

4. **Use Templates for Reusable Prompts**: Store prompts as templates for consistency

   ```python
   # ✅ GOOD: Reusable template
   template = create_prompt_request(
       name="code_reviewer",
       template="Review this {{ language }} code for best practices:\n\n{{ code }}",
       variables=["language", "code"]
   )
   await template_registry.create(template)

   # Use many times
   response = await llm_service.complete_from_prompt(
       "code_reviewer",
       variables={"language": "Python", "code": code_snippet}
   )

   # ❌ BAD: Hardcoded prompts everywhere
   prompt = f"Review this {language} code: {code}"  # Inconsistent
   ```

5. **Monitor Token Usage for Cost Control**: Track usage to manage API costs

   ```python
   # ✅ GOOD: Track and log usage
   response = await chat_provider.complete(request)

   if response.usage:
       logger.info(
           "llm_completion",
           model=response.model,
           prompt_tokens=response.usage.prompt_tokens,
           completion_tokens=response.usage.completion_tokens,
           total_tokens=response.usage.total_tokens
       )

       # Alert if high usage
       if response.usage.total_tokens > 10000:
           logger.warning("high_token_usage", tokens=response.usage.total_tokens)

   # ❌ BAD: Ignore usage
   # Can lead to unexpected costs
   ```

## FAQs

### What LLM providers are supported?

Portico includes adapters for:

- **OpenAI** - GPT-3.5, GPT-4, GPT-4 Turbo models
- **Anthropic** - Claude 3 models (Haiku, Sonnet, Opus)

You can implement custom providers by extending `ChatCompletionProvider`.

### How do I handle streaming responses?

Set `stream=True` in `ChatCompletionRequest`:

```python
request = ChatCompletionRequest(
    messages=messages,
    stream=True
)

# Streaming implementation is provider-specific
# Check provider adapter documentation
```

**Note**: Streaming support varies by provider adapter.

### Can I use different models in the same conversation?

Yes, but it's generally not recommended. Model switching can cause inconsistencies in conversation style and capabilities. If needed:

```python
# Different models for different turns
request1 = ChatCompletionRequest(messages=messages, model="gpt-3.5-turbo")
response1 = await chat_provider.complete(request1)

messages.append(response1.message)
request2 = ChatCompletionRequest(messages=messages, model="gpt-4")
response2 = await chat_provider.complete(request2)
```

### How do I inject RAG context into conversations?

Use the `rag_context` parameter (OpenAI only):

```python
request = ChatCompletionRequest(
    messages=[Message(role=MessageRole.USER, content="What is Portico?")],
    rag_context="Portico is a Python framework for building GPT-powered applications...",
    model="gpt-4"
)

response = await chat_provider.complete(request)
```

For Anthropic, manually add context to the system message.

### How do I implement a custom LLM provider?

Implement the `ChatCompletionProvider` interface:

```python
from portico.ports.llm import (
    ChatCompletionProvider,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Usage
)

class CustomLLMProvider(ChatCompletionProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Call your LLM API
        api_response = await your_llm_api.chat(
            messages=[{"role": m.role.value, "content": m.content} for m in request.messages],
            model=request.model or "default-model",
            temperature=request.temperature
        )

        # Return ChatCompletionResponse
        return ChatCompletionResponse(
            id=api_response["id"],
            model=api_response["model"],
            message=Message(
                role=MessageRole.ASSISTANT,
                content=api_response["content"]
            ),
            usage=Usage(
                prompt_tokens=api_response["usage"]["prompt_tokens"],
                completion_tokens=api_response["usage"]["completion_tokens"],
                total_tokens=api_response["usage"]["total_tokens"]
            )
        )

    async def list_models(self) -> List[str]:
        return await your_llm_api.list_models()
```

Then use in composition:

```python
def llm(**config):
    from your_module import CustomLLMProvider
    from portico.kits.llm import LLMKit

    def factory(database, events):
        provider = CustomLLMProvider(api_key=config["api_key"])
        return LLMKit.create(database, events, config, completion_provider=provider)

    return factory
```
