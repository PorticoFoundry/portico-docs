# Adapters

In Portico's hexagonal architecture, **adapters** are the concrete implementations that connect your business logic to the outside world. While [ports](../ports/index.md) define *what* operations are needed and [kits](../kits/index.md) orchestrate *business logic*, adapters provide the *how* - the actual integration with databases, APIs, cloud services, and external systems.

## What is an Adapter?

An adapter is a concrete implementation of a port interface that handles all the technology-specific details of working with external systems. Think of adapters as translators between your clean domain models and the messy realities of third-party APIs, database schemas, and infrastructure services.

In traditional applications, external integrations often pollute business logic:

```python
# Traditional approach - business logic mixed with infrastructure

from openai import OpenAI
from redis import Redis
import sqlalchemy

class ChatService:
    def __init__(self):
        # Direct dependencies on external libraries
        self.openai_client = OpenAI(api_key="...")
        self.redis = Redis(host="localhost")
        self.db = sqlalchemy.create_engine("postgresql://...")

    async def generate_response(self, prompt: str) -> str:
        # Check cache with Redis-specific code
        cached = self.redis.get(f"chat:{prompt}")
        if cached:
            return cached.decode()

        # Call OpenAI with SDK-specific code
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content

        # Store with Redis-specific code
        self.redis.set(f"chat:{prompt}", result, ex=3600)

        return result
```

This approach creates several problems:

- **Tight coupling** to specific technologies (Redis, OpenAI)
- **Testing requires** running Redis and mocking OpenAI
- **Switching providers** means rewriting the service
- **Configuration details** mixed with business logic

With adapters, infrastructure is isolated:

```python
# Adapter approach - business logic stays clean

from portico.ports.cache import CacheAdapter, CacheKey
from portico.ports.llm import ChatCompletionProvider, ChatCompletionRequest

class ChatService:
    def __init__(
        self,
        llm_provider: ChatCompletionProvider,  # Port interface
        cache: CacheAdapter  # Port interface
    ):
        self.llm = llm_provider
        self.cache = cache

    async def generate_response(self, prompt: str) -> str:
        # Check cache using port interface
        cache_key = CacheKey(key=f"chat:{prompt}")
        cached = await self.cache.get(cache_key)
        if cached:
            return cached.value

        # Call LLM using port interface
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": prompt}]
        )
        response = await self.llm.complete(request)
        result = response.content

        # Store using port interface
        await self.cache.set(cache_key, result, ttl=3600)

        return result
```

The service has **no idea** if it's using OpenAI or Anthropic, Redis or Memcached. All infrastructure details are in adapters that implement the port interfaces.

## Why Adapters Matter

Adapters provide several critical benefits:

### Swap Implementations Freely

Need to switch from OpenAI to Anthropic? Change one line in your composition root:

```python
# Before
app = compose.webapp(
    kits=[compose.llm(provider="openai", api_key="sk-...")]
)

# After - same business logic, different implementation
app = compose.webapp(
    kits=[compose.llm(provider="anthropic", api_key="sk-ant-...")]
)
```

Your kits don't change at all. The adapter handles all provider-specific details.

### Test Without External Dependencies

Adapters make testing fast and reliable:

```python
# Fake adapter for testing
class FakeLLMProvider(ChatCompletionProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(content="Fake response for testing")

# Test without calling real API
fake_llm = FakeLLMProvider()
service = ChatService(llm_provider=fake_llm, cache=fake_cache)

response = await service.generate_response("test prompt")
assert response == "Fake response for testing"
```

No API keys, no network calls, no flaky tests.

### Start Simple, Scale Later

Use lightweight implementations during development, production-grade ones in deployment:

```python
# Development - in-memory, no external services
app = compose.webapp(
    database_url="sqlite:///dev.db",
    kits=[
        compose.cache(backend="memory"),
        compose.llm(provider="openai", api_key="test-key"),
    ]
)

# Production - distributed, resilient
app = compose.webapp(
    database_url="postgresql://prod-db/myapp",
    kits=[
        compose.cache(backend="redis", redis_url="redis://prod-cache"),
        compose.llm(provider="openai", api_key=os.environ["OPENAI_KEY"]),
    ]
)
```

Same business logic, different infrastructure.

## Adapter Anatomy

Every Portico adapter follows consistent structural patterns that make them predictable and composable.

### 1. Port Interface Implementation

Adapters implement port interfaces using Python's abstract base classes:

```python
from portico.ports.cache import CacheAdapter, CacheKey, CacheEntry
from typing import Optional
import redis.asyncio as redis

class RedisCacheAdapter(CacheAdapter):
    """Redis-backed cache implementation.

    Implements the CacheAdapter port using Redis for distributed caching.
    """

    def __init__(self, redis_url: str, db: int = 0):
        """Initialize Redis connection.

        Args:
            redis_url: Redis connection string (redis://host:port)
            db: Redis database number (default: 0)
        """
        self.redis_url = redis_url
        self.db = db
        self._client: Optional[redis.Redis] = None

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve value from Redis cache."""
        client = await self._get_client()
        value = await client.get(key.key)

        if value:
            return CacheEntry(
                key=key.key,
                value=self._deserialize(value),
                ttl=await client.ttl(key.key)
            )
        return None

    async def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in Redis cache."""
        client = await self._get_client()
        serialized = self._serialize(value)
        await client.set(key.key, serialized, ex=ttl)

    async def _get_client(self) -> redis.Redis:
        """Lazy connection initialization."""
        if not self._client:
            self._client = redis.Redis.from_url(self.redis_url, db=self.db)
        return self._client

    def _serialize(self, value: Any) -> bytes:
        """Convert Python object to bytes for storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Convert stored bytes back to Python object."""
        return pickle.loads(data)
```

Key characteristics:

- **Inherits from port interface** (`CacheAdapter`)
- **Implements all abstract methods** required by the port
- **Handles technology specifics** (Redis connection, serialization)
- **Converts between domain and tech models** (CacheKey ↔ Redis keys)
- **Manages resources** (connection pooling, cleanup)

### 2. Configuration

Adapters accept configuration through constructors:

```python
class OpenAIProvider(ChatCompletionProvider):
    """OpenAI LLM provider implementation."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o-mini",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            default_model: Default model to use if not specified in request
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
        """
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize SDK client
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
```

Configuration is:

- **Passed at construction time** (not loaded internally)
- **Validated early** (fail fast if misconfigured)
- **Type-safe** (with type hints)
- **Documented** (with docstrings)

### 3. Domain Model Translation

Adapters translate between domain models (defined in ports) and technology models (from SDKs/libraries):

```python
class OpenAIProvider(ChatCompletionProvider):
    async def complete(
        self,
        request: ChatCompletionRequest  # Domain model from port
    ) -> ChatCompletionResponse:  # Domain model to port
        """Generate chat completion using OpenAI."""

        # Translate domain request to OpenAI SDK format
        openai_messages = [
            {
                "role": msg.role.value.lower(),  # Convert enum to string
                "content": msg.content
            }
            for msg in request.messages
        ]

        # Call OpenAI SDK with provider-specific format
        response = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=openai_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Translate OpenAI response to domain model
        return ChatCompletionResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        )
```


This translation layer:
- **Shields business logic** from SDK changes
- **Normalizes across providers** (OpenAI and Anthropic have different APIs)
- **Enforces domain contracts** (Pydantic validation on domain models)
- **Makes testing easier** (mock domain models, not SDK responses)

### 4. Error Handling

Adapters catch technology-specific errors and translate them to domain exceptions:

```python
from portico.exceptions import LLMError, LLMRateLimitError, LLMInvalidRequestError
from openai import OpenAIError, RateLimitError, APIError
import logging

logger = logging.getLogger(__name__)

class OpenAIProvider(ChatCompletionProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Call OpenAI SDK
            response = await self.client.chat.completions.create(...)
            return self._to_domain_response(response)

        except RateLimitError as e:
            # Translate to domain exception
            logger.warning("openai_rate_limit", error=str(e))
            raise LLMRateLimitError(
                "OpenAI rate limit exceeded. Please try again later."
            ) from e

        except APIError as e:
            # Translate to domain exception
            logger.error("openai_api_error", status=e.status_code, error=str(e))
            if e.status_code == 400:
                raise LLMInvalidRequestError(
                    f"Invalid request to OpenAI: {e.message}"
                ) from e
            raise LLMError(f"OpenAI API error: {e.message}") from e

        except OpenAIError as e:
            # Catch-all for SDK errors
            logger.error("openai_error", error=str(e))
            raise LLMError(f"OpenAI error: {str(e)}") from e

        except Exception as e:
            # Unexpected errors
            logger.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            raise LLMError(f"Unexpected error: {str(e)}") from e
```


Error handling provides:
- **Domain-specific exceptions** that kits understand
- **Consistent error interface** across different adapters
- **Logging** for debugging and monitoring
- **Error chaining** (`from e`) to preserve stack traces
- **Graceful degradation** where appropriate

### 5. Resource Management

Adapters manage external resources (connections, clients, file handles):

```python
class RedisCacheAdapter(CacheAdapter):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        """Lazy initialization of Redis connection."""
        if not self._client:
            self._client = redis.Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
        return self._client

    async def close(self) -> None:
        """Clean up resources on shutdown."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context exit."""
        await self.close()
```


Resource patterns:
- **Lazy initialization** (connect only when needed)
- **Connection pooling** (reuse connections)
- **Cleanup methods** (`close()`, `__aexit__`)
- **Context managers** for automatic cleanup
- **Timeouts** to prevent hanging

### 6. Performance Tracking

Many adapters track metrics for monitoring and optimization:

```python
class MemoryCacheAdapter(CacheAdapter):
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}

        # Performance counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._compressions = 0

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        entry = self._cache.get(key.key)

        if entry:
            self._hits += 1  # Track hit
            return entry
        else:
            self._misses += 1  # Track miss
            return None

    async def get_stats(self) -> CacheStats:
        """Export performance metrics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=len(self._cache),
            total_hits=self._hits,
            total_misses=self._misses,
            total_evictions=self._evictions,
            hit_rate=hit_rate,
            compression_count=self._compressions
        )
```


Metrics help with:
- **Performance monitoring** (hit rates, latency)
- **Capacity planning** (eviction rates, entry counts)
- **Cost optimization** (API call counts, token usage)
- **Debugging** (failure rates, retry counts)

## Adapter Types

Portico adapters fall into several categories based on what they integrate with:

### Infrastructure Adapters

These integrate with foundational infrastructure services:

**Caching**: Memory, Redis, hybrid two-tier caching
**Storage**: Local filesystem, cloud storage (GCS), database BLOBs
**Notifications**: Email, SMS, push notifications
**Session**: Cookie-based, database-backed, distributed


Infrastructure adapters typically:
- Implement **simple, focused operations** (get, set, delete)
- Provide **multiple implementation choices** (local, cloud, distributed)
- Support **configuration tuning** (TTL, size limits, compression)
- Include **development alternatives** (memory-based for testing)

### External Service Adapters

These integrate with third-party APIs and services:

**LLM Providers**: OpenAI, Anthropic, local models
**Embedding Providers**: OpenAI embeddings, custom models
**Managed Platforms**: Graphlit (RAG), Pinecone (vectors)
**Payment Gateways**: Stripe, PayPal (in your apps)


External service adapters typically:
- **Wrap SDK clients** from vendor libraries
- **Handle authentication** (API keys, OAuth)
- **Implement retry logic** for network failures
- **Translate rate limits** to domain exceptions
- **Normalize responses** across providers

### Data Access Adapters

These implement repository interfaces for database operations:

**User Repository**: CRUD operations for users
**Group Repository**: Hierarchical group management
**Audit Repository**: Event logging and queries
**File Metadata Repository**: File tracking


Data access adapters typically:
- **Use SQLAlchemy** for ORM operations
- **Manage transactions** and sessions
- **Convert ORM models** to domain models
- **Handle database-specific** SQL and constraints

### Processing Adapters

These perform data transformation and analysis:

**Document Processor**: Parse and chunk documents
**Template Renderer**: Jinja2 template rendering
**Job Scheduler**: Cron-based job triggering


Processing adapters typically:
- **Implement algorithms** (chunking strategies, rendering)
- **Use specialized libraries** (Jinja2, parsers)
- **Provide configuration options** (chunk size, strategies)
- **Return structured results** (processed documents, rendered templates)

## How Adapters Relate to Ports and Kits

Adapters complete the hexagonal architecture by implementing the port interfaces that kits depend on:

```
┌─────────────────────────────────────────┐
│  Application Layer (Routes, CLI)        │
│  - FastAPI routes                       │
│  - CLI commands                         │
└─────────────────┬───────────────────────┘
                  │ Calls
                  ↓
┌─────────────────────────────────────────┐
│  Kits (Business Logic)                  │
│  - Services orchestrate workflows       │
│  - Repositories abstract data access    │
│  - Depend on PORT INTERFACES            │
└─────────────────┬───────────────────────┘
                  │ Uses
                  ↓
┌─────────────────────────────────────────┐
│  Ports (Interfaces)                     │
│  - Define contracts (ABC/Protocol)      │
│  - Domain models (Pydantic)             │
│  - No implementation                    │
└─────────────────┬───────────────────────┘
                  ↑ Implements
                  │
┌─────────────────────────────────────────┐
│  Adapters (Implementations)             │
│  - Concrete implementations             │
│  - Technology-specific code             │
│  - External library integration         │
└─────────────────────────────────────────┘
```

### The Composition Root Pattern

Here's the critical architectural rule: **Kits never import adapters directly**. Only the composition root (`compose.py`) imports and instantiates adapters:

```python
# portico/compose.py - The ONLY place adapters are imported

def cache(**config):
    """Factory function for cache kit."""
    # Adapter import - ONLY happens here!
    from portico.adapters.cache import RedisCacheAdapter, MemoryCacheAdapter
    from portico.kits.cache import CacheKit

    def factory(database: Database, events: EventBus):
        # Choose adapter based on configuration
        if config.get("backend") == "redis":
            adapter = RedisCacheAdapter(
                redis_url=config["redis_url"],
                db=config.get("db", 0)
            )
        else:
            adapter = MemoryCacheAdapter(
                max_size=config.get("max_memory_items", 1000)
            )

        # Inject adapter into kit
        return CacheKit.create(database, events, config, cache_adapter=adapter)

    return factory
```


This pattern:
- **Enforces clean architecture** (kits can't accidentally couple to adapters)
- **Centralizes configuration** (one place to change implementations)
- **Makes swapping easy** (change one line to use different adapter)
- **Enables testing** (inject fake adapters in tests)

The pattern is **enforced by tooling** - if a kit imports an adapter, `make check-imports` fails:

```bash
$ make check-imports
ERROR: Contract violated: Kits cannot import adapters
  portico/kits/cache/service.py imports portico.adapters.cache.RedisCacheAdapter
```

### Kits Depend on Ports, Not Adapters

```python
# ✅ CORRECT - Kit imports port
from portico.ports.cache import CacheAdapter, CacheKey

class CacheService:
    def __init__(self, cache_adapter: CacheAdapter):  # Interface!
        self.adapter = cache_adapter

    async def get(self, key: str):
        cache_key = CacheKey(key=key)
        return await self.adapter.get(cache_key)

# ❌ WRONG - Kit imports adapter
from portico.adapters.cache import RedisCacheAdapter  # Violation!

class CacheService:
    def __init__(self):
        self.adapter = RedisCacheAdapter(redis_url="...")  # Coupled!
```

## Testing Adapters

Adapters are tested at multiple levels to ensure correctness and reliability.

### Unit Testing Adapters

Test adapter logic in isolation with mocked external dependencies:

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from portico.adapters.llm import OpenAIProvider
from portico.ports.llm import ChatCompletionRequest, Message, MessageRole

@pytest.mark.asyncio
async def test_openai_provider_complete():
    """Test OpenAI adapter translates domain models correctly."""

    # Mock OpenAI SDK response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )

    # Patch OpenAI SDK
    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test adapter
        provider = OpenAIProvider(api_key="test-key")
        request = ChatCompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )

        response = await provider.complete(request)

        # Verify domain model translation
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.usage.total_tokens == 15
```

### Integration Testing Adapters

Test adapters against real external services (with test credentials):

```python
import pytest
from portico.adapters.cache import RedisCacheAdapter
from portico.ports.cache import CacheKey

@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_cache_adapter_integration():
    """Test Redis adapter with real Redis instance."""

    # Use test Redis instance
    adapter = RedisCacheAdapter(redis_url="redis://localhost:6379/15")

    try:
        # Test operations
        key = CacheKey(key="test:integration")

        # Set value
        await adapter.set(key, "test value", ttl=60)

        # Get value
        entry = await adapter.get(key)
        assert entry is not None
        assert entry.value == "test value"

        # Delete value
        deleted = await adapter.delete(key)
        assert deleted is True

        # Verify deleted
        entry = await adapter.get(key)
        assert entry is None

    finally:
        # Cleanup
        await adapter.close()
```

### Testing Kits with Fake Adapters

The real power comes from testing kits with fake adapters:

```python
from portico.ports.cache import CacheAdapter, CacheKey, CacheEntry
from typing import Dict, Optional

class FakeCacheAdapter(CacheAdapter):
    """In-memory fake for testing kits."""

    def __init__(self):
        self._data: Dict[str, CacheEntry] = {}

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        return self._data.get(key.key)

    async def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        self._data[key.key] = CacheEntry(key=key.key, value=value, ttl=ttl)

    async def delete(self, key: CacheKey) -> bool:
        if key.key in self._data:
            del self._data[key.key]
            return True
        return False

# Test kit with fake
@pytest.mark.asyncio
async def test_user_service_with_cache():
    fake_cache = FakeCacheAdapter()
    fake_user_repo = FakeUserRepository()

    service = UserService(
        user_repository=fake_user_repo,
        cache=fake_cache
    )

    # Test caching behavior without running Redis
    user = await service.get_user(user_id)
    assert fake_cache._data  # Cache was populated

    # Second call hits cache
    cached_user = await service.get_user(user_id)
    assert cached_user == user
```

Fake adapters are **fast** (no network), **deterministic** (no flaky tests), and **easy to control** (inject specific behaviors).

## Adapter Composition

Adapters are composed in your application through the `compose.webapp()` function:

### Basic Composition

```python
from portico import compose

app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.cache(backend="redis", redis_url="redis://localhost:6379"),
        compose.llm(provider="openai", api_key="sk-..."),
        compose.file(storage_backend="gcs", gcs_bucket="my-bucket"),
    ]
)
```

The compose system:
1. **Reads configuration** from kit factory functions
2. **Instantiates adapters** based on config
3. **Injects adapters** into kits via dependency injection
4. **Returns configured application** with all kits wired up

### Multiple Adapters per Port

Some kits may use multiple adapters:

```python
# RAG kit uses multiple adapters
app = compose.webapp(
    kits=[
        compose.rag(
            # Each adapter implements a different port
            use_managed_rag=False,  # DIY RAG mode
            vector_store_type="memory",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            document_processor_config={
                "chunking_strategy": "paragraph",
                "chunk_size": 1000,
            }
        )
    ]
)
```

The RAG kit receives:
- **VectorStore adapter** (MemoryVectorStore)
- **EmbeddingProvider adapter** (OpenAIEmbeddingProvider)
- **DocumentProcessor adapter** (BasicDocumentProcessor)

All injected through the composition root.

### Environment-Specific Composition

Use environment variables to configure adapters differently per environment:

```python
import os

# Determine environment
is_production = os.environ.get("ENV") == "production"

app = compose.webapp(
    database_url=os.environ["DATABASE_URL"],
    kits=[
        # Use Redis in production, memory in development
        compose.cache(
            backend="redis" if is_production else "memory",
            redis_url=os.environ.get("REDIS_URL") if is_production else None,
        ),

        # Use different LLM models per environment
        compose.llm(
            provider="openai",
            api_key=os.environ["OPENAI_KEY"],
            model="gpt-4" if is_production else "gpt-4o-mini"
        ),
    ]
)
```

Same application code, different adapters per environment.

## Common Patterns and Best Practices

### Lazy Initialization

Initialize expensive resources only when first needed:

```python
class DatabaseAdapter:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._engine: Optional[Engine] = None

    async def _get_engine(self) -> Engine:
        """Lazy initialization of database engine."""
        if not self._engine:
            self._engine = create_async_engine(
                self.connection_url,
                pool_size=10,
                max_overflow=20
            )
        return self._engine

    async def query(self, sql: str):
        engine = await self._get_engine()  # Initialize on first use
        # ... execute query
```


Benefits:
- **Faster startup** (don't connect to everything immediately)
- **Conditional usage** (only connect if feature is used)
- **Easier testing** (adapter can be created without real connections)

### Connection Pooling

Reuse connections for performance:

```python
class RedisCacheAdapter:
    def __init__(self, redis_url: str, pool_size: int = 10):
        self.redis_url = redis_url
        self.pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=pool_size,
            decode_responses=False
        )
        self._client = redis.Redis(connection_pool=self.pool)

    async def close(self):
        """Clean up connection pool."""
        await self._client.close()
        await self.pool.disconnect()
```

### Graceful Degradation

Handle failures gracefully when possible:

```python
class HybridCacheAdapter:
    """Two-tier cache: memory (L1) + Redis (L2)."""

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        # Try L1 (memory) first
        entry = await self.memory_cache.get(key)
        if entry:
            return entry

        # Try L2 (Redis) second
        try:
            entry = await self.redis_cache.get(key)
            if entry:
                # Warm L1 cache
                await self.memory_cache.set(key, entry.value, ttl=entry.ttl)
            return entry
        except RedisError:
            # Redis down - degrade gracefully to memory-only
            logger.warning("redis_unavailable", key=key.key)
            return None  # Cache miss, not error
```

### Configuration Validation

Validate configuration early to fail fast:

```python
class GCSFileStorageAdapter:
    def __init__(
        self,
        bucket_name: str,
        project_id: str,
        credentials_path: Optional[str] = None
    ):
        # Validate required config
        if not bucket_name:
            raise ValueError("bucket_name is required")
        if not project_id:
            raise ValueError("project_id is required")

        # Validate credentials path exists
        if credentials_path and not os.path.exists(credentials_path):
            raise ValueError(f"Credentials file not found: {credentials_path}")

        self.bucket_name = bucket_name
        self.project_id = project_id
        self.credentials_path = credentials_path
```

### Domain Exception Translation

Always translate technology exceptions to domain exceptions:

```python
from portico.exceptions import CacheError, CacheConnectionError

class RedisCacheAdapter:
    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        try:
            value = await self.client.get(key.key)
            # ... process value
        except redis.ConnectionError as e:
            logger.error("redis_connection_error", error=str(e))
            raise CacheConnectionError(
                "Failed to connect to Redis"
            ) from e
        except redis.RedisError as e:
            logger.error("redis_error", error=str(e))
            raise CacheError(
                f"Redis operation failed: {str(e)}"
            ) from e
```

This allows kits to handle errors without knowing about Redis.

### Async Context Managers

Provide context managers for resource cleanup:

```python
class DatabaseAdapter:
    async def __aenter__(self):
        """Initialize resources."""
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        await self._disconnect()

# Usage
async with DatabaseAdapter(url="...") as db:
    result = await db.query("SELECT ...")
    # Resources cleaned up automatically
```

### Logging and Observability

Include structured logging for debugging and monitoring:

```python
import logging
import structlog

logger = structlog.get_logger(__name__)

class OpenAIProvider:
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        logger.info(
            "llm_request",
            provider="openai",
            model=request.model or self.default_model,
            message_count=len(request.messages)
        )

        try:
            response = await self.client.chat.completions.create(...)

            logger.info(
                "llm_response",
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )

            return self._to_domain(response)

        except Exception as e:
            logger.error(
                "llm_error",
                provider="openai",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
```

## Summary

Adapters are the implementation layer of Portico's hexagonal architecture:

- **Adapters implement ports** - Concrete implementations of abstract interfaces
- **Adapters handle technology** - All external library and API integration
- **Adapters translate models** - Convert between domain and technology formats
- **Adapters manage resources** - Connections, cleanup, pooling
- **Adapters are swappable** - Change implementations without touching kits
- **Adapters are testable** - Unit test with mocks, integration test with real services
- **Adapters are isolated** - Only imported in compose.py, never in kits

When building with Portico:

1. **Implement port interfaces** - Inherit from port ABC/Protocol
2. **Accept configuration** - Via constructor parameters
3. **Translate models** - Domain ↔ Technology conversions
4. **Handle errors gracefully** - Translate to domain exceptions
5. **Manage resources** - Lazy init, pooling, cleanup
6. **Add observability** - Structured logging and metrics
7. **Test at multiple levels** - Unit, integration, and fake adapters

Understanding adapters completes your knowledge of Portico's architecture. They're the bridge between clean domain logic and the messy reality of external systems - and they keep that mess from leaking into your business logic.
