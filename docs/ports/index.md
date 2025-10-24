# Ports

In Portico's hexagonal architecture, **ports** are the stable, framework-free interfaces that define the contract between your business logic and the outside world. They are pure domain abstractions - no implementation details, no external dependencies, just clear interfaces that express what your application needs.

## What is a Port?

A port is an abstract interface that defines a capability your application requires without specifying how that capability is implemented. Think of ports as the "shape" of a dependency - they describe what operations are available and what data flows in and out, but not where that data comes from or how the operations work internally.

In traditional applications, you might write code like this:

```python
# Direct dependency on implementation
from redis import Redis

class UserService:
    def __init__(self):
        self.cache = Redis(host='localhost')  # Coupled to Redis!

    async def get_user(self, user_id: str):
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)
        # ... fetch from database
```

With ports, you instead depend on an abstract interface:

```python
# Dependency on interface (port)
from portico.ports.cache import CacheAdapter

class UserService:
    def __init__(self, cache: CacheAdapter):  # Depends on port!
        self.cache = cache

    async def get_user(self, user_id: str):
        cache_key = CacheKey(key=f"user:{user_id}")
        cached = await self.cache.get(cache_key)
        if cached:
            return cached.value
        # ... fetch from database
```

The second example depends on `CacheAdapter` (a port), not `Redis` (an implementation). This means:

- **You can swap Redis for Memcached** without changing `UserService`
- **You can test with an in-memory cache** without running Redis
- **Your business logic** stays clean and focused on domain concerns

## Port Anatomy

Every Portico port typically contains three types of components:

### 1. Domain Models

These are the core entities and value objects that represent concepts in your domain. They're built with Pydantic for validation and serialization:

```python
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime

class User(BaseModel):
    """Domain model representing a user in the system."""
    id: UUID = Field(default_factory=uuid4)
    username: str
    email: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Domain methods that express business logic
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE
```

Domain models are **frozen in time** - they represent a snapshot of data at a point in time. They can have domain methods that express business logic, but they don't perform I/O operations.

### 2. Request/Response Models

These define the shape of data flowing into and out of operations:

```python
class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(min_length=8)

    # Pydantic validators ensure data quality
    @field_validator('username')
    def username_must_be_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
```

These models use Pydantic's validation to ensure data integrity at the boundary of your system.

### 3. Abstract Interfaces

These define the operations available for a particular capability:

```python
from abc import ABC, abstractmethod
from typing import Optional

class UserRepository(ABC):
    """Port defining user persistence operations."""

    @abstractmethod
    async def create(self, request: CreateUserRequest) -> User:
        """Create a new user.

        Args:
            request: User creation data

        Returns:
            Created user with generated ID and timestamps
        """
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve user by ID.

        Args:
            user_id: Unique user identifier

        Returns:
            User if found, None otherwise
        """
        pass
```

Interfaces use Python's `ABC` (Abstract Base Class) to enforce that implementations provide all required methods.

## Port Types

Portico uses semantic naming conventions to indicate what kind of operations a port provides:

### Repository

**Pattern**: `{Entity}Repository`
**Purpose**: Persistence and retrieval of domain entities
**Examples**: `UserRepository`, `GroupRepository`, `PermissionRepository`

Repositories abstract database operations. They provide CRUD (Create, Read, Update, Delete) operations and domain-specific queries:

```python
class UserRepository(ABC):
    @abstractmethod
    async def create(self, request: CreateUserRequest) -> User:
        """Persist a new user."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Find user by email address."""
        pass

    @abstractmethod
    async def update(self, user_id: UUID, request: UpdateUserRequest) -> User:
        """Update existing user."""
        pass
```

### Provider

**Pattern**: `{Capability}Provider`
**Purpose**: External service integration for computational operations
**Examples**: `ChatCompletionProvider`, `EmbeddingProvider`

Providers integrate with external services, typically for AI/ML capabilities:

```python
class ChatCompletionProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate chat completion using an LLM service."""
        pass
```

### Adapter

**Pattern**: `{Capability}Adapter`
**Purpose**: Technology-specific integration for infrastructure
**Examples**: `CacheAdapter`, `FileStorageAdapter`, `AuditAdapter`, `NotificationAdapter`

Adapters integrate with infrastructure services like caching, storage, logging, and notifications:

```python
class CacheAdapter(ABC):
    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve cached value."""
        pass

    @abstractmethod
    async def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL."""
        pass
```

!!! note
    Don't confuse "adapter" in the port name (like `CacheAdapter`) with the architectural concept of adapters. `CacheAdapter` is a **port** (interface). The **adapter** (implementation) would be something like `RedisCacheAdapter` or `MemoryCacheAdapter`.

### Registry

**Pattern**: `{Entity}Registry`
**Purpose**: Registration and lookup of configured entities
**Examples**: `TemplateRegistry`, `SettingsRegistry`

Registries manage collections of configured items, typically loaded at startup:

```python
class TemplateRegistry(ABC):
    @abstractmethod
    async def register(self, template: Template) -> None:
        """Register a template for later use."""
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Template]:
        """Retrieve registered template by name."""
        pass
```

### Storage

**Pattern**: `{Capability}Storage`
**Purpose**: Low-level data persistence (typically non-relational)
**Examples**: `VectorStore`

Storage ports define persistence operations for specific data structures, typically for specialized data types like vectors.

### Processor

**Pattern**: `{Operation}Processor`
**Purpose**: Data transformation and analysis
**Examples**: `DocumentProcessor`

Processors transform data from one format to another or extract structured information:

```python
class DocumentProcessor(ABC):
    @abstractmethod
    async def process_document(
        self,
        content: DocumentContent
    ) -> ProcessedDocument:
        """Process raw document into structured chunks."""
        pass
```

## Port Categories by Domain

Portico's ports can be grouped by the business domain they serve:

### User & Access Management

Ports for managing users, groups, permissions, and authentication:


- **User Port** (`user.py`) - User CRUD, authentication, role management
- **Group Port** (`group.py`) - Organizational hierarchies and group membership
- **Permissions Port** (`permissions.py`) - Role-based access control (RBAC)

**Common Pattern**: These ports often work together. For example, authentication uses the User port to verify credentials and manage user sessions through the auth kit.

### Infrastructure Services

Ports for external infrastructure like caching, storage, and notifications:


- **Cache Port** (`cache.py`) - High-performance caching with TTL and tag-based invalidation
- **File Storage Port** (`file_storage.py`) - File upload, download, and metadata management
- **Notification Port** (`notification.py`) - Email and SMS delivery with template support
- **Audit Port** (`audit.py`) - Activity logging and compliance tracking

**Common Pattern**: These are typically used as cross-cutting concerns. Kits inject these adapters to add caching, notifications, or audit logging to business operations.

### AI & Machine Learning

Ports for integrating with LLM and vector/embedding services:


- **LLM Port** (`llm.py`) - Chat completions, conversations, and prompt management
- **Embedding Port** (`embedding.py`) - Text vectorization for semantic search
- **Vector Store Port** (`vector_store.py`) - Vector similarity search and document retrieval
- **Document Processor Port** (`document_processor.py`) - Document chunking and analysis
- **Managed RAG Port** (`managed_rag.py`) - Integrated RAG platforms (Graphlit, etc.)

**Common Pattern**: RAG (Retrieval-Augmented Generation) workflows chain these together: documents are processed into chunks, chunks are embedded, embeddings are stored in a vector store, and retrieved chunks augment LLM prompts.

### Background Processing

Ports for asynchronous job processing and scheduling:


- **Job Port** (`job.py`) - Job lifecycle and status tracking (domain models)
- **Job Queue Port** (`job_queue.py`) - Queue operations (enqueue, dequeue, acknowledge)
- **Job Handler Port** (`job_handler.py`) - Business logic for processing jobs
- **Job Trigger Port** (`job_trigger.py`) - Event sources that create jobs (cron, webhooks, etc.)
- **Job Creator Port** (`job_creator.py`) - Interface for creating jobs (used by triggers)

**Common Pattern**: Triggers detect events and use the Job Creator port to enqueue jobs. Jobs are pulled from the queue and dispatched to the appropriate handler based on `job_type`.

### Configuration & Templates

Ports for managing application configuration and templating:


- **Template Port** (`template.py`) - Jinja2 template registry and rendering
- **Settings Port** (`settings.py`) - Application configuration from multiple sources
- **Config Schema Port** (`config.py`) - Declarative configuration schema building

**Common Pattern**: Templates are used throughout Portico - for LLM prompts, notification emails, and HTML rendering. Settings provide runtime configuration.

### Organization & Structure

Ports for representing organizational hierarchies and permissions:

- **Organization Port** (`organization.py`) - Hierarchical organization models and permission matrices

**Common Pattern**: This port provides read models (data structures) for visualizing org charts and permission reports, typically used by organization management kits.

## Common Design Patterns

### Async-First

Nearly all port operations are async to support high-concurrency web applications:

```python
class UserRepository(ABC):
    @abstractmethod
    async def create(self, request: CreateUserRequest) -> User:
        """Async allows non-blocking I/O."""
        pass
```

Even if your initial adapter implementation is synchronous (like SQLite), defining the port as async allows you to swap in async implementations (like PostgreSQL with asyncpg) without changing business logic.

### Pydantic Models for Validation

All data crossing port boundaries uses Pydantic models:

```python
class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

    @field_validator('email')
    def normalize_email(cls, v):
        return v.lower().strip()
```

This ensures:

- **Data integrity**: Invalid data is rejected at the boundary
- **Documentation**: Field types and constraints are self-documenting
- **Serialization**: Easy conversion to/from JSON, dict, etc.

### Optional User Ownership

Many domain models support optional user ownership for multi-tenant scenarios:

```python
class Template(BaseModel):
    id: UUID
    name: str
    content: str
    user_id: Optional[UUID] = None  # If None, template is global
    is_public: bool = False  # If True, visible to all users
```

This pattern allows:

- **Global resources**: `user_id = None` for system-wide templates
- **User-owned resources**: `user_id = <uuid>` for user-specific items
- **Sharing**: `is_public = True` to share user-owned resources

### Metadata Fields for Extensibility

Most domain models include a `metadata: Dict[str, Any]` field:

```python
class FileMetadata(BaseModel):
    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extensible!
```

This allows you to attach custom data without modifying the port schema.

### Pagination Support

List operations typically support limit/offset pagination:

```python
class UserRepository(ABC):
    @abstractmethod
    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[User]:
        """Retrieve paginated user list."""
        pass
```

### Namespace Isolation

Infrastructure ports often support namespaces for multi-tenancy:

```python
class VectorStoreConfig(BaseModel):
    namespace: Optional[str] = None  # Isolate by tenant

# Each tenant's data is isolated
tenant_a_store = VectorStore(config=VectorStoreConfig(namespace="tenant_a"))
tenant_b_store = VectorStore(config=VectorStoreConfig(namespace="tenant_b"))
```

## How Ports Relate to Adapters and Kits

Ports sit at the center of Portico's hexagonal architecture:

```
┌─────────────────────────────────────────┐
│  Kits (Business Logic)                  │
│  - Depend on PORTS (interfaces)         │
│  - Never import adapters directly       │
└─────────────────┬───────────────────────┘
                  │
                  │ Uses
                  ↓
┌─────────────────────────────────────────┐
│  Ports (Interfaces)                     │
│  - Define what operations exist         │
│  - Domain models, enums, requests       │
│  - No implementation, no external deps  │
└─────────────────┬───────────────────────┘
                  ↑
                  │ Implements
                  │
┌─────────────────────────────────────────┐
│  Adapters (Implementations)             │
│  - Implement port interfaces            │
│  - Integrate with external services     │
│  - Can import SDKs, databases, etc.     │
└─────────────────────────────────────────┘
```

### Ports Define the Contract

```python
# portico/ports/cache.py
from abc import ABC, abstractmethod

class CacheAdapter(ABC):
    """Port: defines WHAT caching operations are available."""

    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        pass

    @abstractmethod
    async def set(self, key: CacheKey, value: Any, ttl: Optional[int]) -> None:
        pass
```

### Adapters Implement the Contract

```python
# portico/adapters/cache/redis_adapter.py
from redis.asyncio import Redis
from portico.ports.cache import CacheAdapter

class RedisCacheAdapter(CacheAdapter):
    """Adapter: implements HOW caching works with Redis."""

    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        value = await self.redis.get(key.key)
        if value:
            return CacheEntry(value=json.loads(value))
        return None

    async def set(self, key: CacheKey, value: Any, ttl: Optional[int]) -> None:
        await self.redis.set(key.key, json.dumps(value), ex=ttl)
```

### Kits Use the Port

```python
# portico/kits/user/service.py
from portico.ports.user import UserRepository
from portico.ports.cache import CacheAdapter

class UserService:
    """Kit: uses ports to implement business logic."""

    def __init__(
        self,
        user_repository: UserRepository,  # Port dependency!
        cache: CacheAdapter  # Port dependency!
    ):
        self.users = user_repository
        self.cache = cache

    async def get_user(self, user_id: UUID) -> Optional[User]:
        # Try cache first
        cache_key = CacheKey(key=f"user:{user_id}")
        cached = await self.cache.get(cache_key)
        if cached:
            return User(**cached.value)

        # Cache miss - fetch from repository
        user = await self.users.get_by_id(user_id)
        if user:
            await self.cache.set(cache_key, user.dict(), ttl=300)

        return user
```

Notice how `UserService` has **no idea** whether it's using Redis or Memcached for caching, or PostgreSQL or SQLite for persistence. It only knows the port interfaces.

### Composition Root Wires It Together

```python
# portico/compose.py
def cache(**config):
    """Factory function that creates cache kit with chosen adapter."""
    from portico.adapters.cache import RedisCacheAdapter
    from portico.kits.cache import CacheKit

    def factory(database, events):
        # THIS is where the adapter is chosen
        adapter = RedisCacheAdapter(redis_url=config["redis_url"])
        return CacheKit.create(database, events, config, adapter)

    return factory
```

The `compose.py` module is the **only place** in Portico where adapters are imported and instantiated. This enforces clean architecture - kits can never accidentally import adapters.

## How to Use Ports

### When Writing Business Logic (Kits)

Import and depend on ports, never adapters:

```python
# ✅ CORRECT - Import port
from portico.ports.user import UserRepository, CreateUserRequest

class SignupService:
    def __init__(self, users: UserRepository):  # Depend on interface
        self.users = users

    async def signup(self, username: str, email: str, password: str):
        request = CreateUserRequest(
            username=username,
            email=email,
            password=password
        )
        return await self.users.create(request)
```

```python
# ❌ WRONG - Import adapter
from portico.adapters.storage.postgres import PostgresUserRepository  # NO!

class SignupService:
    def __init__(self):
        self.users = PostgresUserRepository()  # Couples to PostgreSQL!
```

### When Implementing a Custom Adapter

Implement the port interface:

```python
from portico.ports.cache import CacheAdapter, CacheKey, CacheEntry
from typing import Optional, Any
import memcache

class MemcachedAdapter(CacheAdapter):
    """Custom adapter implementing the CacheAdapter port."""

    def __init__(self, servers: list[str]):
        self.client = memcache.Client(servers)

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        value = self.client.get(key.key)
        if value:
            return CacheEntry(value=value, key=key.key)
        return None

    async def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        self.client.set(key.key, value, time=ttl or 0)

    # ... implement other required methods
```

### When Testing

Use fake/mock implementations of ports:

```python
from portico.ports.user import UserRepository, User, CreateUserRequest
from uuid import uuid4

class FakeUserRepository(UserRepository):
    """In-memory fake for testing."""

    def __init__(self):
        self.users = {}

    async def create(self, request: CreateUserRequest) -> User:
        user = User(
            id=uuid4(),
            username=request.username,
            email=request.email
        )
        self.users[user.id] = user
        return user

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        return self.users.get(user_id)

# Now test your service without real database
def test_signup():
    fake_users = FakeUserRepository()
    service = SignupService(users=fake_users)

    user = await service.signup("alice", "alice@example.com", "secret123")
    assert user.username == "alice"
```

## Summary

Ports are the foundation of Portico's clean architecture:

- **Ports are interfaces** - They define what operations exist without specifying how they work
- **Ports contain domain models** - Pydantic models represent your business entities
- **Ports are technology-agnostic** - No databases, no SDKs, no implementation details
- **Ports enable testing** - Mock implementations for fast, isolated tests
- **Ports enable flexibility** - Swap implementations without changing business logic

When building with Portico:

1. **Depend on ports** in your business logic (kits)
2. **Implement ports** when creating custom adapters
3. **Never import adapters** directly - let the composition root wire dependencies

Understanding ports is essential to working effectively with Portico's hexagonal architecture. They are the contracts that keep your codebase clean, testable, and maintainable as your application grows.
