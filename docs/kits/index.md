# Kits

In Portico's hexagonal architecture, **kits** are the business logic layer - the heart of your application where domain rules, workflows, and use cases live. While [ports](../ports/index.md) define *what* operations are possible and adapters provide *how* they're implemented, kits orchestrate these capabilities to deliver actual business value.

## What is a Kit?

A kit is a cohesive package of business logic that solves a specific domain problem. Think of it as a service layer that knows about your business rules but remains agnostic about infrastructure details.

In traditional layered applications, business logic often gets scattered across controllers, models, and utility functions:

```python
# Traditional approach - business logic scattered everywhere

# In routes.py
@app.post("/users")
async def create_user(username: str, email: str, password: str):
    # Validation here
    if len(password) < 8:
        raise ValueError("Password too short")

    # Hashing here
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # Database here
    user = User(username=username, email=email, password_hash=password_hash)
    session.add(user)
    await session.commit()

    # Audit logging here
    audit_log(f"User {username} created")

    return user
```

With kits, business logic is centralized and testable:

```python
# Kit approach - business logic in service layer

class UserService:
    def __init__(self, user_repository: UserRepository, events: EventBus, config: UserKitConfig):
        self.repository = user_repository
        self.events = events
        self.config = config

    async def create_user(self, user_data: CreateUserRequest) -> User:
        # Validation (using config)
        self._validate_password(user_data.password)

        # Business logic
        password_hash = self._hash_password(user_data.password)
        user = await self.repository.create(user_data, password_hash)

        # Event publishing (other kits can react)
        await self.events.publish(UserCreatedEvent(user_id=user.id, email=user.email))

        return user

# In routes.py - thin controller
@app.post("/users")
async def create_user(request: CreateUserRequest):
    user = await app.kits.user.service.create_user(request)
    return user
```

The kit approach means:

- **Business logic is testable** without running a web server
- **Validation rules** are centralized in one place
- **Domain events** decouple systems (audit kit can listen for UserCreatedEvent)
- **Routes are thin** - they just translate HTTP to domain operations

## Kit Anatomy

Every Portico kit typically contains several components working together:

### 1. Service Layer

The service is where your business logic lives. Services orchestrate operations, enforce business rules, and coordinate between repositories and external systems:

```python
class UserService:
    """Service containing user management business logic."""

    def __init__(
        self,
        user_repository: UserRepository,
        events: EventBus,
        config: UserKitConfig
    ):
        self.repository = user_repository
        self.events = events
        self.config = config

    async def create_user(self, user_data: CreateUserRequest) -> User:
        """Create a new user with validation and event publishing."""
        # Business rule: validate password strength
        if len(user_data.password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")

        # Business rule: ensure email uniqueness
        existing = await self.repository.get_by_email(user_data.email)
        if existing:
            raise ValueError("Email already registered")

        # Execute operation
        password_hash = self._hash_password(user_data.password)
        user = await self.repository.create(user_data, password_hash)

        # Publish domain event
        await self.events.publish(UserCreatedEvent(user_id=user.id, email=user.email))

        return user

    def _hash_password(self, password: str) -> str:
        """Private helper for password hashing."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
```

Services depend on **port interfaces**, never concrete implementations. This means the service can work with any repository that implements the `UserRepository` interface.

### 2. Repository Layer

Repositories handle data persistence. They implement the repository pattern, abstracting database operations behind a clean interface:

```python
class UserRepository:
    """Repository for user persistence operations."""

    def __init__(self, database: Database):
        self.database = database
        self.session_factory = database.session_factory

    async def create(self, user_data: CreateUserRequest, password_hash: str) -> User:
        """Persist a new user to the database."""
        async with self.database.transaction() as session:
            user_model = UserModel(
                id=uuid4(),
                username=user_data.username,
                email=user_data.email,
                password_hash=password_hash,
                created_at=datetime.now(UTC)
            )
            session.add(user_model)
            await session.flush()
            await session.refresh(user_model)

            # Convert ORM model to domain model
            return user_model.to_domain()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Retrieve user by email address."""
        async with self.database.session() as session:
            result = await session.execute(
                select(UserModel).where(UserModel.email == email)
            )
            user_model = result.scalar_one_or_none()
            return user_model.to_domain() if user_model else None
```

Repositories:

- **Hide database implementation details** from services
- **Use ORM models internally** but return domain models (Pydantic)
- **Handle transactions** and session management
- **Convert between persistence and domain layers**

### 3. Configuration

Each kit has a configuration class that defines its settings:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UserKitConfig:
    """Configuration for User Kit."""
    password_min_length: int = 8
    password_max_length: int = 128
    username_pattern: str = r"^[a-zA-Z0-9_]{3,50}$"
    allow_duplicate_emails: bool = False
    default_status: str = "active"

    # Optional integrations
    audit_user_actions: bool = True
    cache_user_lookups: bool = True
    cache_ttl_seconds: int = 300
```

Configuration:

- **Uses dataclasses** for simplicity and validation
- **Provides sensible defaults** so kits work out of the box
- **Typed and documented** for IDE autocomplete
- **Validated at kit initialization** time

### 4. Domain Events

Kits publish domain events when significant business activities occur:

```python
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID
from portico.events import Event

@dataclass
class UserCreatedEvent(Event):
    """Published when a new user is created."""
    user_id: UUID
    email: str
    username: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass
class UserLoggedInEvent(Event):
    """Published when a user successfully authenticates."""
    user_id: UUID
    session_id: UUID
    ip_address: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
```

Events enable **loose coupling** between kits:

```python
# User kit publishes event
await self.events.publish(UserCreatedEvent(user_id=user.id, email=user.email))

# Audit kit subscribes to event
@subscribe_to(UserCreatedEvent)
async def log_user_creation(event: UserCreatedEvent):
    await audit_kit.service.log_event(
        action=AuditAction.CREATE,
        resource_type="user",
        resource_id=event.user_id,
        user_id=event.user_id
    )
```

### 5. Kit Class

The kit class is the container that packages everything together:

```python
from typing import List
from sqlalchemy import Table

class UserKit:
    """User management kit providing user CRUD and authentication."""

    # SQLAlchemy models for database tables
    models: List[type] = [UserModel]

    def __init__(
        self,
        database: Database,
        events: EventBus,
        config: Optional[UserKitConfig] = None
    ):
        self.database = database
        self.events = events
        self.config = config or UserKitConfig()

        # Initialize repository
        self.repository = UserRepository(database=database)

        # Initialize service
        self.service = UserService(
            user_repository=self.repository,
            events=events,
            config=self.config
        )

    @classmethod
    def create(
        cls,
        database: Database,
        events: EventBus,
        config: Optional[dict] = None,
        **kwargs
    ) -> "UserKit":
        """Factory method for creating kit with config validation."""
        validated_config = UserKitConfig(**(config or {}))
        return cls(database, events, validated_config, **kwargs)
```

The kit class provides:

- **Single entry point** to all kit functionality
- **Factory method** for validated construction
- **Model registration** so compose.webapp() can create tables
- **Service and repository access** for application code

## Kit Patterns

Portico kits follow consistent patterns that make them predictable and composable.

### Stateful vs Stateless Kits

#### Stateful Kits

Stateful kits manage persistent domain entities and include database models:

```python
class UserKit:
    """Stateful kit - manages User entities in database."""

    models = [UserModel]  # Defines database tables

    def __init__(self, database, events, config):
        self.database = database
        self.repository = UserRepository(database)  # Has repository
        self.service = UserService(self.repository, events, config)
```

Stateful kits typically:

- **Define ORM models** (SQLAlchemy) for database persistence
- **Implement repositories** for data access
- **Manage entity lifecycle** (create, read, update, delete)
- **Publish domain events** when entities change

Examples: User Kit, Group Kit, RBAC Kit, Audit Kit

#### Stateless Kits

Stateless kits orchestrate operations without managing persistent entities:

```python
class CacheKit:
    """Stateless kit - provides caching service."""

    # No models attribute (no database tables)

    def __init__(self, database, events, config, cache_adapter: CacheAdapter):
        self.config = config
        self.service = CacheService(cache_adapter, config)  # Wraps adapter
```

Stateless kits typically:

- **No ORM models** (don't create database tables)
- **Wrap external services** behind clean interfaces
- **Provide infrastructure capabilities** (caching, LLM calls, file storage)
- **Depend on adapters** injected at initialization

Examples: Cache Kit, LLM Kit, RAG Kit, Settings Kit

### Dependency Injection

Kits receive all dependencies through constructor parameters, never by importing them directly:

```python
# ✅ CORRECT - Dependencies injected via constructor
class AuthKit:
    def __init__(
        self,
        database: Database,
        events: EventBus,
        config: AuthKitConfig,
        user_kit: UserKit  # Injected by compose.webapp()
    ):
        self.user_kit = user_kit
        self.service = AuthenticationService(
            user_repository=user_kit.repository,  # Uses injected dependency
            events=events,
            config=config
        )
```

```python
# ❌ WRONG - Importing other kits directly
from portico.kits.user import UserKit  # Circular import risk!

class AuthKit:
    def __init__(self, database, events, config):
        self.user_kit = UserKit(database, events)  # Creates own instance!
```

The composition root (`compose.webapp()`) handles dependency resolution:

```python
app = compose.webapp(
    kits=[
        compose.user(),      # Creates UserKit first
        compose.auth(),      # Creates AuthKit with UserKit injected
    ]
)
```

### Repository Pattern

The repository pattern is fundamental to stateful kits. It separates data access from business logic:

```python
# Port interface (what operations are available)
class UserRepository(ABC):
    @abstractmethod
    async def create(self, user_data: CreateUserRequest) -> User:
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        pass

# Kit repository implementation
class UserRepository:
    """Concrete implementation using SQLAlchemy."""

    def __init__(self, database: Database):
        self.database = database

    async def create(self, user_data: CreateUserRequest, password_hash: str) -> User:
        # Implementation uses ORM models
        async with self.database.transaction() as session:
            user_model = UserModel(...)
            session.add(user_model)
            await session.flush()
            return user_model.to_domain()  # Return domain model
```

This pattern provides:

- **Abstraction** - Business logic doesn't know about SQLAlchemy
- **Testability** - Easy to create fake repositories for testing
- **Consistency** - All data access goes through repositories
- **Separation** - ORM models stay in repository, domain models in service

### Event-Driven Communication

Kits communicate through domain events instead of direct calls:

```python
# User Kit publishes event
class UserService:
    async def delete_user(self, user_id: UUID) -> None:
        await self.repository.delete(user_id)

        # Publish event - User Kit doesn't know who cares
        await self.events.publish(UserDeletedEvent(user_id=user_id))

# Other kits subscribe to events
class GroupService:
    async def handle_user_deleted(self, event: UserDeletedEvent):
        """Automatically remove user from all groups."""
        await self.repository.remove_user_from_all_groups(event.user_id)

class SessionService:
    async def handle_user_deleted(self, event: UserDeletedEvent):
        """Automatically delete user's sessions."""
        await self.repository.delete_user_sessions(event.user_id)
```

This decouples kits - User Kit doesn't need to know about Groups or Sessions, they react independently to the same event.

### Configuration Validation

Kits validate configuration at initialization time, failing fast if misconfigured:

```python
@dataclass
class AuthKitConfig:
    session_secret: str
    session_cookie_name: str = "session_token"
    session_ttl_seconds: int = 3600

    def __post_init__(self):
        # Validate session secret length
        if len(self.session_secret) < 32:
            raise ValueError("session_secret must be at least 32 characters")

        # Validate TTL
        if self.session_ttl_seconds < 60:
            raise ValueError("session_ttl_seconds must be at least 60")

# This fails immediately at startup
app = compose.webapp(
    kits=[
        compose.auth(session_secret="too-short")  # Raises ValueError
    ]
)
```

This prevents invalid configurations from reaching production.

## How Kits Relate to Ports and Adapters

Kits sit at the center of hexagonal architecture, using ports while remaining independent of adapters:

```
┌─────────────────────────────────────────┐
│  Application (Routes, Controllers)      │
│  - Thin layer                           │
│  - Translates HTTP to kit calls         │
└─────────────────┬───────────────────────┘
                  │
                  │ Uses
                  ↓
┌─────────────────────────────────────────┐
│  Kits (Business Logic)                  │
│  - Services with business rules         │
│  - Repositories for data access         │
│  - Depend on PORTS (interfaces)         │
│  - Publish domain events                │
└─────────────────┬───────────────────────┘
                  │
                  │ Depends on
                  ↓
┌─────────────────────────────────────────┐
│  Ports (Interfaces)                     │
│  - Define contracts                     │
│  - No implementation                    │
└─────────────────┬───────────────────────┘
                  ↑
                  │ Implemented by
                  │
┌─────────────────────────────────────────┐
│  Adapters (Implementations)             │
│  - PostgreSQL, Redis, OpenAI, etc.      │
│  - Injected into kits by compose        │
└─────────────────────────────────────────┘
```

### Kits Depend on Ports

```python
# Kit imports port interface
from portico.ports.cache import CacheAdapter, CacheKey, CacheEntry

class UserService:
    def __init__(
        self,
        user_repository: UserRepository,
        cache: CacheAdapter,  # Port interface, not Redis/Memcached
        events: EventBus
    ):
        self.repository = user_repository
        self.cache = cache

    async def get_user(self, user_id: UUID) -> Optional[User]:
        # Service uses port interface
        cache_key = CacheKey(key=f"user:{user_id}")
        cached = await self.cache.get(cache_key)

        if cached:
            return User(**cached.value)

        user = await self.repository.get_by_id(user_id)
        if user:
            await self.cache.set(cache_key, user.dict(), ttl=300)

        return user
```

The service has **no idea** if it's using Redis, Memcached, or in-memory caching. It only knows the `CacheAdapter` interface.

### Adapters are Injected by Compose

```python
# compose.py - The only place that imports adapters
def cache(**config):
    from portico.adapters.cache import RedisCacheAdapter  # Adapter import
    from portico.kits.cache import CacheKit

    def factory(database, events):
        # Choose adapter based on config
        if config.get("backend") == "redis":
            adapter = RedisCacheAdapter(redis_url=config["redis_url"])
        else:
            adapter = MemoryCacheAdapter()

        # Inject adapter into kit
        return CacheKit.create(database, events, config, cache_adapter=adapter)

    return factory
```

### Kits Stay Adapter-Agnostic

```python
# ✅ CORRECT - Kit depends on port
from portico.ports.llm import ChatCompletionProvider

class LLMService:
    def __init__(self, provider: ChatCompletionProvider):
        self.provider = provider  # Could be OpenAI, Anthropic, etc.

# ❌ WRONG - Kit imports adapter
from portico.adapters.llm import OpenAIProvider

class LLMService:
    def __init__(self):
        self.provider = OpenAIProvider()  # Coupled to OpenAI!
```

This architectural rule is enforced by `import-linter` - the build fails if a kit imports an adapter.

## Testing Kits

One of the biggest benefits of the kit architecture is testability. Because kits depend on interfaces, you can test them without external dependencies.

### Testing with Fake Implementations

```python
from portico.ports.user import UserRepository, User, CreateUserRequest
from uuid import uuid4

class FakeUserRepository(UserRepository):
    """In-memory fake for testing."""

    def __init__(self):
        self.users = {}

    async def create(self, user_data: CreateUserRequest, password_hash: str) -> User:
        user = User(
            id=uuid4(),
            username=user_data.username,
            email=user_data.email,
            created_at=datetime.now(UTC)
        )
        self.users[user.id] = user
        return user

    async def get_by_email(self, email: str) -> Optional[User]:
        return next((u for u in self.users.values() if u.email == email), None)

# Test without real database
@pytest.mark.asyncio
async def test_create_user():
    # Arrange
    fake_repo = FakeUserRepository()
    fake_events = FakeEventBus()
    config = UserKitConfig(password_min_length=8)

    service = UserService(
        user_repository=fake_repo,
        events=fake_events,
        config=config
    )

    # Act
    user = await service.create_user(
        CreateUserRequest(username="alice", email="alice@example.com", password="secret123")
    )

    # Assert
    assert user.username == "alice"
    assert user.email == "alice@example.com"

    # Verify event published
    assert len(fake_events.published) == 1
    assert isinstance(fake_events.published[0], UserCreatedEvent)
```

### Testing Business Rules

You can test business logic in isolation:

```python
@pytest.mark.asyncio
async def test_password_too_short():
    service = UserService(FakeUserRepository(), FakeEventBus(), UserKitConfig(password_min_length=10))

    with pytest.raises(ValueError, match="Password must be at least 10 characters"):
        await service.create_user(
            CreateUserRequest(username="alice", email="alice@example.com", password="short")
        )

@pytest.mark.asyncio
async def test_duplicate_email():
    fake_repo = FakeUserRepository()
    service = UserService(fake_repo, FakeEventBus(), UserKitConfig())

    # Create first user
    await service.create_user(
        CreateUserRequest(username="alice", email="alice@example.com", password="secret123")
    )

    # Try to create second user with same email
    with pytest.raises(ValueError, match="Email already registered"):
        await service.create_user(
            CreateUserRequest(username="alice2", email="alice@example.com", password="secret456")
        )
```

### Integration Testing

For integration tests, use real implementations but with test databases:

```python
@pytest.mark.integration
async def test_user_creation_with_real_database():
    # Use test database
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.create_tables([UserModel])

    events = EventBus()
    config = UserKitConfig()

    # Real repository with test database
    repository = UserRepository(database=database)
    service = UserService(repository, events, config)

    # Test with real database
    user = await service.create_user(
        CreateUserRequest(username="alice", email="alice@example.com", password="secret123")
    )

    # Verify persistence
    retrieved = await repository.get_by_email("alice@example.com")
    assert retrieved.id == user.id
```

## Kit Composition

Kits are composed through the `compose.webapp()` function, which handles all the complexity of dependency resolution and initialization.

### Basic Composition

```python
from portico import compose

app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(password_min_length=8),
        compose.auth(
            session_secret="your-32-char-secret-here-at-least",
            session_ttl_seconds=3600
        ),
        compose.group(),
        compose.rbac(),
    ]
)

# Access kits in your application
user_kit = app.kits["user"]
auth_kit = app.kits["auth"]
```

### Kit Dependencies

The compose system automatically resolves dependencies between kits:

```python
# AuthKit depends on UserKit
class AuthKit:
    def __init__(self, database, events, config, user_kit: UserKit):
        # user_kit is automatically injected by compose.webapp()
        self.user_kit = user_kit

# You just declare what you need
app = compose.webapp(
    kits=[
        compose.user(),   # Created first
        compose.auth(),   # Receives user_kit automatically
    ]
)
```

The compose system:
1. **Inspects signatures** to find dependencies
2. **Resolves by name** - parameter `user_kit` matches kit name `"user"`
3. **Resolves by type** - if a parameter type matches a kit class
4. **Fails fast** if dependencies can't be satisfied

### Conditional Features

You can compose kits conditionally based on configuration:

```python
kits = [
    compose.user(),
    compose.auth(),
]

# Add optional features
if config.enable_rbac:
    kits.extend([compose.group(), compose.rbac()])

if config.enable_audit:
    kits.append(compose.audit())

if config.enable_caching:
    kits.append(compose.cache(backend="redis", redis_url=config.redis_url))

app = compose.webapp(database_url=config.database_url, kits=kits)
```

## Common Patterns and Best Practices

### Service Methods Should Be Focused

Each service method should do one thing well:

```python
# ✅ GOOD - Focused methods
class UserService:
    async def create_user(self, user_data: CreateUserRequest) -> User:
        """Create a new user."""
        ...

    async def update_user(self, user_id: UUID, updates: UpdateUserRequest) -> User:
        """Update an existing user."""
        ...

    async def delete_user(self, user_id: UUID) -> None:
        """Delete a user."""
        ...

# ❌ BAD - One method does too much
class UserService:
    async def manage_user(self, action: str, user_id: UUID, data: dict):
        """Do everything."""
        if action == "create":
            ...
        elif action == "update":
            ...
        elif action == "delete":
            ...
```

### Validate Early, Fail Fast

Validate inputs at the service boundary:

```python
class UserService:
    async def create_user(self, user_data: CreateUserRequest) -> User:
        # Validate before doing anything
        if len(user_data.password) < self.config.password_min_length:
            raise ValueError(f"Password too short")

        if not re.match(self.config.username_pattern, user_data.username):
            raise ValueError("Invalid username format")

        # Now proceed with business logic
        ...
```

### Use Domain Models, Not Dictionaries

Always use Pydantic models for type safety:

```python
# ✅ GOOD - Type-safe domain models
async def create_user(self, user_data: CreateUserRequest) -> User:
    user = await self.repository.create(user_data)
    return user  # Returns User (Pydantic model)

# ❌ BAD - Passing dictionaries
async def create_user(self, user_data: dict) -> dict:
    user = await self.repository.create(user_data)
    return {"id": str(user.id), "username": user.username}
```

### Publish Events for Side Effects

Use events instead of direct calls for side effects:

```python
# ✅ GOOD - Event-driven side effects
class UserService:
    async def delete_user(self, user_id: UUID) -> None:
        await self.repository.delete(user_id)

        # Publish event - other kits react independently
        await self.events.publish(UserDeletedEvent(user_id=user_id))

# ❌ BAD - Direct coupling to other kits
class UserService:
    def __init__(self, user_repo, group_kit, session_kit, audit_kit):
        # Too many dependencies!
        ...

    async def delete_user(self, user_id: UUID) -> None:
        await self.repository.delete(user_id)
        await self.group_kit.remove_from_all_groups(user_id)
        await self.session_kit.delete_sessions(user_id)
        await self.audit_kit.log_deletion(user_id)
```

### Keep Configuration Immutable

Treat configuration as immutable after initialization:

```python
@dataclass(frozen=True)  # Immutable
class UserKitConfig:
    password_min_length: int = 8
    allow_duplicate_emails: bool = False
```

### Repository Methods Should Be Atomic

Each repository method should be a complete, atomic operation:

```python
# ✅ GOOD - Atomic operation
class UserRepository:
    async def create(self, user_data: CreateUserRequest, password_hash: str) -> User:
        async with self.database.transaction() as session:
            # Everything happens in one transaction
            user_model = UserModel(...)
            session.add(user_model)
            await session.flush()
            return user_model.to_domain()

# ❌ BAD - Leaky abstraction
class UserRepository:
    async def begin_transaction(self):
        ...

    async def add_user(self, user_model):
        ...

    async def commit_transaction(self):
        ...
```

## Summary

Kits are the business logic layer of Portico:

- **Kits contain services** - Business logic and workflow orchestration
- **Kits use repositories** - Abstract data access behind clean interfaces
- **Kits depend on ports** - Never import adapters directly
- **Kits publish events** - Enable loose coupling between domains
- **Kits are testable** - Inject fake implementations for fast tests
- **Kits are composable** - Compose system handles dependency injection

When building with Portico:

1. **Put business logic in services** - Not in routes or models
2. **Use repositories for data access** - Keep ORM details isolated
3. **Depend on port interfaces** - Never import adapters
4. **Publish domain events** - For side effects and cross-kit communication
5. **Validate configuration early** - Fail fast at initialization
6. **Test with fakes** - Fast, isolated unit tests
7. **Test with real databases** - Integration tests for data access

Understanding kits is essential to building maintainable applications with Portico. They're where your domain expertise lives, isolated from infrastructure concerns and easy to test.
