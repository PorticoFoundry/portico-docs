# Compose: The Composition Root

## What is the Composition Root?

The **composition root** is the single place in your application where all dependency wiring happens. In Portico, this is exclusively the `compose.py` module.

Think of it as the "assembly line" where your application is built:

- **Adapters** (concrete implementations) are imported and instantiated
- **Kits** (business logic) are created and wired with their dependencies
- **Configuration** is validated and applied
- **Dependencies** are automatically resolved and injected

### The "ONLY Place" Principle

This is the most critical architectural rule in Portico:

**`compose.py` is the ONLY place where adapters are imported.**

Everywhere else in your codebase:

- ❌ Kits **cannot** import adapters
- ❌ Ports **cannot** import adapters
- ❌ Application code **cannot** import adapters
- ❌ Tests **cannot** import adapters (except to create fakes)

Only in `compose.py`:

- ✅ Adapters **are** imported
- ✅ Adapters **are** instantiated
- ✅ Adapters **are** injected into kits

### Why This Architecture Matters

Traditional approaches mix infrastructure concerns throughout the codebase:

```python
# ❌ Traditional approach - scattered dependencies
class UserService:
    def __init__(self):
        self.redis = Redis(url="redis://localhost")  # Hard-coded Redis!
        self.db = SessionLocal()  # Tightly coupled to SQLAlchemy!
        self.openai = OpenAI(api_key="...")  # Can't swap providers!
```

With Portico's composition root:

```python
# ✅ Portico approach - dependencies injected
class UserService:
    def __init__(self, cache: CacheAdapter, database: Database, llm: LLMProvider):
        self.cache = cache      # Interface, not Redis
        self.database = database  # Abstract database
        self.llm = llm          # Provider-agnostic
```

The difference? **All concrete implementations are wired in one place** - `compose.py`.

### Composition Root Architecture

```
┌───────────────────────────────────────────────────────┐
│                  Application Code                      │
│  routes/, services/, models/                          │
│                                                       │
│  NO ADAPTER IMPORTS ❌                                │
└───────────────────────────────────────────────────────┘
                         ↓ uses
┌───────────────────────────────────────────────────────┐
│                       Kits                            │
│  UserKit, AuthKit, CacheKit, etc.                     │
│                                                       │
│  NO ADAPTER IMPORTS ❌                                │
│  Depend on Ports only ✅                              │
└───────────────────────────────────────────────────────┘
                         ↓ depend on
┌───────────────────────────────────────────────────────┐
│                       Ports                           │
│  CacheAdapter, LLMProvider, UserRepository            │
│                                                       │
│  NO ADAPTER OR KIT IMPORTS ❌                         │
│  Pure interfaces ✅                                    │
└───────────────────────────────────────────────────────┘
                         ↑ implemented by
┌───────────────────────────────────────────────────────┐
│                     Adapters                          │
│  RedisCacheAdapter, OpenAIProvider, etc.              │
│                                                       │
│  ONLY IMPORTED BY compose.py ✅                       │
└───────────────────────────────────────────────────────┘
                         ↑ imported/instantiated by
┌───────────────────────────────────────────────────────┐
│                    compose.py                         │
│                THE COMPOSITION ROOT                    │
│                                                       │
│  • Imports adapters                                   │
│  • Instantiates adapters                             │
│  • Creates kits                                       │
│  • Injects dependencies                              │
│  • Wires the entire application                      │
└───────────────────────────────────────────────────────┘
```

This architecture is **enforced** by import-linter rules that make it impossible to violate accidentally. If a kit tries to import an adapter, the build fails.

---

## Why the Composition Root Matters

The composition root pattern provides three fundamental benefits:

### 1. Swappability Without Code Changes

Change implementations by modifying **one line** in your configuration:

```python
# Development - use in-memory cache
app = compose.webapp(
    database_url="sqlite+aiosqlite:///app.db",
    kits=[
        compose.cache(backend="memory"),  # ← Change this line
    ],
)

# Production - use Redis cache
app = compose.webapp(
    database_url="postgresql+asyncpg://...",
    kits=[
        compose.cache(backend="redis", redis_url="..."),  # ← One line changed
    ],
)
```

**No kit code changes required.** No service code changes required. Just configuration.

The same applies to:

- **LLM providers:** Switch from OpenAI to Anthropic
- **File storage:** Switch from local disk to Google Cloud Storage
- **Vector stores:** Switch from in-memory to Pinecone
- **Databases:** Switch from SQLite to PostgreSQL

### 2. Testability - Real vs Fake Adapters

In production, use real adapters:

```python
# production.py
app = compose.webapp(
    kits=[
        compose.cache(backend="redis", redis_url=os.environ["REDIS_URL"]),
    ],
)
```

In tests, inject fakes:

```python
# test_service.py
class FakeCacheAdapter(CacheAdapter):
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key.key)

    async def set(self, key, value, ttl=None):
        self.data[key.key] = value

# Create kit with fake adapter (no Redis needed!)
cache_kit = CacheKit.create(
    database=test_db,
    events=test_events,
    config={"backend": "memory"},
    cache_adapter=FakeCacheAdapter(),
)

# Test without external dependencies
await cache_kit.service.set("key", "value")
result = await cache_kit.service.get("key")
assert result == "value"
```

**Benefits:**

- ✅ Tests run without external services (fast)
- ✅ Tests are deterministic (no flaky Redis calls)
- ✅ Tests are isolated (no shared state)

### 3. Architecture Enforcement

The composition root pattern is **enforced by the compiler** through import-linter rules.

If you try to violate it:

```python
# portico/kits/cache/service.py
from portico.adapters.cache import RedisCacheAdapter  # ❌ VIOLATION!

class CacheService:
    def __init__(self):
        self.adapter = RedisCacheAdapter()  # ❌ Direct coupling!
```

The build **fails immediately**:

```
❌ Import-linter check failed
Contract violated: Kits cannot import adapters

portico/kits/cache/service.py imports portico.adapters.cache.RedisCacheAdapter

Kits must depend on port interfaces only.
Adapters can only be imported in compose.py.
```

This makes architectural violations **impossible to commit** - the CI/CD pipeline catches them before they reach production.

---

## Composition Examples

Let's see how composition works in real-world scenarios.

### Example 1: Minimal Webapp (Development)

The simplest possible Portico application - user management only.

```python
from portico import compose

app = compose.webapp(
    database_url="sqlite+aiosqlite:///app.db",
    kits=[
        compose.user(password_min_length=8),
        compose.auth(session_secret="your-32-char-secret-key-here"),
    ],
)

# Initialize (creates database tables)
await app.initialize()

# Use services
user = await app.kits["user"].service.create_user(
    CreateUserRequest(email="user@example.com", password="secure123")
)

session = await app.kits["auth"].service.create_session(user.id)
```

**What gets created:**

- SQLite database (file: `app.db`)
- UserKit with SqlAlchemyUserRepository
- AuthKit with SessionRepository (depends on UserKit)
- Database tables: `users`, `auth_sessions`

**Adapters instantiated:**

- None (repositories use database directly)

### Example 2: Full-Featured Webapp (Production)

A complete production application with all core features.

```python
from portico import compose
import os

app = compose.webapp(
    # Production PostgreSQL database
    database_url=os.environ["DATABASE_URL"],

    # Database connection pool settings
    pool_size=20,
    max_overflow=10,

    kits=[
        # User management
        compose.user(
            password_min_length=12,
            require_email_verification=True,
        ),

        # Session-based authentication
        compose.auth(
            session_secret=os.environ["SESSION_SECRET"],
            session_timeout_minutes=60,
            enable_session_storage=True,
        ),

        # Hierarchical groups/organizations
        compose.group(
            max_groups_per_user=100,
        ),

        # Role-based access control
        compose.rbac(),

        # High-performance Redis caching
        compose.cache(
            backend="redis",
            redis_url=os.environ["REDIS_URL"],
            default_ttl_seconds=3600,
            default_namespace="myapp",
        ),

        # OpenAI LLM integration
        compose.llm(
            provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            temperature=0.7,
        ),

        # Google Cloud Storage for files
        compose.file(
            storage_backend="gcs",
            gcs_bucket=os.environ["GCS_BUCKET"],
            gcs_project=os.environ["GCS_PROJECT"],
            gcs_credentials_path="/secrets/gcs-credentials.json",
            max_file_size_mb=100,
        ),

        # RAG with managed Graphlit platform
        compose.rag(
            use_managed_rag=True,
            managed_rag_provider="graphlit",
            managed_rag_config={
                "environment_id": os.environ["GRAPHLIT_ENV_ID"],
                "organization_id": os.environ["GRAPHLIT_ORG_ID"],
                "jwt_secret": os.environ["GRAPHLIT_JWT_SECRET"],
            },
        ),

        # Audit logging for compliance
        compose.audit(
            enable_auditing=True,
            retention_days=90,
        ),

        # In-app notifications
        compose.browser_notification(),
    ],
)

await app.initialize()
```

**What gets created:**

- PostgreSQL database with connection pooling
- Full user/auth/group/rbac stack with automatic dependency injection
- Redis cache adapter for high performance
- OpenAI provider for LLM completions
- GCS adapter for scalable file storage
- Graphlit managed RAG platform
- Audit logging to database
- Browser notification system

**Adapters instantiated:**

- RedisCacheAdapter
- OpenAIProvider
- GCSFileStorageAdapter
- GraphlitPlatform
- SqlAlchemyAuditAdapter
- BrowserNotificationAdapter

### Example 3: Environment-Based Configuration

Adapt your application based on the deployment environment.

```python
from portico import compose
import os

# Detect environment
is_production = os.getenv("ENV") == "production"
is_staging = os.getenv("ENV") == "staging"
is_development = not (is_production or is_staging)

# Choose database
if is_production:
    database_url = os.environ["DATABASE_URL"]  # PostgreSQL
elif is_staging:
    database_url = os.environ["STAGING_DATABASE_URL"]
else:
    database_url = "sqlite+aiosqlite:///dev.db"

# Choose cache backend
def get_cache_kit():
    if os.getenv("REDIS_URL"):
        return compose.cache(
            backend="redis",
            redis_url=os.environ["REDIS_URL"],
            default_ttl_seconds=3600,
        )
    else:
        return compose.cache(backend="memory")

# Choose file storage
def get_file_kit():
    if is_production:
        return compose.file(
            storage_backend="gcs",
            gcs_bucket=os.environ["GCS_BUCKET"],
            gcs_project=os.environ["GCS_PROJECT"],
        )
    else:
        return compose.file(
            storage_backend="local",
            storage_path="./uploads",
        )

# Choose LLM model
def get_llm_kit():
    if is_production:
        model = "gpt-4"  # Best quality
    elif is_staging:
        model = "gpt-4o-mini"  # Good quality, lower cost
    else:
        model = "gpt-4o-mini"  # Development

    return compose.llm(
        provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        model=model,
    )

# Compose application
app = compose.webapp(
    database_url=database_url,
    kits=[
        compose.user(),
        compose.auth(session_secret=os.environ["SESSION_SECRET"]),
        get_cache_kit(),
        get_file_kit(),
        get_llm_kit(),
    ],
)
```

**Development environment:**

- SQLite database
- Memory cache
- Local file storage
- GPT-4o-mini for cost savings

**Production environment:**

- PostgreSQL database
- Redis cache
- GCS file storage
- GPT-4 for best quality

**Same code, different infrastructure!**

### Example 4: Testing with Fake Adapters

Use real adapters in production, fake adapters in tests.

**Production:**

```python
# app.py
from portico import compose
import os

app = compose.webapp(
    database_url=os.environ["DATABASE_URL"],
    kits=[
        compose.cache(
            backend="redis",
            redis_url=os.environ["REDIS_URL"],
        ),
    ],
)
```

**Testing:**

```python
# test_cache_service.py
import pytest
from portico.ports.cache import CacheAdapter, CacheKey, CacheEntry
from portico.kits.cache import CacheKit
from portico.database import Database
from portico.events import EventBus

class FakeCacheAdapter(CacheAdapter):
    """In-memory fake for testing."""

    def __init__(self):
        self._data = {}

    async def get(self, key: CacheKey):
        entry = self._data.get(key.key)
        return entry.value if entry else None

    async def set(self, key: CacheKey, value, ttl=None):
        self._data[key.key] = CacheEntry(key=key.key, value=value)

    async def delete(self, key: CacheKey):
        self._data.pop(key.key, None)

    async def clear(self):
        self._data.clear()

@pytest.fixture
async def cache_kit():
    """Create CacheKit with fake adapter."""
    database = Database("sqlite+aiosqlite:///:memory:")
    events = EventBus()

    # Inject fake adapter
    fake_adapter = FakeCacheAdapter()

    kit = CacheKit.create(
        database=database,
        events=events,
        config={"backend": "memory"},
        cache_adapter=fake_adapter,
    )

    yield kit

async def test_cache_operations(cache_kit):
    """Test cache without Redis."""
    service = cache_kit.service

    # Set value
    await service.set("user:123", {"name": "John"}, ttl=60)

    # Get value
    result = await service.get("user:123")
    assert result == {"name": "John"}

    # Delete value
    await service.delete("user:123")
    result = await service.get("user:123")
    assert result is None
```

**Benefits:**

- ✅ Tests run without Redis (fast)
- ✅ Tests are deterministic (no network calls)
- ✅ Tests are isolated (no shared state)
- ✅ Same kit code works in tests and production

---

## Summary

The composition root is the **architectural foundation** of Portico applications.

### Key Takeaways

✅ **Single Responsibility:** `compose.py` is the ONLY place adapters are imported and wired

✅ **Configuration-Driven:** Change implementations by changing configuration, not code

✅ **Automatic Dependencies:** Kit dependencies are resolved automatically via type inspection

✅ **Type-Safe:** Configuration is validated using dataclasses with type hints

✅ **Swappable:** Switch from memory cache to Redis by changing one line

✅ **Testable:** Use real adapters in production, fake adapters in tests

✅ **Enforced:** Architecture rules are compiler-enforced via import-linter

### The Composition Flow

```
1. You write configuration:
   compose.cache(backend="redis", redis_url="...")

2. Factory function captures config:
   def cache(**config):
       def factory(database, events):
           ...

3. webapp() calls factory:
   kit = factory(database, events)

4. Factory imports adapter (ONLY place!):
   from portico.adapters.cache import RedisCacheAdapter

5. Factory instantiates adapter:
   adapter = RedisCacheAdapter(redis_url=...)

6. Factory creates kit with adapter:
   return CacheKit.create(..., cache_adapter=adapter)

7. webapp() registers kit:
   app.register_kit("cache", kit)

8. Application uses kit:
   await app.kits["cache"].service.set("key", "value")
```

### When Building with Portico

**Do:**

- ✅ Put all adapter imports in `compose.py`
- ✅ Use factory functions for kit creation
- ✅ Configure via environment variables
- ✅ Validate configuration early
- ✅ Use type hints for dependency injection
- ✅ Test with fake adapters
- ✅ Group related kits together

**Don't:**

- ❌ Import adapters in kits, ports, or application code
- ❌ Hardcode secrets in source code
- ❌ Manually wire kit dependencies (auto-resolution handles it)
- ❌ Skip configuration validation
- ❌ Couple business logic to specific adapters

### Related Documentation

- **[Ports](ports/index.md)** - Learn about the interfaces kits depend on
- **[Kits](kits/index.md)** - Understand the business logic layer
- **[Adapters](adapters/index.md)** - Explore concrete implementations
- **[Demos](demos.md)** - See real-world composition examples

The composition root makes Portico's hexagonal architecture not just a pattern, but an **enforced architectural guarantee**. Change one line of configuration, and your entire infrastructure adapts - without touching a single line of business logic.
