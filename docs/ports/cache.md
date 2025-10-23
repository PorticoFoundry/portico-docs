# Cache Port

## Overview

The Cache Port defines the contract for cache storage backends in Portico applications. It provides a flexible, feature-rich caching abstraction that supports multiple backends (in-memory, Redis, hybrid) with advanced capabilities like tag-based invalidation, namespace isolation, TTL management, and performance monitoring.

**Purpose**: Abstract cache storage operations to enable high-performance data access with pluggable storage backends.

**Domain**: Performance optimization, data access layer

**Key Capabilities**:

- Key-value storage with optional time-to-live (TTL)
- Namespace-based key isolation for multi-tenant scenarios
- Tag-based cache invalidation for related data
- Pattern-based and bulk deletion operations
- Performance statistics and monitoring
- Function call result caching with automatic key generation
- Expired entry cleanup and maintenance

**Port Type**: Adapter (infrastructure abstraction)

**When to Use**:

- Applications requiring fast data access and reduced database load
- Multi-tenant applications needing isolated cache namespaces
- Systems with complex cache invalidation patterns
- Applications requiring cache performance monitoring
- Services caching expensive computations or API calls

## Architecture Role

The Cache Port sits at the boundary between your application's business logic (kits) and cache storage infrastructure. It enables kits to cache data without depending on specific cache technologies.

```
┌─────────────────────────────────────────┐
│  Kits (Business Logic)                  │
│  - CacheKit wraps CacheAdapter          │
│  - Services use CacheService            │
└─────────────────┬───────────────────────┘
                  │ depends on
                  ↓
┌─────────────────────────────────────────┐
│  Cache Port (Interface)                 │
│  - CacheAdapter (ABC)                   │
│  - CacheKey, CacheEntry, CacheStats     │
└─────────────────┬───────────────────────┘
                  ↑ implements
                  │
┌─────────────────────────────────────────┐
│  Adapters (Implementations)             │
│  - MemoryCacheAdapter (in-memory)       │
│  - RedisCacheAdapter (Redis backend)    │
│  - HybridCacheAdapter (L1/L2 cache)     │
└─────────────────────────────────────────┘
```

**Key Responsibilities**:

- Define cache key structure with metadata (namespace, tags)
- Specify cache entry lifecycle (creation, expiration, access tracking)
- Provide cache statistics for monitoring
- Abstract storage operations (get, set, delete, clear)
- Support advanced invalidation patterns (tags, patterns, namespaces)

## Domain Models

### CacheKey

Represents a cache key with metadata for organization and invalidation.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `key` | `str` | Yes | - | The unique cache key identifier |
| `namespace` | `Optional[str]` | No | `None` | Optional namespace for key isolation |
| `tags` | `list[str]` | No | `[]` | Tags for group-based invalidation |

**Properties**:

##### full_key

```python
full_key -> str
```

Returns the complete cache key including namespace prefix if applicable. Format: `{namespace}:{key}` or just `key` if no namespace.

**Class Methods**:

##### from_function_call

```python
@classmethod from_function_call(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], namespace: Optional[str] = None, tags: Optional[list[str]] = None) -> CacheKey
```

Creates a cache key from function call parameters. Generates a deterministic key by hashing the function name, module, and arguments.

**Example**:

```python
from portico.ports.cache import CacheKey

# Simple key
key = CacheKey(key="user:123")
print(key.full_key)  # "user:123"

# Key with namespace
key = CacheKey(
    key="profile",
    namespace="user:123",
    tags=["user", "profile"]
)
print(key.full_key)  # "user:123:profile"

# Key from function call (automatic)
def get_user_profile(user_id: str) -> dict:
    ...

key = CacheKey.from_function_call(
    func=get_user_profile,
    args=(user_id,),
    kwargs={},
    namespace="profiles",
    tags=["user", "profile"]
)
# Generates: "profiles:__main__.get_user_profile:a1b2c3d4e5f6g7h8"
```

**Immutability**: `CacheKey` is frozen (immutable) - all fields are set at creation time.

### CacheEntry

Represents a cached value with metadata for expiration and access tracking.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `key` | `CacheKey` | Yes | - | The cache key for this entry |
| `value` | `Any` | Yes | - | The cached value (can be any type) |
| `created_at` | `datetime` | No | Current UTC time | When entry was created |
| `expires_at` | `Optional[datetime]` | No | `None` | When entry expires (None = no expiration) |
| `hit_count` | `int` | No | `0` | Number of times entry has been accessed |
| `last_accessed` | `datetime` | No | Current UTC time | Last access timestamp |

**Properties**:

##### is_expired

```python
is_expired -> bool
```

Returns `True` if the entry has expired (current time > `expires_at`), `False` otherwise. Entries with `expires_at=None` never expire.

**Methods**:

##### touch

```python
touch() -> CacheEntry
```

Returns a new `CacheEntry` with updated `last_accessed` time and incremented `hit_count`. Used internally when accessing cached values.

**Example**:

```python
from datetime import timedelta
from portico.ports.cache import CacheKey, CacheEntry

key = CacheKey(key="user:123")
entry = CacheEntry(
    key=key,
    value={"name": "Alice", "email": "alice@example.com"},
    expires_at=datetime.now(UTC) + timedelta(hours=1)
)

# Check expiration
if not entry.is_expired:
    # Entry is still valid
    data = entry.value

# Update access metadata
entry = entry.touch()
print(entry.hit_count)  # 1
print(entry.last_accessed)  # Recent timestamp
```

**Immutability**: `CacheEntry` is frozen - use `touch()` or `model_copy()` to create updated versions.

### CacheStats

Performance statistics for cache monitoring and optimization.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `total_entries` | `int` | Yes | - | Total number of cached entries |
| `total_hits` | `int` | Yes | - | Total cache hits |
| `total_misses` | `int` | Yes | - | Total cache misses |
| `hit_rate` | `float` | Yes | - | Cache hit rate (0.0 to 1.0) |
| `memory_usage_bytes` | `Optional[int]` | No | `None` | Approximate memory usage in bytes |
| `oldest_entry` | `Optional[datetime]` | No | `None` | Timestamp of oldest cached entry |
| `newest_entry` | `Optional[datetime]` | No | `None` | Timestamp of newest cached entry |

**Example**:

```python
stats = await cache_adapter.get_stats()

print(f"Cache entries: {stats.total_entries}")
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Hits: {stats.total_hits}, Misses: {stats.total_misses}")

if stats.memory_usage_bytes:
    print(f"Memory: {stats.memory_usage_bytes / 1024 / 1024:.2f} MB")
```

**Immutability**: `CacheStats` is frozen - snapshots represent point-in-time cache state.

## Port Interfaces

### CacheAdapter

The `CacheAdapter` abstract base class defines the contract for all cache storage backends.

#### Retrieval Operations

##### get

```python
async def get(key: CacheKey) -> Optional[CacheEntry]
```

Retrieves a cached value by key.

**Parameters**:

- `key`: The cache key to retrieve

**Returns**: `CacheEntry` if found and not expired, `None` otherwise.

**Example**:

```python
key = CacheKey(key="user:123")
entry = await cache_adapter.get(key)

if entry and not entry.is_expired:
    user_data = entry.value
else:
    # Cache miss - fetch from database
    user_data = await db.get_user(123)
```

##### exists

```python
async def exists(key: CacheKey) -> bool
```

Checks if a key exists in the cache (without retrieving the value).

**Parameters**:

- `key`: The cache key to check

**Returns**: `True` if key exists in cache, `False` otherwise.

**Example**:

```python
if await cache_adapter.exists(key):
    # Key exists, get will succeed
    entry = await cache_adapter.get(key)
```

#### Storage Operations

##### set

```python
async def set(key: CacheKey, value: Any, ttl: Optional[timedelta] = None) -> None
```

Stores a value in the cache with optional time-to-live.

**Parameters**:

- `key`: The cache key to store
- `value`: The value to cache (can be any serializable type)
- `ttl`: Optional time-to-live duration. If provided, entry expires after this duration.

**Example**:

```python
from datetime import timedelta

# Cache without expiration
await cache_adapter.set(
    key=CacheKey(key="user:123"),
    value={"name": "Alice"}
)

# Cache with 1-hour TTL
await cache_adapter.set(
    key=CacheKey(key="session:xyz"),
    value=session_data,
    ttl=timedelta(hours=1)
)

# Cache with tags for invalidation
await cache_adapter.set(
    key=CacheKey(
        key="product:456",
        tags=["product", "catalog"]
    ),
    value=product_data,
    ttl=timedelta(minutes=30)
)
```

#### Deletion Operations

##### delete

```python
async def delete(key: CacheKey) -> bool
```

Deletes a single cached value.

**Parameters**:

- `key`: The cache key to delete

**Returns**: `True` if key existed and was deleted, `False` if key didn't exist.

**Example**:

```python
deleted = await cache_adapter.delete(CacheKey(key="user:123"))
if deleted:
    print("Cache entry removed")
```

##### delete_by_pattern

```python
async def delete_by_pattern(pattern: str) -> int
```

Deletes all keys matching a pattern (supports wildcards).

**Parameters**:

- `pattern`: Pattern to match cache keys against (e.g., `"user:*"`, `"session:*:profile"`)

**Returns**: Number of keys deleted.

**Example**:

```python
# Delete all user caches
count = await cache_adapter.delete_by_pattern("user:*")
print(f"Deleted {count} user cache entries")

# Delete specific pattern
count = await cache_adapter.delete_by_pattern("temp:*:data")
```

##### delete_by_namespace

```python
async def delete_by_namespace(namespace: str) -> int
```

Deletes all keys in a specific namespace.

**Parameters**:

- `namespace`: The namespace to delete all keys from

**Returns**: Number of keys deleted.

**Example**:

```python
# Delete all caches in user's namespace
count = await cache_adapter.delete_by_namespace("user:123")

# Clear tenant namespace
count = await cache_adapter.delete_by_namespace(f"tenant:{tenant_id}")
```

##### delete_by_tags

```python
async def delete_by_tags(tags: list[str]) -> int
```

Deletes all entries with any of the given tags.

**Parameters**:

- `tags`: List of tags to match for deletion (entries matching ANY tag are deleted)

**Returns**: Number of entries deleted.

**Example**:

```python
# Invalidate all product-related caches
count = await cache_adapter.delete_by_tags(["product", "catalog"])

# Invalidate user and profile caches
count = await cache_adapter.delete_by_tags(["user:123"])
```

##### clear

```python
async def clear() -> None
```

Clears all cached entries (use with caution in production).

**Example**:

```python
# Clear entire cache
await cache_adapter.clear()
```

#### Maintenance Operations

##### cleanup_expired

```python
async def cleanup_expired() -> int
```

Removes expired entries from the cache.

**Returns**: Number of expired entries removed.

**Example**:

```python
# Periodic cleanup (run in background task)
removed = await cache_adapter.cleanup_expired()
logger.info(f"Cleaned up {removed} expired cache entries")
```

##### get_stats

```python
async def get_stats() -> CacheStats
```

Retrieves cache performance statistics.

**Returns**: `CacheStats` object with current cache metrics.

**Example**:

```python
stats = await cache_adapter.get_stats()

# Log performance metrics
logger.info(
    "cache_stats",
    entries=stats.total_entries,
    hit_rate=stats.hit_rate,
    hits=stats.total_hits,
    misses=stats.total_misses
)

# Alert on low hit rate
if stats.hit_rate < 0.5:
    logger.warning("Low cache hit rate", hit_rate=stats.hit_rate)
```

## Usage Patterns

### Basic Caching

```python
from portico.ports.cache import CacheKey, CacheAdapter
from datetime import timedelta

async def get_user_profile(
    user_id: str,
    cache: CacheAdapter
) -> dict:
    # Check cache first
    cache_key = CacheKey(key=f"profile:{user_id}")
    entry = await cache.get(cache_key)

    if entry and not entry.is_expired:
        return entry.value

    # Cache miss - fetch from database
    profile = await db.get_user_profile(user_id)

    # Store in cache with 1-hour TTL
    await cache.set(
        key=cache_key,
        value=profile,
        ttl=timedelta(hours=1)
    )

    return profile
```

### Namespace Isolation

```python
# Multi-tenant caching
async def cache_tenant_data(
    tenant_id: str,
    data_key: str,
    data: Any,
    cache: CacheAdapter
):
    key = CacheKey(
        key=data_key,
        namespace=f"tenant:{tenant_id}"
    )
    await cache.set(key, data, ttl=timedelta(hours=2))

# Clear all data for a tenant
async def clear_tenant_cache(tenant_id: str, cache: CacheAdapter):
    count = await cache.delete_by_namespace(f"tenant:{tenant_id}")
    print(f"Cleared {count} cache entries for tenant {tenant_id}")
```

### Tag-Based Invalidation

```python
# Cache with tags
async def cache_product(product_id: str, product: dict, cache: CacheAdapter):
    key = CacheKey(
        key=f"product:{product_id}",
        tags=["product", f"category:{product['category_id']}"]
    )
    await cache.set(key, product, ttl=timedelta(hours=1))

# Invalidate all products in a category
async def invalidate_category(category_id: str, cache: CacheAdapter):
    count = await cache.delete_by_tags([f"category:{category_id}"])
    print(f"Invalidated {count} products in category")

# Invalidate all product caches
async def invalidate_all_products(cache: CacheAdapter):
    count = await cache.delete_by_tags(["product"])
    print(f"Invalidated {count} product caches")
```

### Function Result Caching

```python
from portico.ports.cache import CacheKey

async def expensive_computation(x: int, y: int) -> int:
    # Generate cache key from function signature
    cache_key = CacheKey.from_function_call(
        func=expensive_computation,
        args=(x, y),
        kwargs={},
        tags=["computation"]
    )

    # Check cache
    entry = await cache.get(cache_key)
    if entry and not entry.is_expired:
        return entry.value

    # Perform computation
    result = x ** 2 + y ** 2  # Expensive operation

    # Cache result
    await cache.set(
        key=cache_key,
        value=result,
        ttl=timedelta(minutes=15)
    )

    return result
```

### Pattern-Based Deletion

```python
# Delete all temporary caches
await cache.delete_by_pattern("temp:*")

# Delete user session caches
await cache.delete_by_pattern(f"session:{user_id}:*")

# Delete all caches for a specific date
await cache.delete_by_pattern(f"*:2024-01-15:*")
```

### Cache Performance Monitoring

```python
from portico.kits.logging import get_logger

logger = get_logger(__name__)

async def monitor_cache_performance(cache: CacheAdapter):
    stats = await cache.get_stats()

    logger.info(
        "cache_performance",
        total_entries=stats.total_entries,
        hit_rate=stats.hit_rate,
        total_hits=stats.total_hits,
        total_misses=stats.total_misses,
        memory_usage_mb=stats.memory_usage_bytes / 1024 / 1024 if stats.memory_usage_bytes else None
    )

    # Alert on performance issues
    if stats.hit_rate < 0.6:
        logger.warning(
            "low_cache_hit_rate",
            hit_rate=stats.hit_rate,
            recommendation="Review cache TTLs and access patterns"
        )

    if stats.memory_usage_bytes and stats.memory_usage_bytes > 1_000_000_000:  # 1GB
        logger.warning(
            "high_cache_memory_usage",
            memory_mb=stats.memory_usage_bytes / 1024 / 1024,
            recommendation="Consider cache size limits or cleanup"
        )
```

### Conditional Caching

```python
async def get_user_data(
    user_id: str,
    cache: CacheAdapter,
    force_refresh: bool = False
) -> dict:
    cache_key = CacheKey(
        key=f"user:{user_id}",
        tags=["user"]
    )

    # Skip cache if force refresh
    if not force_refresh:
        entry = await cache.get(cache_key)
        if entry and not entry.is_expired:
            return entry.value

    # Fetch fresh data
    user_data = await db.get_user(user_id)

    # Update cache
    await cache.set(
        key=cache_key,
        value=user_data,
        ttl=timedelta(minutes=30)
    )

    return user_data
```

## Integration with Kits

The Cache Port is used by the **CacheKit** to provide high-level caching services to your application.

### Accessing CacheKit

```python
from portico import compose
from portico.kits.fastapi import Dependencies

# Configure cache in webapp
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.cache(
            backend="redis",
            redis_url="redis://localhost:6379/0",
            default_ttl_seconds=3600
        ),
    ],
)

deps = Dependencies(app)

# Access cache service
cache_service = app.kits["cache"].service
```

### Using CacheService

The `CacheService` provides a higher-level API built on top of `CacheAdapter`:

```python
from portico.ports.cache import CacheKey
from datetime import timedelta

# Store value
await cache_service.set(
    key="user:123",
    value=user_data,
    ttl=timedelta(hours=1),
    namespace="users",
    tags=["user", "profile"]
)

# Retrieve value
user_data = await cache_service.get(key="user:123", namespace="users")

# Invalidate by tags
await cache_service.invalidate_tags(["user:123"])

# Get statistics
stats = await cache_service.get_stats()
```

### Dependency Injection

```python
from portico.kits.cache import CacheService

async def get_cache_service(
    cache_kit = Depends(lambda: app.kits["cache"])
) -> CacheService:
    return cache_kit.service

@app.get("/product/{product_id}")
async def get_product(
    product_id: str,
    cache: CacheService = Depends(get_cache_service)
):
    # Try cache first
    cached = await cache.get(
        key=f"product:{product_id}",
        namespace="products"
    )

    if cached:
        return cached

    # Fetch and cache
    product = await db.get_product(product_id)
    await cache.set(
        key=f"product:{product_id}",
        value=product,
        namespace="products",
        ttl=timedelta(hours=1)
    )

    return product
```

## Best Practices

### 1. Use Namespaces for Isolation

```python
# ✅ GOOD: Isolated namespaces
CacheKey(key="profile", namespace=f"user:{user_id}")
CacheKey(key="settings", namespace=f"tenant:{tenant_id}")

# ❌ BAD: No isolation
CacheKey(key=f"user_{user_id}_profile")
```

### 2. Tag Related Data

```python
# ✅ GOOD: Tags enable bulk invalidation
CacheKey(
    key=f"product:{product_id}",
    tags=["product", f"category:{category_id}", f"vendor:{vendor_id}"]
)

# Invalidate all products in category
await cache.delete_by_tags([f"category:{category_id}"])
```

### 3. Set Appropriate TTLs

```python
# ✅ GOOD: Different TTLs for different data
await cache.set(key=session_key, value=session, ttl=timedelta(hours=1))  # Sessions
await cache.set(key=config_key, value=config, ttl=timedelta(days=1))    # Config
await cache.set(key=temp_key, value=temp, ttl=timedelta(minutes=5))     # Temporary

# ❌ BAD: No TTL for temporary data
await cache.set(key=temp_key, value=temp)  # Never expires!
```

### 4. Handle Cache Misses Gracefully

```python
# ✅ GOOD: Fallback to source
entry = await cache.get(key)
if entry and not entry.is_expired:
    data = entry.value
else:
    data = await fetch_from_source()
    await cache.set(key, data, ttl=timedelta(hours=1))

# ❌ BAD: Assume cache always has data
data = (await cache.get(key)).value  # May raise AttributeError!
```

### 5. Monitor Cache Performance

```python
# ✅ GOOD: Regular monitoring
async def monitor_cache():
    stats = await cache.get_stats()
    logger.info("cache_stats", hit_rate=stats.hit_rate, entries=stats.total_entries)

    if stats.hit_rate < 0.5:
        logger.warning("low_hit_rate", hit_rate=stats.hit_rate)

# Schedule periodic monitoring
asyncio.create_task(periodic_monitor(interval=300))
```

### 6. Clean Up Expired Entries

```python
# ✅ GOOD: Periodic cleanup
async def cleanup_task():
    while True:
        await asyncio.sleep(3600)  # Every hour
        removed = await cache.cleanup_expired()
        logger.info("cache_cleanup", removed=removed)

asyncio.create_task(cleanup_task())
```

### 7. Invalidate on Updates

```python
# ✅ GOOD: Invalidate when data changes
async def update_product(product_id: str, updates: dict):
    # Update database
    await db.update_product(product_id, updates)

    # Invalidate cache
    await cache.delete(CacheKey(key=f"product:{product_id}"))

# ✅ EVEN BETTER: Use tags
await cache.delete_by_tags([f"product:{product_id}"])
```

### 8. Use Type-Safe Keys

```python
# ✅ GOOD: CacheKey objects
from portico.ports.cache import CacheKey

key = CacheKey(key="user:123", namespace="profiles")
await cache.get(key)

# ❌ BAD: Raw strings (if adapter accepts them)
await cache.get("user:123:profiles")  # Error-prone
```

### 9. Avoid Over-Caching

```python
# ✅ GOOD: Cache expensive operations only
async def get_analytics_report():  # Expensive query
    key = CacheKey(key="daily_report")
    # Cache this!

# ❌ BAD: Caching trivial operations
async def get_user_id():  # Simple field access
    # Don't cache this
```

### 10. Document Cache Dependencies

```python
async def get_user_dashboard(user_id: str) -> dict:
    """
    Get user dashboard data.

    Cache dependencies:
    - Namespace: user:{user_id}
    - Tags: user, dashboard
    - TTL: 15 minutes

    Invalidate when:
    - User profile updated
    - User preferences changed
    - Dashboard settings modified
    """
    # Implementation...
```

## FAQs

### When should I use namespaces vs. key prefixes?

**Use namespaces** when you need to:

- Clear all caches for a tenant/user in one operation
- Logically isolate cache data
- Implement multi-tenancy

**Use key prefixes** when:

- You need pattern-based matching (`"user:*"`)
- Namespace isolation isn't required

```python
# Namespace (better for bulk operations)
CacheKey(key="profile", namespace=f"user:{user_id}")
await cache.delete_by_namespace(f"user:{user_id}")

# Prefix (better for pattern matching)
CacheKey(key=f"user:{user_id}:profile")
await cache.delete_by_pattern(f"user:{user_id}:*")
```

### How do tags differ from namespaces?

- **Namespaces**: Hierarchical isolation (one namespace per key)
- **Tags**: Multi-dimensional categorization (multiple tags per key)

```python
# Can delete by namespace OR by any tag
key = CacheKey(
    key="product:123",
    namespace="catalog",
    tags=["product", "category:electronics", "featured"]
)

await cache.delete_by_namespace("catalog")           # Deletes this
await cache.delete_by_tags(["product"])             # Deletes this
await cache.delete_by_tags(["category:electronics"]) # Deletes this
```

### What happens if I don't specify a TTL?

Entries without TTL never expire automatically. You must manually delete them or use `cache.clear()`. This is appropriate for:

- Configuration data that rarely changes
- Reference data (e.g., country codes)
- Data with manual invalidation logic

```python
# No TTL - cache forever
await cache.set(key=config_key, value=config)

# With TTL - auto-expire
await cache.set(key=session_key, value=session, ttl=timedelta(hours=1))
```

### How is cache expiration handled?

1. **On retrieval**: `get()` checks `is_expired` before returning
2. **Manual cleanup**: Call `cleanup_expired()` to remove expired entries
3. **Adapter-specific**: Some adapters (Redis) have native TTL support

```python
# Automatically handled on get
entry = await cache.get(key)  # Returns None if expired

# Manual cleanup
removed = await cache.cleanup_expired()
```

### Can I cache complex objects?

Yes! `CacheEntry.value` accepts `Any` type. Adapters handle serialization:

- **MemoryCacheAdapter**: Stores objects directly (no serialization)
- **RedisCacheAdapter**: Uses JSON/pickle serialization
- **HybridCacheAdapter**: L1 (memory) stores objects, L2 (Redis) serializes

```python
# Cache complex objects
await cache.set(
    key=CacheKey(key="complex"),
    value={
        "user": user_obj,
        "settings": settings_obj,
        "metadata": {"last_login": datetime.now()}
    }
)
```

### How do I implement cache-aside pattern?

The cache-aside (lazy loading) pattern:

1. Check cache for data
2. If miss, fetch from source
3. Store in cache for next time

```python
async def get_with_cache_aside(key: str) -> dict:
    cache_key = CacheKey(key=key)

    # 1. Try cache
    entry = await cache.get(cache_key)
    if entry and not entry.is_expired:
        return entry.value

    # 2. Fetch from source
    data = await fetch_from_database(key)

    # 3. Populate cache
    await cache.set(cache_key, data, ttl=timedelta(hours=1))

    return data
```

### Should I cache authenticated user data?

Yes, but with appropriate TTLs and invalidation:

```python
# ✅ GOOD: Cache with short TTL + invalidation
await cache.set(
    key=CacheKey(
        key="profile",
        namespace=f"user:{user_id}",
        tags=["user", "profile"]
    ),
    value=user_profile,
    ttl=timedelta(minutes=15)  # Short TTL for sensitive data
)

# Invalidate on updates
async def update_profile(user_id: str, updates: dict):
    await db.update_user(user_id, updates)
    await cache.delete_by_namespace(f"user:{user_id}")
```

### How do I test code using CacheAdapter?

Use the `MemoryCacheAdapter` in tests:

```python
import pytest
from portico.adapters.cache import MemoryCacheAdapter
from portico.ports.cache import CacheKey

@pytest.fixture
async def cache_adapter():
    return MemoryCacheAdapter(max_size=100)

async def test_caching(cache_adapter):
    key = CacheKey(key="test")

    # Store value
    await cache_adapter.set(key, "test_value")

    # Retrieve value
    entry = await cache_adapter.get(key)
    assert entry.value == "test_value"
```

### What's the difference between `delete_by_pattern` and `delete_by_tags`?

- **`delete_by_pattern`**: Matches key strings with wildcards (`"user:*"`)
- **`delete_by_tags`**: Matches tags metadata (`["user", "profile"]`)

Tags are more structured and explicit:

```python
# Pattern matching (string-based)
await cache.delete_by_pattern("user:123:*")  # Deletes "user:123:profile", "user:123:settings"

# Tag matching (metadata-based)
await cache.delete_by_tags(["user:123"])     # Deletes all entries tagged with "user:123"
```

### How do I handle cache stampede?

Cache stampede occurs when many requests simultaneously miss cache and hit the database. Solutions:

```python
from asyncio import Lock

locks: dict[str, Lock] = {}

async def get_with_lock(key: str) -> dict:
    cache_key = CacheKey(key=key)

    # Check cache
    entry = await cache.get(cache_key)
    if entry and not entry.is_expired:
        return entry.value

    # Acquire lock for this key
    if key not in locks:
        locks[key] = Lock()

    async with locks[key]:
        # Double-check cache (another request may have populated it)
        entry = await cache.get(cache_key)
        if entry and not entry.is_expired:
            return entry.value

        # Fetch and cache
        data = await fetch_from_database(key)
        await cache.set(cache_key, data, ttl=timedelta(hours=1))
        return data
```

### Can I use CacheAdapter for distributed caching?

Yes! Use `RedisCacheAdapter` for distributed caching across multiple application instances:

```python
from portico import compose

app = compose.webapp(
    kits=[
        compose.cache(
            backend="redis",
            redis_url="redis://cache-cluster:6379/0"
        )
    ]
)
```

All instances share the same Redis backend, ensuring cache consistency.
