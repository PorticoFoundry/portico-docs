# Cache Kit

## Overview

**Purpose**: Provide high-performance caching with multiple backend support (memory, Redis, hybrid) to reduce database load and improve application response times.

**Key Features**:

- Multiple backend options: in-memory, Redis, or hybrid (both)
- Simple get/set/delete operations with TTL support
- Advanced invalidation: by namespace, tags, or pattern matching
- Function result caching with decorators
- FastAPI route response caching
- Cache statistics and performance monitoring
- Automatic expiration and cleanup
- Namespace and tag-based organization

**Dependencies**:

- **Injected services**: None
- **Port dependencies**: CacheAdapter (storage backend interface)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose

# Basic configuration with memory backend
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.cache(
            backend="memory",
            max_memory_items=1000,
            default_ttl_seconds=3600,
        ),
    ]
)

# Access the cache service
cache_service = app.kits["cache"].service

# Basic caching operations
await cache_service.set("user:123", user_data, ttl=timedelta(minutes=5))
user_data = await cache_service.get("user:123")
await cache_service.delete("user:123")
```

## Core Concepts

### Cache Backends

The Cache Kit supports three backend types:

```python
# Memory backend - fast, volatile, single-process
compose.cache(
    backend="memory",
    max_memory_items=1000,  # LRU eviction when full
)

# Redis backend - persistent, distributed, shared across processes
compose.cache(
    backend="redis",
    redis_url="redis://localhost:6379",
)

# Hybrid backend - memory for hot data, Redis for persistence
compose.cache(
    backend="hybrid",
    redis_url="redis://localhost:6379",
    max_memory_items=500,  # Hot data size
)
```

**Memory Backend**:

- Fastest performance (no network I/O)
- Per-process cache (not shared)
- LRU eviction when `max_memory_items` reached
- Lost on restart
- Best for: Single-server apps, development, testing

**Redis Backend**:

- Shared across processes/servers
- Survives application restarts
- Network latency overhead
- Scalable with Redis cluster
- Best for: Multi-server apps, session storage, production

**Hybrid Backend**:

- Memory cache for frequently accessed keys (L1 cache)
- Redis for persistence and sharing (L2 cache)
- Automatic promotion to memory on access
- Best for: High-traffic apps needing both speed and persistence

### Time-To-Live (TTL)

All cached entries can have an expiration time:

```python
from datetime import timedelta

# Set with specific TTL
await cache_service.set(
    "session:abc123",
    session_data,
    ttl=timedelta(minutes=30)
)

# Use default TTL from config
await cache_service.set("temp_data", value)
# Uses default_ttl_seconds from config

# No expiration (cache forever until manually deleted)
await cache_service.set("static_config", config, ttl=None)
```

### Namespaces and Tags

Organize cache entries for efficient invalidation:

```python
# Namespaces group related entries
await cache_service.set(
    "profile",
    user_profile,
    namespace="user:123"
)

await cache_service.set(
    "settings",
    user_settings,
    namespace="user:123"
)

# Invalidate entire namespace at once
count = await cache_service.invalidate_namespace("user:123")
# Both profile and settings deleted

# Tags allow cross-cutting invalidation
await cache_service.set(
    "product:456",
    product_data,
    tags=["products", "category:electronics"]
)

await cache_service.set(
    "product:789",
    product_data,
    tags=["products", "category:electronics"]
)

# Invalidate by tag
count = await cache_service.invalidate_tags(["category:electronics"])
# Both products deleted
```

### Cache-Aside Pattern

The most common caching pattern - check cache first, then compute:

```python
# Manual cache-aside
async def get_user_profile(user_id: UUID):
    # Try cache first
    cached = await cache_service.get(f"profile:{user_id}")
    if cached is not None:
        return cached

    # Cache miss - fetch from database
    profile = await db.query(UserProfile).filter_by(user_id=user_id).first()

    # Store in cache
    await cache_service.set(
        f"profile:{user_id}",
        profile,
        ttl=timedelta(minutes=15)
    )

    return profile

# Or use get_or_set helper
async def get_user_profile(user_id: UUID):
    return await cache_service.get_or_set_async(
        key=f"profile:{user_id}",
        value_factory=lambda: fetch_profile_from_db(user_id),
        ttl=timedelta(minutes=15)
    )
```

## Configuration

### Required Settings

None - all settings have sensible defaults.

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `"memory" \| "redis" \| "hybrid"` | `"memory"` | Cache backend type |
| `default_ttl_seconds` | `int` | `3600` | Default time-to-live in seconds (1 hour) |
| `default_namespace` | `str \| None` | `None` | Default namespace for cache keys |
| `redis_url` | `str \| None` | `None` | Redis connection URL (required for redis/hybrid) |
| `max_memory_items` | `int` | `1000` | Maximum items in memory cache (LRU eviction) |
| `enable_stats` | `bool` | `True` | Enable cache statistics tracking |

**Example Configurations:**

```python
from portico import compose

# Development - simple memory cache
compose.cache(backend="memory")

# Production - Redis with custom TTL
compose.cache(
    backend="redis",
    redis_url="redis://:password@redis-host:6379/0",
    default_ttl_seconds=1800,  # 30 minutes
)

# High-performance - hybrid with namespacing
compose.cache(
    backend="hybrid",
    redis_url="redis://localhost:6379",
    max_memory_items=5000,
    default_namespace="myapp",
    enable_stats=True,
)
```

## Usage Examples

### Example 1: Caching Database Queries

```python
from datetime import timedelta
from uuid import UUID

async def get_user(user_id: UUID):
    cache_service = app.kits["cache"].service

    # Try cache first
    cached_user = await cache_service.get(
        f"user:{user_id}",
        namespace="users"
    )

    if cached_user:
        return cached_user

    # Cache miss - query database
    user = await user_service.get_by_id(user_id)

    if user:
        # Cache for 10 minutes
        await cache_service.set(
            f"user:{user_id}",
            user,
            ttl=timedelta(minutes=10),
            namespace="users",
            tags=["users"],
        )

    return user
```

### Example 2: Using the Cached Decorator

```python
from portico.kits.cache import cached
from datetime import timedelta

cache_service = app.kits["cache"].service

@cached(cache_service, ttl=timedelta(minutes=5))
async def get_expensive_data(user_id: int) -> dict:
    # This computation is expensive
    await asyncio.sleep(2)  # Simulate slow operation
    return {"user_id": user_id, "data": "expensive result"}

# First call - cache miss, takes 2 seconds
result = await get_expensive_data(123)

# Second call - cache hit, instant
result = await get_expensive_data(123)

# Custom cache key generation
@cached(
    cache_service,
    ttl=timedelta(hours=1),
    key_func=lambda user_id, include_details: f"user-{user_id}-{include_details}"
)
async def get_user_data(user_id: int, include_details: bool = False):
    return await fetch_from_api(user_id, include_details)
```

### Example 3: Caching FastAPI Routes

```python
from fastapi import FastAPI
from portico.kits.cache import cached_route
from datetime import timedelta

app_fastapi = FastAPI()
cache_service = app.kits["cache"].service

@app_fastapi.get("/products/{product_id}")
@cached_route(
    cache_service,
    ttl=timedelta(minutes=15),
    namespace="api",
    key_func=lambda req: f"product-{req.path_params['product_id']}"
)
async def get_product(product_id: int):
    # Expensive database query or API call
    product = await fetch_product(product_id)
    return {"product": product}

# Cache is automatically managed:
# - First request: cache miss, executes handler
# - Subsequent requests: cache hit, returns cached response
# - After 15 minutes: cache expires, re-executes handler
```

### Example 4: Cache Invalidation

```python
from uuid import UUID

async def update_user_profile(user_id: UUID, profile_data: dict):
    cache_service = app.kits["cache"].service

    # Update in database
    await user_service.update_profile(user_id, profile_data)

    # Invalidate cached user data
    await cache_service.delete(f"user:{user_id}", namespace="users")

    # Or invalidate all user-related caches
    await cache_service.invalidate_namespace(f"user:{user_id}")

    # Or invalidate by tag
    await cache_service.invalidate_tags(["user_profiles"])

    return {"success": True}
```

### Example 5: Monitoring Cache Performance

```python
@app.get("/admin/cache/stats")
async def get_cache_stats():
    cache_service = app.kits["cache"].service

    stats = await cache_service.get_stats()

    return {
        "total_entries": stats.total_entries,
        "total_hits": stats.total_hits,
        "total_misses": stats.total_misses,
        "hit_rate": f"{stats.hit_rate:.2%}",
        "memory_usage_bytes": stats.memory_usage_bytes,
        "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
        "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
    }
```

## Domain Models

### CacheKey

Represents a cache key with metadata for organization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `key` | `str` | - | The cache key identifier |
| `namespace` | `str \| None` | `None` | Optional namespace for grouping |
| `tags` | `List[str]` | `[]` | Tags for invalidation |

**Properties**:

- `full_key`: Returns `"{namespace}:{key}"` or just `key` if no namespace

**Methods**:

- `from_function_call()`: Generates cache key from function signature and arguments

### CacheEntry

Represents a cached value with metadata.

| Field | Type | Description |
|-------|------|-------------|
| `key` | `CacheKey` | The cache key |
| `value` | `Any` | The cached value |
| `created_at` | `datetime` | When entry was created (UTC) |
| `expires_at` | `datetime \| None` | When entry expires (None = never) |
| `hit_count` | `int` | Number of times accessed |
| `last_accessed` | `datetime` | Last access time (UTC) |

**Properties**:

- `is_expired`: Returns `True` if entry has passed expiration time

**Methods**:

- `touch()`: Updates access time and increments hit count

### CacheStats

Statistics about cache performance.

| Field | Type | Description |
|-------|------|-------------|
| `total_entries` | `int` | Total number of cached entries |
| `total_hits` | `int` | Total cache hits |
| `total_misses` | `int` | Total cache misses |
| `hit_rate` | `float` | Hit rate (0.0 to 1.0) |
| `memory_usage_bytes` | `int \| None` | Memory usage in bytes (if available) |
| `oldest_entry` | `datetime \| None` | Timestamp of oldest entry |
| `newest_entry` | `datetime \| None` | Timestamp of newest entry |

## Events

This kit does not publish any events. Cache operations are synchronous and don't trigger domain events.

## Best Practices

### 1. Choose Appropriate TTLs

Match TTL to data volatility:

```python
# ✅ GOOD - TTL matches data change frequency
# User profile - changes occasionally
await cache_service.set("profile", data, ttl=timedelta(minutes=15))

# API rate limits - need precise timing
await cache_service.set("rate_limit", count, ttl=timedelta(seconds=60))

# Static configuration - changes rarely
await cache_service.set("config", data, ttl=timedelta(hours=24))

# ❌ BAD - TTL doesn't match data
# Real-time stock prices with 1 hour cache
await cache_service.set("stock_price", price, ttl=timedelta(hours=1))
# Stale data causes incorrect decisions
```

### 2. Use Namespaces for Organization

Group related cache entries for efficient management:

```python
# ✅ GOOD - Namespace per entity
await cache_service.set(
    "profile",
    profile_data,
    namespace=f"user:{user_id}"
)

await cache_service.set(
    "preferences",
    prefs_data,
    namespace=f"user:{user_id}"
)

# Easy bulk invalidation
await cache_service.invalidate_namespace(f"user:{user_id}")

# ❌ BAD - No organization
await cache_service.set(f"user_{user_id}_profile", profile_data)
await cache_service.set(f"user_{user_id}_preferences", prefs_data)
# Must delete each key individually
```

### 3. Handle Cache Misses Gracefully

Always have a fallback when cache is empty:

```python
# ✅ GOOD - Always returns data
async def get_product(product_id: int):
    # Try cache
    cached = await cache_service.get(f"product:{product_id}")
    if cached is not None:
        return cached

    # Fallback to database
    product = await db.fetch_product(product_id)

    # Populate cache
    if product:
        await cache_service.set(
            f"product:{product_id}",
            product,
            ttl=timedelta(minutes=30)
        )

    return product

# ❌ BAD - Returns None on cache miss
async def get_product(product_id: int):
    return await cache_service.get(f"product:{product_id}")
    # User gets None if cache is cold!
```

### 4. Invalidate Cache on Updates

Keep cache in sync with data changes:

```python
# ✅ GOOD - Invalidate after update
async def update_product(product_id: int, data: dict):
    # Update database
    await db.update_product(product_id, data)

    # Invalidate cache
    await cache_service.delete(f"product:{product_id}")

    # Also invalidate related caches
    await cache_service.invalidate_tags(["products", f"category:{data['category']}"])

    return {"success": True}

# ❌ BAD - Cache becomes stale
async def update_product(product_id: int, data: dict):
    await db.update_product(product_id, data)
    return {"success": True}
    # Cache still has old data!
```

### 5. Use Cache-Aside for Read-Heavy Operations

Cache results of expensive computations:

```python
# ✅ GOOD - Cache expensive operations
@cached(cache_service, ttl=timedelta(minutes=10))
async def calculate_user_analytics(user_id: UUID):
    # Expensive aggregation query
    return await db.execute("""
        SELECT ... FROM events
        WHERE user_id = ? AND created_at > ?
        GROUP BY ...
    """)

# ❌ BAD - Compute every time
async def calculate_user_analytics(user_id: UUID):
    return await db.execute("SELECT ... GROUP BY ...")
    # Slow query runs on every request
```

### 6. Monitor Cache Hit Rates

Track performance and adjust TTLs:

```python
# ✅ GOOD - Regular monitoring
async def monitor_cache_performance():
    stats = await cache_service.get_stats()

    if stats.hit_rate < 0.7:  # Less than 70% hit rate
        logger.warning(
            "low_cache_hit_rate",
            hit_rate=stats.hit_rate,
            total_hits=stats.total_hits,
            total_misses=stats.total_misses
        )
        # Consider increasing TTL or cache size

# ❌ BAD - No monitoring
# Cache misconfigured, wasting resources
```

### 7. Use Tags for Cross-Cutting Invalidation

Tag entries for efficient bulk invalidation:

```python
# ✅ GOOD - Tag-based invalidation
# Cache multiple product variants
for variant in product_variants:
    await cache_service.set(
        f"variant:{variant.id}",
        variant,
        tags=["products", f"product:{product.id}"]
    )

# When product changes, invalidate all variants
await cache_service.invalidate_tags([f"product:{product.id}"])

# ❌ BAD - Manual tracking
# Must remember all variant IDs to invalidate
for variant_id in [1, 2, 3, 4, 5]:  # Hard to maintain
    await cache_service.delete(f"variant:{variant_id}")
```

## Security Considerations

### Sensitive Data in Cache

Be cautious when caching sensitive information:

- **Don't cache secrets**: Passwords, API keys, tokens
- **Encrypt sensitive data**: Consider encrypting cached PII
- **Use short TTLs**: For sensitive data, expire quickly
- **Namespace isolation**: Use user-specific namespaces for multi-tenant apps

```python
# Don't cache sensitive data indefinitely
await cache_service.set(
    f"user:{user_id}:pii",
    sensitive_data,
    ttl=timedelta(minutes=5),  # Short TTL
    namespace=f"tenant:{tenant_id}"  # Isolated
)
```

### Cache Poisoning

Validate data before caching:

```python
# Validate before caching to prevent poisoning
async def get_config(config_key: str):
    cached = await cache_service.get(f"config:{config_key}")
    if cached:
        return cached

    # Fetch from trusted source
    config = await fetch_from_db(config_key)

    # Validate before caching
    if is_valid_config(config):
        await cache_service.set(f"config:{config_key}", config)

    return config
```

### Memory Exhaustion

Prevent unbounded cache growth:

```python
# Set reasonable limits
compose.cache(
    backend="memory",
    max_memory_items=10000,  # LRU eviction
    default_ttl_seconds=1800,  # 30 min auto-expiration
)

# Schedule cleanup
async def cleanup_expired_cache():
    count = await cache_service.cleanup_expired()
    logger.info("cache_cleanup", entries_removed=count)

# Run every hour
scheduler.add_job(cleanup_expired_cache, "interval", hours=1)
```

## FAQs

### Q: When should I use memory vs Redis backend?

A: Use **memory** for:

- Single-server deployments
- Development/testing
- Caching expensive computations (not shared data)
- Maximum performance (no network I/O)

Use **Redis** for:

- Multi-server deployments (shared cache)
- Session storage (survives restarts)
- Production environments
- When cache persistence is important

Use **hybrid** for:

- High-traffic applications
- Best of both worlds (speed + persistence)
- Hot data in memory, cold data in Redis

### Q: How do I handle cache stampede?

A: Cache stampede occurs when many requests simultaneously try to regenerate an expired cache entry. Solutions:

```python
# Use async locking (not provided by kit, implement at app level)
import asyncio

cache_locks = {}

async def get_with_lock(key: str, factory):
    cached = await cache_service.get(key)
    if cached:
        return cached

    # Acquire lock for this key
    if key not in cache_locks:
        cache_locks[key] = asyncio.Lock()

    async with cache_locks[key]:
        # Double-check after acquiring lock
        cached = await cache_service.get(key)
        if cached:
            return cached

        # Only one request computes
        result = await factory()
        await cache_service.set(key, result)
        return result
```

### Q: Can I use the Cache Kit without Redis in production?

A: Yes, but with limitations:

- Memory backend works fine for single-server apps
- Each server has its own cache (not shared)
- Cache lost on restart
- Consider if your app needs shared cache across servers

### Q: How do I cache paginated results?

A: Include pagination parameters in the cache key:

```python
async def get_products_page(page: int, page_size: int):
    cache_key = f"products:page-{page}:size-{page_size}"

    cached = await cache_service.get(cache_key, namespace="products")
    if cached:
        return cached

    products = await db.fetch_products(page, page_size)

    await cache_service.set(
        cache_key,
        products,
        ttl=timedelta(minutes=5),
        namespace="products",
        tags=["products"]
    )

    return products

# When products change, invalidate all pages
await cache_service.invalidate_tags(["products"])
```

### Q: How do I debug cache issues?

A: Use cache statistics and logging:

```python
# Check if key exists
exists = await cache_service.exists("my_key")

# Get statistics
stats = await cache_service.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")

# Clear cache for testing
await cache_service.clear_all()

# Log cache operations
async def get_with_logging(key: str):
    result = await cache_service.get(key)
    if result:
        logger.info("cache_hit", key=key)
    else:
        logger.info("cache_miss", key=key)
    return result
```

### Q: Can I use the Cache Kit with synchronous code?

A: No, the Cache Kit is async-only. All methods use `async/await`. If you have synchronous code, you'll need to:

- Convert your code to async
- Use `asyncio.run()` to call async functions from sync code (not recommended in web apps)
- Use a different caching solution (like `functools.lru_cache` for pure in-memory sync caching)

### Q: How do I cache database query results with SQLAlchemy?

A: Use the cache-aside pattern:

```python
from sqlalchemy import select

async def get_user_by_email(email: str):
    # Check cache
    cache_key = f"user:email:{email}"
    cached = await cache_service.get(cache_key, namespace="users")
    if cached:
        return User(**cached)  # Reconstruct domain model

    # Query database
    async with database.get_session() as session:
        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user_model = result.scalar_one_or_none()

        if user_model:
            user = user_model.to_domain()

            # Cache the result (as dict for JSON serialization)
            await cache_service.set(
                cache_key,
                user.dict(),
                ttl=timedelta(minutes=10),
                namespace="users"
            )

            return user

        return None
```

### Q: What's the performance overhead of caching?

A: Depends on backend:

- **Memory**: ~0.1ms per operation (negligible)
- **Redis**: ~1-5ms per operation (network latency)
- **Hybrid**: 0.1ms (hit in memory) or 1-5ms (hit in Redis)

Always measure in your specific environment. For most applications, cache overhead is much less than database query time.

### Q: How do I handle cache versioning?

A: Include version in namespace or key:

```python
# Version in namespace
CACHE_VERSION = "v2"

await cache_service.set(
    "user_data",
    data,
    namespace=f"{CACHE_VERSION}:users"
)

# When changing data structure, increment version
# Old cache entries automatically expire
CACHE_VERSION = "v3"

# Or include version in key
await cache_service.set(
    f"v2:user:{user_id}",
    data
)
```
