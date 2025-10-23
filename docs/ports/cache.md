# Cache Port

## Overview

The Cache Port defines the contract for cache storage backends in Portico applications.

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

**Port Type**: Adapter

**When to Use**:

- Applications requiring fast data access and reduced database load
- Multi-tenant applications needing isolated cache namespaces
- Systems with complex cache invalidation patterns
- Applications requiring cache performance monitoring
- Services caching expensive computations or API calls

## Domain Models

### CacheKey

Represents a cache key with metadata for organization and invalidation. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `key` | `str` | Yes | - | The unique cache key identifier |
| `namespace` | `Optional[str]` | No | `None` | Optional namespace for key isolation |
| `tags` | `list[str]` | No | `[]` | Tags for group-based invalidation |

**Properties**:

- `full_key -> str` - Returns complete cache key with namespace prefix (`namespace:key` or just `key`)

**Class Methods**:

- `@classmethod from_function_call(func, args, kwargs, namespace, tags) -> CacheKey` - Creates cache key from function parameters by hashing function signature and arguments

**Example**:

```python
from portico.ports.cache import CacheKey

# Simple key
key = CacheKey(key="user:123")

# Key with namespace
key = CacheKey(
    key="profile",
    namespace="user:123",
    tags=["user", "profile"]
)
print(key.full_key)  # "user:123:profile"

# Key from function call (automatic)
key = CacheKey.from_function_call(
    func=get_user_profile,
    args=(user_id,),
    kwargs={},
    namespace="profiles",
    tags=["user"]
)
```

### CacheEntry

Represents a cached value with metadata for expiration and access tracking. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `key` | `CacheKey` | Yes | - | The cache key for this entry |
| `value` | `Any` | Yes | - | The cached value (can be any type) |
| `created_at` | `datetime` | No | Current UTC time | When entry was created |
| `expires_at` | `Optional[datetime]` | No | `None` | When entry expires (None = no expiration) |
| `hit_count` | `int` | No | `0` | Number of times entry has been accessed |
| `last_accessed` | `datetime` | No | Current UTC time | Last access timestamp |

**Properties**:

- `is_expired -> bool` - Returns True if entry has expired (current time > expires_at)

**Methods**:

- `touch() -> CacheEntry` - Returns new CacheEntry with updated last_accessed and incremented hit_count

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
    data = entry.value

# Update access metadata
entry = entry.touch()
```

### CacheStats

Performance statistics for cache monitoring and optimization. Immutable snapshot.

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

print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Total entries: {stats.total_entries}")
if stats.memory_usage_bytes:
    print(f"Memory: {stats.memory_usage_bytes / 1024 / 1024:.2f} MB")
```

## Port Interfaces

### CacheAdapter

The `CacheAdapter` abstract base class defines the contract for all cache storage backends.

**Location**: `portico.ports.cache.CacheAdapter`

#### Key Methods

##### get

```python
async def get(key: CacheKey) -> Optional[CacheEntry]
```

Retrieves a cached value by key. Primary method for cache reads.

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
    await cache_adapter.set(
        CacheKey(key="user:123"),
        user_data,
        ttl=timedelta(hours=1)
    )
```

##### set

```python
async def set(key: CacheKey, value: Any, ttl: Optional[timedelta] = None) -> None
```

Stores a value in the cache with optional time-to-live. Primary method for cache writes.

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

#### Other Methods

##### delete

```python
async def delete(key: CacheKey) -> bool
```

Deletes a single cached value. Returns True if key existed and was deleted.

##### delete_by_pattern

```python
async def delete_by_pattern(pattern: str) -> int
```

Deletes all keys matching pattern (supports wildcards like `"user:*"`). Returns number of keys deleted.

##### delete_by_namespace

```python
async def delete_by_namespace(namespace: str) -> int
```

Deletes all keys in a specific namespace. Returns number of keys deleted.

##### delete_by_tags

```python
async def delete_by_tags(tags: list[str]) -> int
```

Deletes all entries with any of the given tags. Returns number of entries deleted.

##### clear

```python
async def clear() -> None
```

Clears all cached entries (use with caution in production).

##### exists

```python
async def exists(key: CacheKey) -> bool
```

Checks if key exists in cache without retrieving the value.

##### get_stats

```python
async def get_stats() -> CacheStats
```

Retrieves cache performance statistics.

##### cleanup_expired

```python
async def cleanup_expired() -> int
```

Removes expired entries from the cache. Returns number of expired entries removed.

## Common Patterns

### Cache-Aside Pattern (Lazy Loading)

```python
from portico.ports.cache import CacheKey, CacheAdapter
from datetime import timedelta

async def get_user_profile(
    user_id: str,
    cache: CacheAdapter
) -> dict:
    """Get user profile with cache-aside pattern."""

    # 1. Try cache first
    cache_key = CacheKey(key=f"profile:{user_id}")
    entry = await cache.get(cache_key)

    if entry and not entry.is_expired:
        return entry.value

    # 2. Cache miss - fetch from database
    profile = await db.get_user_profile(user_id)

    # 3. Store in cache for next time
    await cache.set(
        key=cache_key,
        value=profile,
        ttl=timedelta(hours=1)
    )

    return profile
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

## Integration with Kits

The Cache Port is used by the **CacheKit** to provide high-level caching services.

```python
from portico import compose

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

# Access cache service
cache_service = app.kits["cache"].service

# Use cache service
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

The Cache Kit provides:

- Memory, Redis, and hybrid cache adapters
- Automatic serialization/deserialization
- Cache warming and preloading
- Performance monitoring and metrics

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Use Namespaces for Isolation**: Isolate cache keys by tenant/context for easy bulk operations

   ```python
   # ✅ GOOD: Isolated namespaces
   CacheKey(key="profile", namespace=f"user:{user_id}")
   CacheKey(key="settings", namespace=f"tenant:{tenant_id}")

   # ❌ BAD: No isolation
   CacheKey(key=f"user_{user_id}_profile")
   ```

2. **Tag Related Data**: Use tags to enable bulk invalidation of related cache entries

   ```python
   # ✅ GOOD: Tags enable bulk invalidation
   CacheKey(
       key=f"product:{product_id}",
       tags=["product", f"category:{category_id}", f"vendor:{vendor_id}"]
   )

   # Invalidate all products in category
   await cache.delete_by_tags([f"category:{category_id}"])
   ```

3. **Set Appropriate TTLs**: Different data types need different expiration times

   ```python
   # ✅ GOOD: Different TTLs for different data
   await cache.set(session_key, session, ttl=timedelta(hours=1))    # Sessions
   await cache.set(config_key, config, ttl=timedelta(days=1))       # Config
   await cache.set(temp_key, temp, ttl=timedelta(minutes=5))        # Temporary

   # ❌ BAD: No TTL for temporary data
   await cache.set(temp_key, temp)  # Never expires!
   ```

4. **Handle Cache Misses Gracefully**: Always have a fallback to the source of truth

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

5. **Monitor Cache Performance**: Regular monitoring helps optimize hit rates

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

await cache.delete_by_namespace("catalog")            # Deletes this
await cache.delete_by_tags(["product"])              # Deletes this
await cache.delete_by_tags(["category:electronics"]) # Deletes this
```

### What happens if I don't specify a TTL?

Entries without TTL never expire automatically. You must manually delete them or use `cache.clear()`. This is appropriate for:
- Configuration data that rarely changes
- Reference data (e.g., country codes)
- Data with manual invalidation logic

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

### How do I prevent cache stampede?

Cache stampede occurs when many requests simultaneously miss cache and hit the database. Use locking:

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
