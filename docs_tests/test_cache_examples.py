"""Test examples for Cache port documentation."""

from datetime import timedelta

import pytest

from portico.adapters.cache import MemoryCacheAdapter
from portico.ports.cache import CacheKey, CacheStats


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_cache_usage():
    """Basic cache operations."""
    cache = MemoryCacheAdapter()

    # Create a cache key
    key = CacheKey(key="user:123", namespace="users")

    # Store a value
    await cache.set(key, {"name": "Alice", "email": "alice@example.com"})

    # Retrieve the value
    entry = await cache.get(key)
    assert entry is not None
    assert entry.value["name"] == "Alice"


# --8<-- [end:basic-usage]


# --8<-- [start:ttl-usage]
@pytest.mark.asyncio
async def test_cache_with_ttl():
    """Cache entries with time-to-live."""
    cache = MemoryCacheAdapter()
    key = CacheKey(key="session:abc123")

    # Store value with 5 minute TTL
    await cache.set(
        key, {"user_id": "123", "expires": "2025-01-01"}, ttl=timedelta(minutes=5)
    )

    # Value is available
    entry = await cache.get(key)
    assert entry is not None
    assert entry.value["user_id"] == "123"


# --8<-- [end:ttl-usage]


# --8<-- [start:namespace-operations]
@pytest.mark.asyncio
async def test_namespace_operations():
    """Organize cache entries by namespace."""
    cache = MemoryCacheAdapter()

    # Store entries in different namespaces
    await cache.set(CacheKey(key="1", namespace="users"), {"name": "Alice"})
    await cache.set(CacheKey(key="2", namespace="users"), {"name": "Bob"})
    await cache.set(CacheKey(key="1", namespace="posts"), {"title": "Hello"})

    # Delete all entries in "users" namespace
    deleted = await cache.delete_by_namespace("users")
    assert deleted == 2

    # "posts" namespace unaffected
    entry = await cache.get(CacheKey(key="1", namespace="posts"))
    assert entry is not None
    assert entry.value["title"] == "Hello"


# --8<-- [end:namespace-operations]


# --8<-- [start:tag-invalidation]
@pytest.mark.asyncio
async def test_tag_based_invalidation():
    """Invalidate related cache entries using tags."""
    cache = MemoryCacheAdapter()

    # Store entries with tags
    await cache.set(
        CacheKey(key="user:123", tags=["user", "profile"]), {"name": "Alice"}
    )
    await cache.set(
        CacheKey(key="posts:123", tags=["user", "content"]), {"author": "Alice"}
    )
    await cache.set(CacheKey(key="settings", tags=["config"]), {"theme": "dark"})

    # Invalidate all entries tagged with "user"
    deleted = await cache.delete_by_tags(["user"])
    assert deleted == 2

    # Config entry still exists
    entry = await cache.get(CacheKey(key="settings", tags=["config"]))
    assert entry is not None


# --8<-- [end:tag-invalidation]


# --8<-- [start:eviction-policies]
@pytest.mark.asyncio
async def test_eviction_policies():
    """Configure cache eviction behavior."""
    # LRU eviction with max 100 entries
    cache = MemoryCacheAdapter(
        max_size=100,
        eviction_policy="lru",  # Least Recently Used
    )

    # Add entries
    for i in range(150):
        await cache.set(CacheKey(key=f"item:{i}"), f"value-{i}")

    # Only 100 entries remain (oldest evicted)
    stats = await cache.get_stats()
    assert stats.total_entries == 100


# --8<-- [end:eviction-policies]


# --8<-- [start:cache-stats]
@pytest.mark.asyncio
async def test_cache_statistics():
    """Monitor cache performance."""
    cache = MemoryCacheAdapter()

    # Perform operations
    key = CacheKey(key="test")
    await cache.set(key, "value")

    await cache.get(key)  # Hit
    await cache.get(CacheKey(key="missing"))  # Miss

    # Get statistics
    stats = await cache.get_stats()
    assert stats.total_entries == 1
    assert stats.total_hits == 1
    assert stats.total_misses == 1
    assert stats.hit_rate == 0.5  # 50% hit rate


# --8<-- [end:cache-stats]


# --8<-- [start:pattern-deletion]
@pytest.mark.asyncio
async def test_pattern_based_deletion():
    """Delete cache entries matching a pattern."""
    cache = MemoryCacheAdapter()

    # Store various keys
    await cache.set(CacheKey(key="user:123"), "Alice")
    await cache.set(CacheKey(key="user:456"), "Bob")
    await cache.set(CacheKey(key="post:789"), "Hello")

    # Delete all "user:" keys using regex
    deleted = await cache.delete_by_pattern(r"user:\d+")
    assert deleted == 2

    # Post key remains
    entry = await cache.get(CacheKey(key="post:789"))
    assert entry is not None


# --8<-- [end:pattern-deletion]


# --8<-- [start:compression]
@pytest.mark.asyncio
async def test_value_compression():
    """Automatically compress large cached values."""
    cache = MemoryCacheAdapter(
        enable_compression=True,
        compression_threshold=1024,  # Compress values > 1KB
    )

    # Large value gets compressed
    large_data = "x" * 10000
    await cache.set(CacheKey(key="large"), large_data)

    # Transparently decompressed on retrieval
    entry = await cache.get(CacheKey(key="large"))
    assert entry.value == large_data

    # Check compression stats
    comp_stats = cache.get_compression_stats()
    assert comp_stats["compressions"] > 0


# --8<-- [end:compression]


# --8<-- [start:exists-check]
@pytest.mark.asyncio
async def test_key_existence():
    """Check if a key exists without retrieving the value."""
    cache = MemoryCacheAdapter()

    key = CacheKey(key="test")

    # Key doesn't exist yet
    exists = await cache.exists(key)
    assert not exists

    # Store value
    await cache.set(key, "value")

    # Now it exists
    exists = await cache.exists(key)
    assert exists


# --8<-- [end:exists-check]


# --8<-- [start:cleanup-expired]
@pytest.mark.asyncio
async def test_cleanup_expired_entries():
    """Manually cleanup expired cache entries."""
    cache = MemoryCacheAdapter()

    # Store entry with very short TTL
    await cache.set(CacheKey(key="temp"), "value", ttl=timedelta(microseconds=1))

    # Wait for expiration
    import asyncio

    await asyncio.sleep(0.001)

    # Cleanup expired entries
    removed = await cache.cleanup_expired()
    assert removed == 1


# --8<-- [end:cleanup-expired]

# ========== Cache Service (Kit) Examples ==========

from portico.kits.cache import CacheService


# --8<-- [start:cache-service-basic]
@pytest.mark.asyncio
async def test_cache_service_basic():
    """Using CacheService for basic operations."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(
        cache_adapter=cache_adapter,
        default_ttl=timedelta(hours=1),
        default_namespace="app",
    )

    # Simple set
    await cache_service.set("user:123", {"name": "Alice"})

    # Simple get
    user = await cache_service.get("user:123")
    assert user == {"name": "Alice"}

    # Delete
    deleted = await cache_service.delete("user:123")
    assert deleted is True

    # Get after delete returns None
    user = await cache_service.get("user:123")
    assert user is None


# --8<-- [end:cache-service-basic]


# --8<-- [start:cache-service-get-or-set]
@pytest.mark.asyncio
async def test_cache_service_get_or_set():
    """Get or set pattern with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(cache_adapter=cache_adapter)

    call_count = 0

    async def fetch_user(user_id):
        nonlocal call_count
        call_count += 1
        return {"id": user_id, "name": f"User{user_id}"}

    # First call: computes and caches
    user1 = await cache_service.get_or_set_async(
        key="user:123",
        value_factory=lambda: fetch_user(123),
        ttl=timedelta(minutes=30),
    )
    assert user1 == {"id": 123, "name": "User123"}
    assert call_count == 1

    # Second call: returns cached value
    user2 = await cache_service.get_or_set_async(
        key="user:123",
        value_factory=lambda: fetch_user(123),
        ttl=timedelta(minutes=30),
    )
    assert user2 == {"id": 123, "name": "User123"}
    assert call_count == 1  # Not called again


# --8<-- [end:cache-service-get-or-set]


# --8<-- [start:cache-service-namespace-invalidation]
@pytest.mark.asyncio
async def test_cache_service_namespace_invalidation():
    """Invalidate entire namespace with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(
        cache_adapter=cache_adapter,
        default_namespace="users",
    )

    # Set multiple items
    await cache_service.set("user:1", {"name": "Alice"})
    await cache_service.set("user:2", {"name": "Bob"})
    await cache_service.set("user:3", {"name": "Charlie"})

    # Invalidate namespace
    deleted = await cache_service.invalidate_namespace("users")
    assert deleted == 3

    # All items gone
    assert await cache_service.get("user:1") is None
    assert await cache_service.get("user:2") is None


# --8<-- [end:cache-service-namespace-invalidation]


# --8<-- [start:cache-service-tag-invalidation]
@pytest.mark.asyncio
async def test_cache_service_tag_invalidation():
    """Invalidate by tags with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(cache_adapter=cache_adapter)

    # Set items with tags
    await cache_service.set(
        "user:1:profile", {"name": "Alice"}, tags=["user:1", "profile"]
    )
    await cache_service.set(
        "user:1:settings", {"theme": "dark"}, tags=["user:1", "settings"]
    )
    await cache_service.set(
        "user:2:profile", {"name": "Bob"}, tags=["user:2", "profile"]
    )

    # Invalidate all items tagged "user:1"
    deleted = await cache_service.invalidate_tags(["user:1"])
    assert deleted == 2

    # User 1 items gone
    assert await cache_service.get("user:1:profile") is None
    assert await cache_service.get("user:1:settings") is None

    # User 2 still there
    assert await cache_service.get("user:2:profile") == {"name": "Bob"}


# --8<-- [end:cache-service-tag-invalidation]


# --8<-- [start:cache-service-default-ttl]
@pytest.mark.asyncio
async def test_cache_service_default_ttl():
    """CacheService with default TTL."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(
        cache_adapter=cache_adapter,
        default_ttl=timedelta(hours=2),
    )

    # Set without specifying TTL uses default
    await cache_service.set("key1", "value1")

    # Entry exists
    value = await cache_service.get("key1")
    assert value == "value1"

    # Can override default TTL
    await cache_service.set("key2", "value2", ttl=timedelta(minutes=5))
    value = await cache_service.get("key2")
    assert value == "value2"


# --8<-- [end:cache-service-default-ttl]


# --8<-- [start:cache-service-stats]
@pytest.mark.asyncio
async def test_cache_service_stats():
    """Get cache statistics via CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(cache_adapter=cache_adapter)

    # Perform some operations
    await cache_service.set("key1", "value1")
    await cache_service.get("key1")  # Hit
    await cache_service.get("key2")  # Miss

    # Get statistics
    stats = await cache_service.get_stats()

    assert isinstance(stats, CacheStats)
    assert stats.total_hits >= 1
    assert stats.total_misses >= 1
    assert stats.total_entries >= 1


# --8<-- [end:cache-service-stats]


# --8<-- [start:cache-service-exists]
@pytest.mark.asyncio
async def test_cache_service_exists():
    """Check key existence with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(cache_adapter=cache_adapter)

    # Key doesn't exist
    exists = await cache_service.exists("nonexistent")
    assert exists is False

    # Set key
    await cache_service.set("mykey", "myvalue")

    # Now it exists
    exists = await cache_service.exists("mykey")
    assert exists is True


# --8<-- [end:cache-service-exists]


# --8<-- [start:cache-service-string-keys]
@pytest.mark.asyncio
async def test_cache_service_string_keys():
    """CacheService accepts string keys."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(
        cache_adapter=cache_adapter,
        default_namespace="app",
    )

    # Use simple strings as keys
    await cache_service.set("session:abc123", {"user_id": "user_1"})
    await cache_service.set("counter:page_views", 42)

    session = await cache_service.get("session:abc123")
    assert session["user_id"] == "user_1"

    counter = await cache_service.get("counter:page_views")
    assert counter == 42


# --8<-- [end:cache-service-string-keys]


# --8<-- [start:cache-service-namespace-usage]
@pytest.mark.asyncio
async def test_cache_service_namespace_usage():
    """Using namespaces with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(
        cache_adapter=cache_adapter,
        default_namespace="prod",
    )

    # Use default namespace
    await cache_service.set("config", {"version": "1.0"})

    # Override namespace per operation
    await cache_service.set("config", {"version": "2.0"}, namespace="staging")

    # Different namespaces have different values
    prod_config = await cache_service.get("config", namespace="prod")
    staging_config = await cache_service.get("config", namespace="staging")

    assert prod_config["version"] == "1.0"
    assert staging_config["version"] == "2.0"


# --8<-- [end:cache-service-namespace-usage]


# --8<-- [start:cache-service-complex-objects]
@pytest.mark.asyncio
async def test_cache_service_complex_objects():
    """Caching complex objects with CacheService."""
    cache_adapter = MemoryCacheAdapter()
    cache_service = CacheService(cache_adapter=cache_adapter)

    # Cache complex nested structures
    user_data = {
        "id": "user_123",
        "name": "Alice",
        "profile": {
            "age": 30,
            "location": "San Francisco",
        },
        "tags": ["premium", "early-adopter"],
        "metadata": {
            "last_login": "2025-01-01",
            "preferences": {
                "theme": "dark",
                "notifications": True,
            },
        },
    }

    await cache_service.set("user:123:full", user_data)

    # Retrieved object matches
    retrieved = await cache_service.get("user:123:full")
    assert retrieved == user_data
    assert retrieved["profile"]["location"] == "San Francisco"
    assert "premium" in retrieved["tags"]


# --8<-- [end:cache-service-complex-objects]
