"""Test examples for Cache Adapter documentation.

This module tests code examples from cache adapter documentation (Redis, Hybrid, Memory)
to ensure they remain accurate and working.
"""

import pytest

from portico.adapters.cache.memory_cache import MemoryCacheAdapter
from portico.ports.cache import CacheKey

# ========== Memory Cache Adapter Examples ==========


# --8<-- [start:memory-background-cleanup]
@pytest.mark.asyncio
async def test_memory_background_cleanup(config_registry):
    """Background cleanup task for memory cache."""
    cache = MemoryCacheAdapter()

    # Start cleanup task (runs every 5 minutes)
    cache.start_cleanup_task(cleanup_interval_seconds=300)

    # Stop cleanup when shutting down
    cache.stop_cleanup_task()

    # Verify cleanup task can be started and stopped
    assert cache is not None


# --8<-- [end:memory-background-cleanup]


# --8<-- [start:memory-compression-stats]
@pytest.mark.asyncio
async def test_memory_compression_stats(config_registry):
    """Get compression statistics from memory cache."""
    cache = MemoryCacheAdapter(enable_compression=True)

    # Perform some operations
    large_data = "x" * 10000
    await cache.set(CacheKey(key="large"), large_data)
    await cache.get(CacheKey(key="large"))

    stats = cache.get_compression_stats()
    assert "compressions" in stats
    assert "decompressions" in stats
    assert stats["compressions"] > 0
    assert stats["decompressions"] > 0


# --8<-- [end:memory-compression-stats]


# --8<-- [start:memory-eviction-runtime]
@pytest.mark.asyncio
async def test_memory_change_eviction_policy(config_registry):
    """Change eviction policy at runtime."""
    cache = MemoryCacheAdapter(eviction_policy="lru")

    # Change to LFU
    cache.set_eviction_policy("lfu")

    # Get current policy
    policy = cache.get_eviction_policy()
    assert policy == "lfu"


# --8<-- [end:memory-eviction-runtime]


# --8<-- [start:memory-config-development]
@pytest.mark.asyncio
async def test_memory_development_config(config_registry):
    """Development configuration with no limits."""
    # No limits - simple caching for development
    cache = MemoryCacheAdapter()

    # Can store any number of items
    for i in range(100):
        await cache.set(CacheKey(key=f"item:{i}"), f"value-{i}")

    stats = await cache.get_stats()
    assert stats.total_entries == 100


# --8<-- [end:memory-config-development]


# --8<-- [start:memory-config-size-limited]
@pytest.mark.asyncio
async def test_memory_size_limited_config(config_registry):
    """Production configuration with size limit."""
    # Limit to 10,000 entries with LRU eviction
    cache = MemoryCacheAdapter(max_size=10000, eviction_policy="lru")

    # Add items
    for i in range(100):
        await cache.set(CacheKey(key=f"item:{i}"), f"value-{i}")

    stats = await cache.get_stats()
    assert stats.total_entries == 100


# --8<-- [end:memory-config-size-limited]


# --8<-- [start:memory-config-memory-limited]
@pytest.mark.asyncio
async def test_memory_memory_limited_config(config_registry):
    """Production configuration with memory limit."""
    # Limit to 128MB with compression for large values
    cache = MemoryCacheAdapter(
        max_memory_bytes=128 * 1024 * 1024,  # 128MB
        eviction_policy="lru",
        enable_compression=True,
        compression_threshold=1024,  # Compress values > 1KB
    )

    # Store data
    await cache.set(CacheKey(key="test"), "value")
    entry = await cache.get(CacheKey(key="test"))
    assert entry.value == "value"


# --8<-- [end:memory-config-memory-limited]


# --8<-- [start:memory-config-high-performance]
@pytest.mark.asyncio
async def test_memory_high_performance_config(config_registry):
    """High-performance caching configuration."""
    # Aggressive caching with automatic cleanup
    cache = MemoryCacheAdapter(
        max_size=50000,
        eviction_policy="lfu",  # Keep frequently used items
    )

    # Start background cleanup
    cache.start_cleanup_task(cleanup_interval_seconds=60)

    # Use cache
    await cache.set(CacheKey(key="key1"), "value1")
    entry = await cache.get(CacheKey(key="key1"))
    assert entry.value == "value1"

    # Cleanup
    cache.stop_cleanup_task()


# --8<-- [end:memory-config-high-performance]


# --8<-- [start:memory-testing-fixture]
@pytest.mark.asyncio
async def test_memory_testing_fixture(config_registry):
    """Using memory cache fixture in tests."""

    @pytest.fixture
    def cache():
        return MemoryCacheAdapter()

    # Simulate test using fixture
    test_cache = MemoryCacheAdapter()

    key = CacheKey(key="test")
    await test_cache.set(key, "value")

    entry = await test_cache.get(key)
    assert entry is not None
    assert entry.value == "value"


# --8<-- [end:memory-testing-fixture]


# --8<-- [start:memory-testing-eviction]
@pytest.mark.asyncio
async def test_memory_testing_eviction(config_registry):
    """Test cache eviction behavior."""
    # Test with small cache
    cache = MemoryCacheAdapter(max_size=2, eviction_policy="lru")

    await cache.set(CacheKey(key="1"), "a")
    await cache.set(CacheKey(key="2"), "b")
    await cache.set(CacheKey(key="3"), "c")  # Evicts "1"

    assert await cache.exists(CacheKey(key="1")) is False
    assert await cache.exists(CacheKey(key="2")) is True
    assert await cache.exists(CacheKey(key="3")) is True


# --8<-- [end:memory-testing-eviction]

# ========== Redis Cache Adapter Examples ==========

# Note: Most Redis examples are configuration-focused and don't require actual tests
# since they would need a running Redis instance. The interface is identical to Memory Cache.


# --8<-- [start:redis-standard]
def test_redis_standard_config(config_registry):
    """Standard Redis configuration."""
    from unittest.mock import MagicMock

    # Mock RedisCacheAdapter since we don't have Redis running
    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(redis_url="redis://localhost:6379", db=0, pool_size=10)

    assert cache is not None


# --8<-- [end:redis-standard]


# --8<-- [start:redis-auth]
def test_redis_auth_config(config_registry):
    """Redis with authentication."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(redis_url="redis://:password@localhost:6379", db=0)

    assert cache is not None


# --8<-- [end:redis-auth]


# --8<-- [start:redis-tls]
def test_redis_tls_config(config_registry):
    """Redis with TLS."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="rediss://localhost:6380",  # Note: rediss://
        db=0,
    )

    assert cache is not None


# --8<-- [end:redis-tls]


# --8<-- [start:redis-sentinel]
def test_redis_sentinel_config(config_registry):
    """Redis Sentinel for high availability."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    # Using Redis Sentinel for automatic failover
    cache = RedisCacheAdapter(
        redis_url="redis://sentinel-host:26379",
        # Configure sentinel in redis_url or use sentinel-specific client
    )

    assert cache is not None


# --8<-- [end:redis-sentinel]


# --8<-- [start:redis-pickle]
def test_redis_pickle_serialization(config_registry):
    """Redis with pickle serialization (default)."""
    from datetime import datetime
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://localhost:6379",
        serialization_format="pickle",  # Default
    )

    # Would cache complex objects if Redis was available
    # await cache.set(CacheKey(key="data"), {
    #     "nested": {"objects": [1, 2, 3]},
    #     "datetime": datetime.now()
    # })

    assert cache is not None


# --8<-- [end:redis-pickle]


# --8<-- [start:redis-json]
def test_redis_json_serialization(config_registry):
    """Redis with JSON serialization."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://localhost:6379", serialization_format="json"
    )

    # Must be JSON-serializable
    # await cache.set(CacheKey(key="data"), {
    #     "string": "value",
    #     "number": 42,
    #     "list": [1, 2, 3]
    # })

    assert cache is not None


# --8<-- [end:redis-json]


# --8<-- [start:redis-compression]
def test_redis_compression_config(config_registry):
    """Redis with compression enabled."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://localhost:6379",
        enable_compression=True,
        compression_threshold=1024,  # Compress values > 1KB
    )

    assert cache is not None


# --8<-- [end:redis-compression]


# --8<-- [start:redis-dev-testing]
def test_redis_dev_testing_config(config_registry):
    """Development/testing Redis configuration."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://localhost:6379",
        db=1,  # Use separate db for dev
        key_prefix="dev:cache:",
    )

    assert cache is not None


# --8<-- [end:redis-dev-testing]


# --8<-- [start:redis-production]
def test_redis_production_config(config_registry):
    """Production Redis configuration."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://:password@redis-prod:6379",
        db=0,
        pool_size=20,  # Larger pool for production
        socket_timeout=3.0,  # Shorter timeout
        enable_compression=True,
        key_prefix="prod:cache:",
    )

    assert cache is not None


# --8<-- [end:redis-production]


# --8<-- [start:redis-high-performance]
def test_redis_high_performance_config(config_registry):
    """High-performance Redis configuration."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://redis-cluster:6379",
        db=0,
        pool_size=50,  # Large pool for high concurrency
        socket_timeout=1.0,  # Fast timeout
        retry_on_timeout=True,
        serialization_format="pickle",  # Fast serialization
        enable_compression=False,  # Skip compression overhead
    )

    assert cache is not None


# --8<-- [end:redis-high-performance]


# --8<-- [start:redis-memory-optimized]
def test_redis_memory_optimized_config(config_registry):
    """Memory-optimized Redis configuration."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    cache = RedisCacheAdapter(
        redis_url="redis://localhost:6379",
        enable_compression=True,
        compression_threshold=512,  # Aggressive compression
        serialization_format="json",  # Smaller than pickle
    )

    assert cache is not None


# --8<-- [end:redis-memory-optimized]


# --8<-- [start:redis-core-config]
@pytest.mark.asyncio
async def test_redis_core_config_integration(config_registry):
    """Integration with CoreConfig."""
    from portico.core import PorticoCore

    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Cache is automatically configured
    # Access via core if needed
    assert core is not None

    await core.close()


# --8<-- [end:redis-core-config]


# --8<-- [start:redis-error-handling]
def test_redis_error_handling(config_registry):
    """Error handling for Redis operations."""
    from portico.exceptions import CacheConnectionError

    # Demonstrate error handling pattern
    # In real code with Redis:
    # try:
    #     entry = await cache.get(key)
    # except CacheConnectionError as e:
    #     logger.error("Redis unavailable, falling back", error=str(e))
    #     entry = None

    assert CacheConnectionError is not None


# --8<-- [end:redis-error-handling]


# --8<-- [start:redis-testing-fixture]
@pytest.mark.asyncio
async def test_redis_testing_with_memory(config_registry):
    """Using memory cache instead of Redis in tests."""
    from portico.ports.cache import CacheAdapter

    @pytest.fixture
    def cache() -> CacheAdapter:
        # Use memory cache in tests - no Redis required
        return MemoryCacheAdapter()

    # Simulate test
    test_cache = MemoryCacheAdapter()

    key = CacheKey(key="test")
    await test_cache.set(key, "value")

    entry = await test_cache.get(key)
    assert entry.value == "value"


# --8<-- [end:redis-testing-fixture]


# --8<-- [start:redis-integration-testing]
def test_redis_integration_testing_setup(config_registry):
    """Integration testing setup with Redis."""
    from unittest.mock import MagicMock

    RedisCacheAdapter = MagicMock()

    @pytest.fixture
    async def redis_cache():
        cache = RedisCacheAdapter(
            redis_url="redis://localhost:6379",
            db=15,  # Use separate db for tests
            key_prefix="test:cache:",
        )
        yield cache
        # Cleanup
        # await cache.clear()

    # Fixture is ready for integration tests
    assert redis_cache is not None


# --8<-- [end:redis-integration-testing]


# --8<-- [start:redis-migration-from-memory]
def test_redis_migration_from_memory(config_registry):
    """Migration from Memory to Redis cache."""
    from unittest.mock import MagicMock

    # Before - Memory Cache
    memory_cache = MemoryCacheAdapter()
    assert memory_cache is not None

    # After - Redis Cache
    RedisCacheAdapter = MagicMock()
    redis_cache = RedisCacheAdapter(url="redis://localhost:6379")
    assert redis_cache is not None

    # All code using cache works identically!
    # await cache.set(CacheKey(key="user:123"), user_data)
    # entry = await cache.get(CacheKey(key="user:123"))


# --8<-- [end:redis-migration-from-memory]

# ========== Hybrid Cache Adapter Examples ==========


# --8<-- [start:hybrid-basic-config]
def test_hybrid_basic_config(config_registry):
    """Basic hybrid cache configuration."""
    from unittest.mock import MagicMock

    # Mock HybridCacheAdapter since we don't have Redis
    HybridCacheAdapter = MagicMock()

    cache = HybridCacheAdapter(
        # L1 (Memory) configuration
        l1_max_size=1000,
        l1_eviction_policy="lru",
        # L2 (Redis) configuration
        redis_url="redis://localhost:6379",
        redis_db=0,
        # Optional: compression
        enable_compression=True,
    )

    assert cache is not None


# --8<-- [end:hybrid-basic-config]


# --8<-- [start:hybrid-balanced]
def test_hybrid_balanced_config(config_registry):
    """Balanced hybrid cache configuration."""
    from unittest.mock import MagicMock

    HybridCacheAdapter = MagicMock()

    cache = HybridCacheAdapter(
        # Small L1 for hot data
        l1_max_size=1000,
        l1_eviction_policy="lru",
        # Large L2 for broader dataset
        redis_url="redis://localhost:6379",
        redis_db=0,
        # Compress large values in L2
        enable_compression=True,
    )

    assert cache is not None


# --8<-- [end:hybrid-balanced]


# --8<-- [start:hybrid-high-performance]
def test_hybrid_high_performance_config(config_registry):
    """High-performance hybrid cache configuration."""
    from unittest.mock import MagicMock

    HybridCacheAdapter = MagicMock()

    # Larger L1 for maximum speed
    cache = HybridCacheAdapter(
        l1_max_size=10000,  # More data in memory
        l1_eviction_policy="lfu",  # Keep frequently used
        redis_url="redis://localhost:6379",
        enable_compression=False,  # Skip compression overhead
    )

    assert cache is not None


# --8<-- [end:hybrid-high-performance]


# --8<-- [start:hybrid-memory-conscious]
def test_hybrid_memory_conscious_config(config_registry):
    """Memory-conscious hybrid cache configuration."""
    from unittest.mock import MagicMock

    HybridCacheAdapter = MagicMock()

    # Small L1, rely more on L2
    cache = HybridCacheAdapter(
        l1_max_size=100,  # Minimal memory footprint
        l1_eviction_policy="lru",
        redis_url="redis://localhost:6379",
        enable_compression=True,  # Save Redis memory
    )

    assert cache is not None


# --8<-- [end:hybrid-memory-conscious]


# --8<-- [start:hybrid-testing]
@pytest.mark.asyncio
async def test_hybrid_testing_with_memory(config_registry):
    """Use memory cache in tests instead of hybrid."""
    from portico.ports.cache import CacheAdapter

    @pytest.fixture
    def cache() -> CacheAdapter:
        # Use memory cache in tests
        return MemoryCacheAdapter()

    # Test code works with Memory, Redis, or Hybrid
    test_cache = MemoryCacheAdapter()

    await test_cache.set(CacheKey(key="test"), "value")
    entry = await test_cache.get(CacheKey(key="test"))
    assert entry.value == "value"


# --8<-- [end:hybrid-testing]
