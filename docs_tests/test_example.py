"""Example test file showing the pattern for documentation tests.

This demonstrates how to write tests that verify code examples in documentation.
Uses pytest snippet markers to include code in docs via mkdocs-snippets.

Usage in documentation markdown:
    ```python
    --8<-- "docs_tests/test_example.py:basic-usage"
    ```

This will include the code between the start and end markers.
"""

import pytest

from portico.adapters.cache import MemoryCacheAdapter
from portico.ports.cache import CacheKey


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_cache_usage():
    """Basic cache operations - example for documentation."""
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


@pytest.mark.asyncio
async def test_another_example():
    """Another example test without snippet markers.

    This test won't be included in documentation,
    but still validates that the code works.
    """
    cache = MemoryCacheAdapter()
    key = CacheKey(key="test")

    await cache.set(key, "value")
    entry = await cache.get(key)

    assert entry is not None
    assert entry.value == "value"
