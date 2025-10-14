"""Pytest configuration for documentation tests.

These tests verify that all code examples in the documentation actually work.
"""

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ConfigRegistry fixture removed - deprecated functionality
# Use compose.webapp() for application composition instead
