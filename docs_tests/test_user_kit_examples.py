"""Test examples for user management kit documentation.

All examples are extracted into docs using snippet markers.
"""

import pytest

from portico.core import PorticoCore
from portico.ports.user import CreateUserRequest, UpdateUserRequest


@pytest.fixture
async def test_core(config_registry):
    """Create test PorticoCore instance."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    yield core
    await core.close()


# --8<-- [start:create-user]
@pytest.mark.asyncio
async def test_create_user(test_core):
    """Create a new user."""
    user = await test_core.create_user(
        CreateUserRequest(
            email="alice@example.com",
            username="alice",
            password="SecurePassword123!",
        )
    )

    assert user is not None
    assert user.email == "alice@example.com"
    assert user.username == "alice"
    assert user.is_active is True


# --8<-- [end:create-user]


# --8<-- [start:get-user]
@pytest.mark.asyncio
async def test_get_user(test_core):
    """Get user by ID."""
    # Create user
    created = await test_core.create_user(
        CreateUserRequest(
            email="bob@example.com",
            username="bob",
            password="SecurePassword123!",
        )
    )

    # Get user
    user = await test_core.get_user(created.id)

    assert user is not None
    assert user.email == "bob@example.com"


# --8<-- [end:get-user]


# --8<-- [start:update-user]
@pytest.mark.asyncio
async def test_update_user(test_core):
    """Update user information."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="charlie@example.com",
            username="charlie",
            password="SecurePassword123!",
        )
    )

    # Update email
    updated = await test_core.update_user(
        user.id,
        UpdateUserRequest(email="charlie.new@example.com"),
    )

    assert updated is not None
    assert updated.email == "charlie.new@example.com"


# --8<-- [end:update-user]


# --8<-- [start:delete-user]
@pytest.mark.asyncio
async def test_delete_user(test_core):
    """Delete a user."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="dave@example.com",
            username="dave",
            password="SecurePassword123!",
        )
    )

    # Delete user
    deleted = await test_core.delete_user(user.id)
    assert deleted is True

    # Verify deleted
    user_after = await test_core.get_user(user.id)
    assert user_after is None


# --8<-- [end:delete-user]


# --8<-- [start:list-users]
@pytest.mark.asyncio
async def test_list_users(test_core):
    """List users with pagination."""
    # Create users
    for i in range(5):
        await test_core.create_user(
            CreateUserRequest(
                email=f"user{i}@example.com",
                username=f"user{i}",
                password="SecurePassword123!",
            )
        )

    # List with pagination
    users = await test_core.list_users(limit=3, offset=0)

    assert len(users) == 3


# --8<-- [end:list-users]


# --8<-- [start:get-all-users]
@pytest.mark.asyncio
async def test_get_all_users(test_core):
    """Get all users (using list_users with high limit)."""
    # Create users
    await test_core.create_user(
        CreateUserRequest(
            email="alpha@example.com",
            username="alpha",
            password="SecurePassword123!",
        )
    )
    await test_core.create_user(
        CreateUserRequest(
            email="beta@example.com",
            username="beta",
            password="SecurePassword123!",
        )
    )

    # Get all users (no pagination)
    users = await test_core.list_users(limit=1000, offset=0)

    assert len(users) >= 2
    emails = [u.email for u in users]
    assert "alpha@example.com" in emails
    assert "beta@example.com" in emails


# --8<-- [end:get-all-users]


# --8<-- [start:create-admin]
@pytest.mark.asyncio
async def test_create_admin(test_core):
    """Create an admin user."""
    admin = await test_core.create_user(
        CreateUserRequest(
            email="admin@example.com",
            username="admin",
            password="SecurePassword123!",
            global_role="admin",
        )
    )

    assert admin.global_role == "admin"


# --8<-- [end:create-admin]


# --8<-- [start:deactivate-user]
@pytest.mark.asyncio
async def test_deactivate_user(test_core):
    """Deactivate a user account."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="eve@example.com",
            username="eve",
            password="SecurePassword123!",
        )
    )

    # Deactivate
    updated = await test_core.update_user(user.id, UpdateUserRequest(is_active=False))

    assert updated is not None
    assert updated.is_active is False


# --8<-- [end:deactivate-user]
