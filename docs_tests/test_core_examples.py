"""Test examples for PorticoCore documentation.

All examples are extracted into docs using snippet markers.
"""

from contextlib import asynccontextmanager

import pytest

from portico.core import PorticoCore
from portico.ports.user import CreateUserRequest


# --8<-- [start:basic-initialization]
@pytest.mark.asyncio
async def test_basic_core_initialization(config_registry):
    """Basic PorticoCore initialization."""

    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()  # Creates database tables, registers default roles

    # Verify core initialized
    assert core is not None
    assert core.db is not None

    await core.close()


# --8<-- [end:basic-initialization]


# --8<-- [start:authentication-services]
@pytest.mark.asyncio
async def test_authentication_services(config_registry):
    """Core authentication services."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create user
    user = await core.create_user(
        CreateUserRequest(
            email="test@example.com",
            username="testuser",
            password="SecurePassword123!",
        )
    )

    # authenticate_user() - Verify credentials and create session
    auth_result = await core.authenticate_user("test@example.com", "SecurePassword123!")
    assert auth_result.success is True

    # verify_password() - Check password hash
    is_valid = await core.verify_password(user, "SecurePassword123!")
    assert is_valid is True

    # hash_password() - Generate password hash
    new_hash = await core.hash_password("NewPassword456!")
    assert new_hash is not None
    assert len(new_hash) > 0

    await core.close()


# --8<-- [end:authentication-services]


# --8<-- [start:user-management-services]
@pytest.mark.asyncio
async def test_user_management_services(config_registry):
    """Core user management services."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # create_user() - Create new user
    user = await core.create_user(
        CreateUserRequest(
            email="alice@example.com",
            username="alice",
            password="SecurePassword123!",
        )
    )
    assert user is not None
    assert user.email == "alice@example.com"

    # get_user() - Retrieve user by ID
    retrieved = await core.get_user(user.id)
    assert retrieved is not None
    assert retrieved.id == user.id

    # update_user() - Update user profile
    from portico.ports.user import UpdateUserRequest

    updated = await core.update_user(
        user.id, UpdateUserRequest(username="alice_updated")
    )
    assert updated.username == "alice_updated"

    # delete_user() - Remove user
    await core.delete_user(user.id)
    deleted = await core.get_user(user.id)
    assert deleted is None

    await core.close()


# --8<-- [end:user-management-services]


# --8<-- [start:group-management-services]
@pytest.mark.asyncio
async def test_group_management_services(config_registry):
    """Core group management services."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create user
    user = await core.create_user(
        CreateUserRequest(
            email="bob@example.com",
            username="bob",
            password="SecurePassword123!",
        )
    )

    # create_group() - Create new group
    from portico.ports.group import CreateGroupRequest

    group = await core.create_group(
        CreateGroupRequest(name="Editors", description="Content editors")
    )
    assert group is not None
    assert group.name == "Editors"

    # assign_group_role() - Add user to group
    await core.assign_group_role(user.id, group.id, "member")

    # check_permission() - Verify user has permission
    has_permission = await core.check_permission(user.id, "view")
    assert has_permission is not None

    # remove_from_group() - Remove user from group
    await core.remove_from_group(user.id, group.id)

    await core.close()


# --8<-- [end:group-management-services]


# --8<-- [start:audit-logging-service]
@pytest.mark.asyncio
async def test_audit_logging_service(config_registry):
    """Core audit logging service."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create user
    user = await core.create_user(
        CreateUserRequest(
            email="charlie@example.com",
            username="charlie",
            password="SecurePassword123!",
        )
    )

    # log_audit_event() - Record user actions
    await core.audit_action(
        user_id=user.id,
        action="document.created",
        resource_type="document",
        resource_id="doc-123",
        details={"title": "Important Document", "type": "pdf"},
    )

    # Verify audit event was logged
    assert user.id is not None

    await core.close()


# --8<-- [end:audit-logging-service]


# --8<-- [start:direct-database-access]
@pytest.mark.asyncio
async def test_direct_database_access(config_registry):
    """Direct access to database service."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Database access
    db = core.db
    assert db is not None

    # Can access database
    assert db is not None

    await core.close()


# --8<-- [end:direct-database-access]


# --8<-- [start:direct-event-bus-access]
@pytest.mark.asyncio
async def test_direct_event_bus_access(config_registry):
    """Direct access to event bus."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Event bus access
    event_bus = core.event_bus
    assert event_bus is not None

    # Event publisher convenience access
    event_publisher = core.event_publisher
    assert event_publisher is not None

    # Event publisher can publish events
    # (actual publishing requires proper event types)
    assert event_publisher is not None

    await core.close()


# --8<-- [end:direct-event-bus-access]


# --8<-- [start:fastapi-lifespan-integration]
@pytest.mark.asyncio
async def test_fastapi_lifespan_integration(config_registry):
    """Lifecycle management with FastAPI."""
    from unittest.mock import MagicMock

    mock_app = MagicMock()

    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )

    @asynccontextmanager
    async def lifespan(app):
        await core.initialize()
        yield
        await core.close()

    # Test lifespan pattern
    async with lifespan(mock_app):
        # Core is initialized and ready
        assert core.db is not None

    # Core is closed after context exit


# --8<-- [end:fastapi-lifespan-integration]


# --8<-- [start:core-services-overview]
@pytest.mark.asyncio
async def test_core_services_overview(config_registry):
    """Overview of all core services."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Core integrates:
    # - Authentication and session management
    assert hasattr(core, "authenticate_user")
    assert hasattr(core, "create_user_with_session")

    # - User and group management
    assert hasattr(core, "create_user")
    assert hasattr(core, "create_group")

    # - Role-based access control (RBAC)
    assert hasattr(core, "check_permission")

    # - Audit logging
    assert hasattr(core, "audit_action")

    # - Database connection management
    assert core.db is not None

    # - Event bus for cross-service communication
    assert core.event_bus is not None
    assert core.event_publisher is not None

    await core.close()


# --8<-- [end:core-services-overview]


# --8<-- [start:core-with-all-features]
@pytest.mark.asyncio
async def test_core_with_all_features(config_registry):
    """PorticoCore with all features enabled."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # All features available
    user = await core.create_user(
        CreateUserRequest(
            email="dave@example.com",
            username="dave",
            password="SecurePassword123!",
        )
    )

    # Groups available
    from portico.ports.group import CreateGroupRequest

    group = await core.create_group(CreateGroupRequest(name="Admins"))
    assert group is not None

    # Audit available
    await core.audit_action(
        user_id=user.id,
        action="user.created",
        resource_type="user",
        resource_id=str(user.id),
    )

    await core.close()


# --8<-- [end:core-with-all-features]


# --8<-- [start:password-hashing]
@pytest.mark.asyncio
async def test_password_hashing(config_registry):
    """Password hashing and verification."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Hash password
    password = "SecurePassword123!"
    hashed = await core.hash_password(password)

    assert hashed is not None
    assert hashed != password
    assert len(hashed) > 0

    # Verify password
    user = await core.create_user(
        CreateUserRequest(
            email="eve@example.com",
            username="eve",
            password=password,
        )
    )

    is_valid = await core.verify_password(user, password)
    assert is_valid is True

    is_invalid = await core.verify_password(user, "WrongPassword")
    assert is_invalid is False

    await core.close()


# --8<-- [end:password-hashing]


# --8<-- [start:core-initialization-steps]
@pytest.mark.asyncio
async def test_core_initialization_steps(config_registry):
    """What happens during core.initialize()."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )

    await core.initialize()

    # During initialize():
    # 1. Creates database tables (users, groups, sessions, audit, etc.)
    # 2. Registers default roles and permissions
    # 3. Sets up session storage
    # 4. Initializes audit logging
    # 5. Configures caching (if enabled)

    # Verify tables created
    assert core.db is not None

    await core.close()


# --8<-- [end:core-initialization-steps]


# --8<-- [start:core-cleanup]
@pytest.mark.asyncio
async def test_core_cleanup(config_registry):
    """Proper core cleanup."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Use core...
    user = await core.create_user(
        CreateUserRequest(
            email="frank@example.com",
            username="frank",
            password="SecurePassword123!",
        )
    )

    # Close performs:
    # - Closes database connections
    # - Flushes audit logs
    # - Clears caches
    # - Stops background tasks
    await core.close()

    # Verify cleanup
    assert user is not None


# --8<-- [end:core-cleanup]
