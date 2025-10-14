"""Test examples for audit logging kit documentation.

All examples are extracted into docs using snippet markers.
"""

import pytest

from portico.core import PorticoCore
from portico.ports.group import CreateGroupRequest
from portico.ports.user import CreateUserRequest


@pytest.fixture
async def test_core(config_registry):
    """Create test PorticoCore instance."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()
    await core.seed_default_rbac_data()

    yield core
    await core.close()


# --8<-- [start:basic-audit]
@pytest.mark.asyncio
async def test_basic_audit(test_core):
    """Log a basic audit action."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Log audit action
    await test_core.audit_action(
        user_id=user.id,
        action="document.create",
        resource_type="document",
        resource_id="doc-123",
        details={"title": "Important Document"},
        success=True,
    )

    # Audit entry is now recorded in the database


# --8<-- [end:basic-audit]


# --8<-- [start:audit-with-details]
@pytest.mark.asyncio
async def test_audit_with_details(test_core):
    """Log audit with detailed information."""
    user = await test_core.create_user(
        CreateUserRequest(
            email="admin@example.com",
            username="admin",
            password="SecurePassword123!",
        )
    )

    # Log with detailed information
    await test_core.audit_action(
        user_id=user.id,
        action="user.update",
        resource_type="user",
        resource_id=str(user.id),
        details={
            "changed_fields": ["email", "username"],
            "old_email": "old@example.com",
            "new_email": "admin@example.com",
            "ip_address": "192.168.1.1",
        },
        success=True,
    )


# --8<-- [end:audit-with-details]


# --8<-- [start:audit-failure]
@pytest.mark.asyncio
async def test_audit_failure(test_core):
    """Log failed operations."""
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Log failed operation
    await test_core.audit_action(
        user_id=user.id,
        action="file.delete",
        resource_type="file",
        resource_id="file-456",
        details={
            "reason": "Permission denied",
            "required_permission": "files.delete",
        },
        success=False,  # Mark as failure
    )


# --8<-- [end:audit-failure]


# --8<-- [start:audit-anonymous]
@pytest.mark.asyncio
async def test_audit_anonymous(test_core):
    """Log actions without a user (system actions)."""
    # Log system action (no user)
    await test_core.audit_action(
        user_id=None,  # System action
        action="system.cleanup",
        resource_type="system",
        resource_id=None,
        details={
            "task": "delete_expired_sessions",
            "deleted_count": 150,
        },
        success=True,
    )


# --8<-- [end:audit-anonymous]


# --8<-- [start:audit-group-action]
@pytest.mark.asyncio
async def test_audit_group_action(test_core):
    """Log group-related actions."""
    # Create user and group
    user = await test_core.create_user(
        CreateUserRequest(
            email="manager@example.com",
            username="manager",
            password="SecurePassword123!",
        )
    )

    group = await test_core.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="team",
        )
    )

    # Log group member addition
    await test_core.assign_group_role(user.id, group.id, "admin")

    await test_core.audit_action(
        user_id=user.id,
        action="group.add_member",
        resource_type="group",
        resource_id=str(group.id),
        details={
            "group_name": "Engineering",
            "member_id": str(user.id),
            "role": "admin",
        },
        success=True,
    )


# --8<-- [end:audit-group-action]


# --8<-- [start:audit-patterns]
@pytest.mark.asyncio
async def test_audit_patterns(test_core):
    """Common audit logging patterns."""
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Pattern 1: CRUD operations
    await test_core.audit_action(
        user_id=user.id,
        action="document.create",
        resource_type="document",
        resource_id="doc-1",
        details={"title": "New Doc"},
    )

    await test_core.audit_action(
        user_id=user.id,
        action="document.read",
        resource_type="document",
        resource_id="doc-1",
    )

    await test_core.audit_action(
        user_id=user.id,
        action="document.update",
        resource_type="document",
        resource_id="doc-1",
        details={"changed_fields": ["title", "content"]},
    )

    await test_core.audit_action(
        user_id=user.id,
        action="document.delete",
        resource_type="document",
        resource_id="doc-1",
    )


# --8<-- [end:audit-patterns]


# --8<-- [start:audit-security]
@pytest.mark.asyncio
async def test_audit_security(test_core):
    """Log security-related events."""
    user = await test_core.create_user(
        CreateUserRequest(
            email="secure@example.com",
            username="secure",
            password="SecurePassword123!",
        )
    )

    # Login attempt
    await test_core.audit_action(
        user_id=user.id,
        action="auth.login",
        resource_type="user",
        resource_id=str(user.id),
        details={"method": "password", "ip_address": "192.168.1.100"},
        success=True,
    )

    # Password change
    await test_core.audit_action(
        user_id=user.id,
        action="auth.password_change",
        resource_type="user",
        resource_id=str(user.id),
        details={"method": "user_initiated"},
        success=True,
    )

    # Failed login
    await test_core.audit_action(
        user_id=None,  # User might not be identified yet
        action="auth.login_failed",
        resource_type="user",
        resource_id=None,
        details={
            "username": "unknown@example.com",
            "reason": "invalid_credentials",
            "ip_address": "192.168.1.200",
        },
        success=False,
    )


# --8<-- [end:audit-security]
