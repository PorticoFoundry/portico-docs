"""Test examples for group management kit documentation.

All examples are extracted into docs using snippet markers.
"""

import pytest

from portico.core import PorticoCore
from portico.ports.group import CreateGroupRequest, UpdateGroupRequest
from portico.ports.user import CreateUserRequest


@pytest.fixture
async def test_core(config_registry):
    """Create test PorticoCore instance."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    yield core
    await core.close()


# --8<-- [start:create-group]
@pytest.mark.asyncio
async def test_create_group(test_core):
    """Create a new group."""
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="department",
            description="Engineering department",
        )
    )

    assert group is not None
    assert group.name == "Engineering"
    assert group.group_type == "department"
    assert group.is_active is True


# --8<-- [end:create-group]


# --8<-- [start:create-group-with-admin]
@pytest.mark.asyncio
async def test_create_group_with_admin(test_core):
    """Create a group and assign creator as admin."""
    # Create user first
    user = await test_core.create_user(
        CreateUserRequest(
            email="admin@example.com",
            username="admin",
            password="SecurePassword123!",
        )
    )

    # Create group with admin
    group = await test_core.create_group_with_admin(
        CreateGroupRequest(
            name="Project Alpha",
            group_type="project",
            description="Alpha project team",
        ),
        admin_user_id=user.id,
    )

    assert group is not None
    assert group.name == "Project Alpha"

    # Verify user is admin
    memberships = await test_core.get_user_memberships(user.id)
    assert len(memberships) == 1
    assert memberships[0].role == "admin"


# --8<-- [end:create-group-with-admin]


# --8<-- [start:get-group]
@pytest.mark.asyncio
async def test_get_group(test_core):
    """Get group by ID."""
    # Create group
    created = await test_core.create_group(
        CreateGroupRequest(
            name="Sales",
            group_type="department",
        )
    )

    # Get group
    group = await test_core.get_group(created.id)

    assert group is not None
    assert group.id == created.id
    assert group.name == "Sales"


# --8<-- [end:get-group]


# --8<-- [start:update-group]
@pytest.mark.asyncio
async def test_update_group(test_core):
    """Update group information."""
    # Create group
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Marketing",
            group_type="department",
        )
    )

    # Update description
    updated = await test_core.update_group(
        group.id,
        UpdateGroupRequest(description="Marketing and Communications"),
    )

    assert updated is not None
    assert updated.description == "Marketing and Communications"


# --8<-- [end:update-group]


# --8<-- [start:delete-group]
@pytest.mark.asyncio
async def test_delete_group(test_core):
    """Delete a group."""
    # Create group
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Temp Team",
            group_type="team",
        )
    )

    # Delete group
    deleted = await test_core.delete_group(group.id)
    assert deleted is True

    # Verify deleted
    group_after = await test_core.get_group(group.id)
    assert group_after is None


# --8<-- [end:delete-group]


# --8<-- [start:assign-role]
@pytest.mark.asyncio
async def test_assign_group_role(test_core):
    """Assign a user to a group with a role."""
    # Create user and group
    user = await test_core.create_user(
        CreateUserRequest(
            email="member@example.com",
            username="member",
            password="SecurePassword123!",
        )
    )

    group = await test_core.create_group(
        CreateGroupRequest(
            name="Dev Team",
            group_type="team",
        )
    )

    # Assign role
    assigned = await test_core.assign_group_role(
        user_id=user.id,
        group_id=group.id,
        role="member",
    )

    assert assigned is True

    # Verify membership
    members = await test_core.get_group_members(group.id)
    assert len(members) == 1
    assert members[0].user_id == user.id
    assert members[0].role == "member"


# --8<-- [end:assign-role]


# --8<-- [start:update-role]
@pytest.mark.asyncio
async def test_update_group_role(test_core):
    """Update a user's role in a group."""
    # Create user and group
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    group = await test_core.create_group(
        CreateGroupRequest(
            name="Team A",
            group_type="team",
        )
    )

    # Assign as member
    await test_core.assign_group_role(
        user_id=user.id,
        group_id=group.id,
        role="member",
    )

    # Update to admin
    updated = await test_core.update_group_role(
        user_id=user.id,
        group_id=group.id,
        new_role="admin",
    )

    assert updated is True

    # Verify new role
    members = await test_core.get_group_members(group.id)
    assert members[0].role == "admin"


# --8<-- [end:update-role]


# --8<-- [start:remove-member]
@pytest.mark.asyncio
async def test_remove_from_group(test_core):
    """Remove a user from a group."""
    # Create user and group
    user = await test_core.create_user(
        CreateUserRequest(
            email="remove@example.com",
            username="remove",
            password="SecurePassword123!",
        )
    )

    group = await test_core.create_group(
        CreateGroupRequest(
            name="Team B",
            group_type="team",
        )
    )

    # Assign role
    await test_core.assign_group_role(
        user_id=user.id,
        group_id=group.id,
        role="member",
    )

    # Remove from group
    removed = await test_core.remove_from_group(user.id, group.id)
    assert removed is True

    # Verify removal
    members = await test_core.get_group_members(group.id)
    active_members = [m for m in members if m.is_active]
    assert len(active_members) == 0


# --8<-- [end:remove-member]


# --8<-- [start:list-members]
@pytest.mark.asyncio
async def test_get_group_members(test_core):
    """Get all members of a group."""
    # Create group
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Team C",
            group_type="team",
        )
    )

    # Add multiple members
    for i in range(3):
        user = await test_core.create_user(
            CreateUserRequest(
                email=f"user{i}@example.com",
                username=f"user{i}",
                password="SecurePassword123!",
            )
        )
        await test_core.assign_group_role(
            user_id=user.id,
            group_id=group.id,
            role="member",
        )

    # Get members
    members = await test_core.get_group_members(group.id)

    assert len(members) == 3


# --8<-- [end:list-members]


# --8<-- [start:user-memberships]
@pytest.mark.asyncio
async def test_get_user_memberships(test_core):
    """Get all groups a user belongs to."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="multi@example.com",
            username="multi",
            password="SecurePassword123!",
        )
    )

    # Create multiple groups and add user
    for i in range(3):
        group = await test_core.create_group(
            CreateGroupRequest(
                name=f"Group {i}",
                group_type="team",
            )
        )
        await test_core.assign_group_role(
            user_id=user.id,
            group_id=group.id,
            role="member",
        )

    # Get user's memberships
    memberships = await test_core.get_user_memberships(user.id)

    assert len(memberships) == 3


# --8<-- [end:user-memberships]


# --8<-- [start:hierarchical-groups]
@pytest.mark.asyncio
async def test_hierarchical_groups(test_core):
    """Create hierarchical group structure."""
    # Create parent organization
    company = await test_core.create_group(
        CreateGroupRequest(
            name="Acme Corp",
            group_type="organization",
        )
    )

    # Create child department
    engineering = await test_core.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="department",
            parent_id=company.id,
        )
    )

    # Create sub-team
    backend = await test_core.create_group(
        CreateGroupRequest(
            name="Backend Team",
            group_type="team",
            parent_id=engineering.id,
        )
    )

    # Verify hierarchy
    assert backend.parent_id == engineering.id
    assert engineering.parent_id == company.id


# --8<-- [end:hierarchical-groups]


# --8<-- [start:group-metadata]
@pytest.mark.asyncio
async def test_group_with_metadata(test_core):
    """Create a group with custom metadata."""
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Sales West",
            group_type="team",
            metadata={
                "region": "US-West",
                "budget_code": "SALES-2024",
                "manager_email": "manager@example.com",
            },
        )
    )

    assert group.metadata is not None
    assert group.metadata["region"] == "US-West"
    assert group.metadata["budget_code"] == "SALES-2024"


# --8<-- [end:group-metadata]
