"""Test examples for RBAC documentation.

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

    # Seed default RBAC data (permissions and roles)
    await core.seed_default_rbac_data()

    yield core
    await core.close()


# --8<-- [start:check-permission]
@pytest.mark.asyncio
async def test_check_permission(test_core):
    """Check if user has a permission."""
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
            name="Dev Team",
            group_type="team",
        )
    )

    # Assign user as member
    await test_core.assign_group_role(user.id, group.id, "member")

    # Check permission in group
    can_read = await test_core.check_permission(
        user_id=user.id,
        permission="files.read",
        group_id=group.id,
    )

    assert can_read is True


# --8<-- [end:check-permission]


# --8<-- [start:global-permission]
@pytest.mark.asyncio
async def test_global_permission(test_core):
    """Check global permissions for admin users."""
    # Create admin user
    admin = await test_core.create_user(
        CreateUserRequest(
            email="admin@example.com",
            username="admin",
            password="SecurePassword123!",
            global_role="admin",
        )
    )

    # Verify admin role
    role = await test_core.get_user_global_role(admin.id)
    assert role == "admin"

    # Admin users have elevated privileges globally
    # They can perform administrative tasks


# --8<-- [end:global-permission]


# --8<-- [start:group-permission]
@pytest.mark.asyncio
async def test_group_permission(test_core):
    """Check permissions in specific groups."""
    # Create user and group
    user = await test_core.create_user(
        CreateUserRequest(
            email="member@example.com",
            username="member",
            password="SecurePassword123!",
        )
    )

    team = await test_core.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="team",
        )
    )

    # Assign as admin
    await test_core.assign_group_role(user.id, team.id, "admin")

    # Check group-specific permission
    can_write = await test_core.has_group_permission(
        user_id=user.id,
        permission="files.write",
        group_id=team.id,
    )

    assert can_write is True


# --8<-- [end:group-permission]


# --8<-- [start:role-permissions]
@pytest.mark.asyncio
async def test_role_permissions(test_core):
    """Different roles have different permissions."""
    # Create users
    admin_user = await test_core.create_user(
        CreateUserRequest(
            email="admin@example.com",
            username="admin",
            password="SecurePassword123!",
        )
    )

    member_user = await test_core.create_user(
        CreateUserRequest(
            email="member@example.com",
            username="member",
            password="SecurePassword123!",
        )
    )

    # Create group
    group = await test_core.create_group(
        CreateGroupRequest(
            name="Project",
            group_type="project",
        )
    )

    # Assign roles
    await test_core.assign_group_role(admin_user.id, group.id, "admin")
    await test_core.assign_group_role(member_user.id, group.id, "member")

    # Admin can delete
    admin_can_delete = await test_core.has_group_permission(
        admin_user.id, "files.delete", group.id
    )
    assert admin_can_delete is True

    # Member cannot delete
    member_cannot_delete = await test_core.has_group_permission(
        member_user.id, "files.delete", group.id
    )
    assert member_cannot_delete is False


# --8<-- [end:role-permissions]


# --8<-- [start:hierarchical-permissions]
@pytest.mark.asyncio
async def test_hierarchical_permissions(test_core):
    """Permissions inherited through group hierarchy."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Create organization
    org = await test_core.create_group(
        CreateGroupRequest(
            name="Acme Corp",
            group_type="organization",
        )
    )

    # Create department under org
    dept = await test_core.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="department",
            parent_id=org.id,
        )
    )

    # Assign user as owner in org
    await test_core.assign_group_role(user.id, org.id, "owner")

    # User has owner permissions in org
    can_delete_org = await test_core.has_group_permission(
        user.id, "files.delete", org.id
    )
    assert can_delete_org is True

    # Check if user has permissions in child department
    # (This depends on hierarchy implementation)
    roles_in_hierarchy = await test_core.get_user_roles_in_hierarchy(user.id, dept.id)
    # roles_in_hierarchy contains {org.id: "owner"}
    assert org.id in roles_in_hierarchy


# --8<-- [end:hierarchical-permissions]


# --8<-- [start:list-roles]
@pytest.mark.asyncio
async def test_list_roles(test_core):
    """List available roles and permissions."""
    # List global roles
    global_roles = await test_core.list_global_roles()
    assert len(global_roles) > 0
    assert any(r.name == "admin" for r in global_roles)

    # List group roles
    group_roles = await test_core.list_group_roles()
    assert len(group_roles) > 0
    assert any(r.name == "admin" for r in group_roles)
    assert any(r.name == "member" for r in group_roles)

    # List permissions
    permissions = await test_core.list_permissions()
    assert len(permissions) > 0


# --8<-- [end:list-roles]


# --8<-- [start:custom-permissions]
@pytest.mark.asyncio
async def test_custom_permissions(test_core):
    """Register custom permissions and roles."""
    # Register custom permissions
    await test_core.register_permissions(
        [
            {
                "name": "documents.publish",
                "description": "Publish documents",
                "scope": "group",
                "category": "documents",
            },
            {
                "name": "documents.archive",
                "description": "Archive documents",
                "scope": "group",
                "category": "documents",
            },
        ]
    )

    # Register custom group role
    await test_core.register_group_roles(
        {
            "publisher": [
                "files.read",
                "files.write",
                "documents.publish",
            ],
        }
    )

    # Verify permission registered
    permissions = await test_core.list_permissions()
    assert any(p.name == "documents.publish" for p in permissions)

    # Verify role registered
    group_roles = await test_core.list_group_roles()
    assert any(r.name == "publisher" for r in group_roles)


# --8<-- [end:custom-permissions]


# --8<-- [start:assign-global-role]
@pytest.mark.asyncio
async def test_assign_global_role(test_core):
    """Assign global roles to users."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
            global_role="user",
        )
    )

    # Verify initial role
    role = await test_core.get_user_global_role(user.id)
    assert role == "user"

    # Upgrade to admin
    await test_core.assign_global_role(user.id, "admin")

    # Verify new role
    new_role = await test_core.get_user_global_role(user.id)
    assert new_role == "admin"


# --8<-- [end:assign-global-role]


# --8<-- [start:permission-workflow]
@pytest.mark.asyncio
async def test_permission_workflow(test_core):
    """Complete permission workflow example."""
    # Create organization and users
    owner = await test_core.create_user(
        CreateUserRequest(
            email="owner@company.com",
            username="owner",
            password="SecurePassword123!",
        )
    )

    developer = await test_core.create_user(
        CreateUserRequest(
            email="dev@company.com",
            username="developer",
            password="SecurePassword123!",
        )
    )

    org = await test_core.create_group(
        CreateGroupRequest(
            name="Tech Corp",
            group_type="organization",
        )
    )

    # Assign roles
    await test_core.assign_group_role(owner.id, org.id, "owner")
    await test_core.assign_group_role(developer.id, org.id, "member")

    # Owner has full permissions
    owner_can_delete = await test_core.has_group_permission(
        owner.id, "files.delete", org.id
    )
    assert owner_can_delete is True

    # Member has read but not write or delete
    dev_can_read = await test_core.has_group_permission(
        developer.id, "files.read", org.id
    )
    dev_cannot_write = await test_core.has_group_permission(
        developer.id, "files.write", org.id
    )
    dev_cannot_delete = await test_core.has_group_permission(
        developer.id, "files.delete", org.id
    )

    assert dev_can_read is True
    assert dev_cannot_write is False
    assert dev_cannot_delete is False

    # Promote developer to admin
    await test_core.update_group_role(developer.id, org.id, "admin")

    # Now developer can delete
    dev_can_delete = await test_core.has_group_permission(
        developer.id, "files.delete", org.id
    )
    assert dev_can_delete is True


# --8<-- [end:permission-workflow]
