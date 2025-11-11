# RBAC Kit

## Overview

The RBAC (Role-Based Access Control) Kit provides authorization and permission management capabilities for Portico applications. It enables fine-grained access control through roles and permissions, supporting both global (system-wide) and group-scoped authorization with hierarchical permission cascading.

**Purpose**: Manage authorization through roles and permissions with flexible scoping and hierarchical delegation.

**Domain**: Authorization, access control, permissions, roles, delegation

**Capabilities**:

- Permission and role management
- Global (system-wide) role assignments
- Group-scoped role assignments
- Hierarchical permission cascading through group hierarchy
- Permission checking with context awareness
- Role and permission querying
- Event publishing for audit trails
- Database persistence for roles, permissions, and assignments
- Integration with Group Kit for multi-tenant authorization

**Architecture Type**: Stateful kit with database models

**When to Use**:

- Multi-user applications requiring authorization
- Multi-tenant systems with group-based access control
- Applications with hierarchical organizations
- Systems requiring fine-grained permissions
- Role-based delegation and administration
- Compliance requirements for access control

## Quick Start

### Basic Setup

```python
from portico import compose

app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.user(),
        compose.group(),  # Optional: for group-scoped permissions
        compose.rbac(),
    ],
)

await app.initialize()

# Get RBAC service
rbac_service = app.kits["rbac"].service

# Create permissions
from portico.kits.rbac import CreatePermissionRequest

await rbac_service.create_permission(
    CreatePermissionRequest(
        name="users.create",
        description="Create new users",
        cascades=False
    )
)

await rbac_service.create_permission(
    CreatePermissionRequest(
        name="users.delete",
        description="Delete users",
        cascades=False
    )
)

# Create role with permissions
from portico.kits.rbac import CreateRoleRequest

admin_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="admin",
        description="System administrator",
        permissions=["users.create", "users.delete"]
    )
)

# Assign global role to user
await rbac_service.assign_global_role(
    user_id=user.id,
    role_name="admin",
    assigned_by=current_user.id
)

# Check permission
has_permission = await rbac_service.check_permission(
    user_id=user.id,
    permission="users.create"
)

if has_permission:
    # User can create users
    pass
```

## Core Concepts

### Permissions

Permissions are atomic capabilities that can be assigned to roles. They follow a hierarchical naming convention (e.g., `resource.action`) for organization.

**Naming Convention**:

- `{resource}.{action}` - e.g., `users.create`, `files.read`, `reports.delete`
- `{module}.{resource}.{action}` - e.g., `admin.users.create`

**Cascading**: Permissions can cascade through group hierarchies when both the permission has `cascades=True` and the group has `permission_cascade_enabled=True`.

### Roles

Roles are collections of permissions that can be assigned to users. Roles simplify permission management by grouping related permissions.

**Types**:

- **System Roles**: Predefined roles (marked with `is_system=True`)
- **Custom Roles**: Application-defined roles

### Global Roles vs Group-Scoped Roles

**Global Roles**:

- Apply across the entire system
- User has permissions everywhere
- Suitable for system administrators, support staff
- Example: System admin, global moderator

**Group-Scoped Roles**:

- Apply only within a specific group (and optionally child groups)
- User has permissions only in that group context
- Suitable for multi-tenant, hierarchical organizations
- Example: Project admin, team member, department manager

### Permission Cascading

When both conditions are met:

1. Permission has `cascades=True`
2. Parent group has `permission_cascade_enabled=True`

Then permission checks walk up the group hierarchy, allowing administrators in parent groups to manage child groups.

```
Organization (cascade enabled)
└── Department (cascade enabled)
    └── Team (cascade disabled)

User with "manage.members" permission (cascading) in Organization:
- Can manage members in Organization ✓
- Can manage members in Department ✓ (cascaded)
- Cannot manage members in Team ✗ (team disables cascade)
```

### Authorization Service

The `AuthorizationService` provides all permission and role management operations:

**Permission Operations**:

- `create_permission()` - Define new permission
- `get_permission()` - Retrieve permission by name
- `list_permissions()` - List all permissions

**Role Operations**:

- `create_role()` - Create role with permissions
- `get_role()` - Retrieve role by name
- `list_roles()` - List all roles

**Global Role Assignment**:

- `assign_global_role()` - Assign system-wide role
- `revoke_global_role()` - Remove system-wide role
- `get_user_global_roles()` - Get user's global roles

**Group Role Assignment**:

- `assign_group_role()` - Assign group-scoped role
- `revoke_group_role()` - Remove group-scoped role
- `get_user_group_roles()` - Get user's roles in a group

**Permission Checking**:

- `check_permission()` - Check if user has permission
- `get_user_permissions()` - Get all user permissions

## Configuration

### RBACKitConfig

```python
from dataclasses import dataclass

@dataclass
class RBACKitConfig:
    enable_global_roles: bool = True   # Enable system-wide roles
    enable_group_roles: bool = True    # Enable group-scoped roles
```

### Composing the RBAC Kit

```python
from portico import compose

# Standard configuration (both role types enabled)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.user(),
        compose.group(),
        compose.rbac(),  # Both global and group roles enabled
    ],
)

# Global roles only (no group scoping)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.user(),
        compose.rbac(enable_group_roles=False),
    ],
)

# Group roles only (no global roles)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.user(),
        compose.group(),
        compose.rbac(enable_global_roles=False),
    ],
)
```

## Usage Examples

### 1. Creating Permissions and Roles

```python
from portico.kits.rbac import CreatePermissionRequest, CreateRoleRequest

# Define permissions
permissions = [
    CreatePermissionRequest(
        name="users.view",
        description="View user list",
        cascades=False
    ),
    CreatePermissionRequest(
        name="users.create",
        description="Create new users",
        cascades=False
    ),
    CreatePermissionRequest(
        name="users.edit",
        description="Edit user details",
        cascades=False
    ),
    CreatePermissionRequest(
        name="users.delete",
        description="Delete users",
        cascades=False
    ),
]

for perm in permissions:
    await rbac_service.create_permission(perm)

# Create roles with different permission sets
viewer_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="viewer",
        description="Read-only access",
        permissions=["users.view"]
    )
)

editor_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="editor",
        description="Can view and edit",
        permissions=["users.view", "users.edit"]
    )
)

admin_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="admin",
        description="Full access",
        permissions=["users.view", "users.create", "users.edit", "users.delete"]
    )
)
```

### 2. Global Role Assignment

```python
# Assign global admin role
await rbac_service.assign_global_role(
    user_id=user.id,
    role_name="admin",
    assigned_by=current_user.id
)

# Check global permissions (works anywhere in the system)
can_create = await rbac_service.check_permission(
    user_id=user.id,
    permission="users.create"
)

# Get all global roles for user
global_roles = await rbac_service.get_user_global_roles(user.id)
for role in global_roles:
    print(f"Role: {role.name}, Permissions: {role.permissions}")

# Revoke global role
await rbac_service.revoke_global_role(
    user_id=user.id,
    role_name="admin"
)
```

### 3. Group-Scoped Role Assignment

```python
# Create a project group
from portico.kits.group import CreateGroupRequest

project = await group_service.create_group(
    CreateGroupRequest(
        name="Project Alpha",
        description="Top secret project"
    )
)

# Assign user as project admin (scoped to this group only)
await rbac_service.assign_group_role(
    user_id=user.id,
    group_id=project.id,
    role_name="admin",
    assigned_by=current_user.id
)

# Check permission in group context
can_delete_in_project = await rbac_service.check_permission(
    user_id=user.id,
    permission="users.delete",
    group_id=project.id  # Context-aware check
)

# Get user's roles in this group
group_roles = await rbac_service.get_user_group_roles(user.id, project.id)
for role in group_roles:
    print(f"Role in {project.name}: {role.name}")
```

### 4. Hierarchical Permission Cascading

```python
# Create cascading permission
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="projects.manage",
        description="Manage projects and sub-projects",
        cascades=True  # Cascades through hierarchy
    )
)

# Create role with cascading permission
manager_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="project_manager",
        permissions=["projects.manage"]
    )
)

# Create organization hierarchy
from portico.kits.group import CreateGroupRequest

company = await group_service.create_group(
    CreateGroupRequest(
        name="ACME Corp",
        permission_cascade_enabled=True  # Enable cascading
    )
)

department = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering",
        parent_ids=[company.id],
        permission_cascade_enabled=True
    )
)

team = await group_service.create_group(
    CreateGroupRequest(
        name="Backend Team",
        parent_ids=[department.id],
        permission_cascade_enabled=True
    )
)

# Assign user as manager at company level
await rbac_service.assign_group_role(
    user_id=user.id,
    group_id=company.id,
    role_name="project_manager"
)

# User can manage all child groups (cascading)
can_manage_company = await rbac_service.check_permission(
    user_id=user.id,
    permission="projects.manage",
    group_id=company.id
)  # True

can_manage_dept = await rbac_service.check_permission(
    user_id=user.id,
    permission="projects.manage",
    group_id=department.id
)  # True (cascaded from parent)

can_manage_team = await rbac_service.check_permission(
    user_id=user.id,
    permission="projects.manage",
    group_id=team.id
)  # True (cascaded from grandparent)
```

### 5. FastAPI Route Protection

```python
from fastapi import Depends, HTTPException
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

async def require_permission(permission: str):
    """Dependency to check user has permission."""
    async def check(current_user = deps.current_user):
        rbac_service = app.kits["rbac"].service

        has_perm = await rbac_service.check_permission(
            user_id=current_user.id,
            permission=permission
        )

        if not has_perm:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {permission}"
            )

        return current_user

    return check

# Protect routes with permissions
@app.post("/users")
async def create_user(
    user_data: CreateUserRequest,
    current_user = Depends(require_permission("users.create"))
):
    # Only users with users.create permission can access
    return await user_service.create_user(user_data)

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user = Depends(require_permission("users.delete"))
):
    # Only users with users.delete permission can access
    await user_service.delete_user(user_id)
    return {"deleted": True}
```

## Domain Models

### Permission

Atomic capability that can be granted to roles.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Permission identifier |
| `name` | `str` | Permission name (e.g., "users.create") |
| `description` | `Optional[str]` | Human-readable description |
| `cascades` | `bool` | Whether permission cascades through hierarchy |
| `created_at` | `datetime` | Creation timestamp |

### Role

Collection of permissions that can be assigned to users.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Role identifier |
| `name` | `str` | Role name (e.g., "admin") |
| `description` | `Optional[str]` | Human-readable description |
| `permissions` | `List[str]` | List of permission names |
| `is_system` | `bool` | Whether role is system-defined |
| `created_at` | `datetime` | Creation timestamp |

### UserRole

Global role assignment to a user.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Assignment identifier |
| `user_id` | `UUID` | User identifier |
| `role_id` | `UUID` | Role identifier |
| `assigned_at` | `datetime` | Assignment timestamp |
| `assigned_by` | `Optional[UUID]` | User who made the assignment |

### UserGroupRole

Group-scoped role assignment to a user.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Assignment identifier |
| `user_id` | `UUID` | User identifier |
| `group_id` | `UUID` | Group identifier |
| `role_id` | `UUID` | Role identifier |
| `assigned_at` | `datetime` | Assignment timestamp |
| `assigned_by` | `Optional[UUID]` | User who made the assignment |

### CreatePermissionRequest

Request for creating a new permission.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Permission name |
| `description` | `Optional[str]` | Permission description |
| `cascades` | `bool` | Enable cascading (default: False) |

### CreateRoleRequest

Request for creating a new role.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Role name |
| `description` | `Optional[str]` | Role description |
| `permissions` | `List[str]` | List of permission names |

## Database Models

### RbacPermissionModel

SQLAlchemy model for `rbac_permissions` table.

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | `UUID` | Primary key |
| `name` | `String(255)` | Unique, indexed |
| `description` | `String(500)` | Nullable |
| `cascades` | `Boolean` | Default: False |
| `created_at` | `DateTime(tz)` | Default: now() |

### RbacRoleModel

SQLAlchemy model for `rbac_roles` table.

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | `UUID` | Primary key |
| `name` | `String(255)` | Unique, indexed |
| `description` | `String(500)` | Nullable |
| `is_system` | `Boolean` | Default: False |
| `created_at` | `DateTime(tz)` | Default: now() |

### RbacRolePermissionModel

Join table for `rbac_role_permissions`.

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | `UUID` | Primary key |
| `role_id` | `UUID` | FK: rbac_roles.id (CASCADE) |
| `permission_id` | `UUID` | FK: rbac_permissions.id (CASCADE) |
| | | Unique: (role_id, permission_id) |

### RbacUserRoleModel

SQLAlchemy model for `rbac_user_roles` table (global roles).

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | `UUID` | Primary key |
| `user_id` | `UUID` | FK: users.id (CASCADE) |
| `role_id` | `UUID` | FK: rbac_roles.id (CASCADE) |
| `assigned_at` | `DateTime(tz)` | Default: now() |
| `assigned_by` | `UUID` | FK: users.id (SET NULL), nullable |
| | | Unique: (user_id, role_id) |

### RbacUserGroupRoleModel

SQLAlchemy model for `rbac_user_group_roles` table (group-scoped roles).

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | `UUID` | Primary key |
| `user_id` | `UUID` | FK: users.id (CASCADE) |
| `group_id` | `UUID` | FK: kit_groups.id (CASCADE) |
| `role_id` | `UUID` | FK: rbac_roles.id (CASCADE) |
| `assigned_at` | `DateTime(tz)` | Default: now() |
| `assigned_by` | `UUID` | FK: users.id (SET NULL), nullable |
| | | Unique: (user_id, group_id, role_id) |

## Events

### PermissionCreatedEvent

Published when a permission is created.

| Field | Type | Description |
|-------|------|-------------|
| `permission_id` | `UUID` | Created permission ID |
| `name` | `str` | Permission name |
| `timestamp` | `datetime` | Event timestamp |

### RoleCreatedEvent

Published when a role is created.

| Field | Type | Description |
|-------|------|-------------|
| `role_id` | `UUID` | Created role ID |
| `name` | `str` | Role name |
| `timestamp` | `datetime` | Event timestamp |

### RoleAssignedEvent

Published when a global role is assigned.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `UUID` | User ID |
| `role_id` | `UUID` | Role ID |
| `assigned_by` | `Optional[UUID]` | Assigner user ID |
| `timestamp` | `datetime` | Event timestamp |

### RoleRevokedEvent

Published when a global role is revoked.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `UUID` | User ID |
| `role_id` | `UUID` | Role ID |
| `timestamp` | `datetime` | Event timestamp |

### GroupRoleAssignedEvent

Published when a group-scoped role is assigned.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `UUID` | User ID |
| `group_id` | `UUID` | Group ID |
| `role_id` | `UUID` | Role ID |
| `assigned_by` | `Optional[UUID]` | Assigner user ID |
| `timestamp` | `datetime` | Event timestamp |

### GroupRoleRevokedEvent

Published when a group-scoped role is revoked.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `UUID` | User ID |
| `group_id` | `UUID` | Group ID |
| `role_id` | `UUID` | Role ID |
| `timestamp` | `datetime` | Event timestamp |

## Best Practices

### 1. Use Hierarchical Permission Naming

Organize permissions hierarchically for clarity and maintainability.

```python
# GOOD - Hierarchical naming
permissions = [
    "users.view",
    "users.create",
    "users.edit",
    "users.delete",
    "files.view",
    "files.upload",
    "files.delete",
    "admin.settings.view",
    "admin.settings.edit",
]

# BAD - Flat, inconsistent naming
permissions = [
    "view_users",
    "create_user",
    "editUser",
    "DELETE_USER",
    "file_view",
]
```

**Why**: Hierarchical naming enables pattern matching, improves readability, and makes permission management more intuitive.

### 2. Create Roles for Common Permission Sets

Group related permissions into roles rather than assigning individual permissions.

```python
# GOOD - Role-based assignment
await rbac_service.create_role(
    CreateRoleRequest(
        name="content_editor",
        description="Can create and edit content",
        permissions=[
            "articles.create",
            "articles.edit",
            "articles.publish",
            "media.upload",
        ]
    )
)

await rbac_service.assign_global_role(user.id, "content_editor")

# BAD - Individual permission assignment (not supported, requires manual implementation)
# This pattern requires creating a role for each user
```

**Why**: Roles simplify permission management, reduce errors, and make it easier to update permissions for multiple users.

### 3. Use Group-Scoped Roles for Multi-Tenant Applications

Scope roles to groups for tenant isolation and hierarchical authorization.

```python
# GOOD - Group-scoped roles for multi-tenancy
# Create tenant-specific admin
await rbac_service.assign_group_role(
    user_id=tenant_admin.id,
    group_id=tenant_group.id,
    role_name="admin"
)

# Admin can only manage users in their tenant
can_manage = await rbac_service.check_permission(
    user_id=tenant_admin.id,
    permission="users.manage",
    group_id=tenant_group.id  # Scoped to tenant
)

# BAD - Global roles for multi-tenant (no isolation)
await rbac_service.assign_global_role(tenant_admin.id, "admin")
# Admin can manage users in ALL tenants - security issue!
```

**Why**: Group-scoped roles provide tenant isolation, preventing cross-tenant access and enabling delegated administration.

### 4. Enable Cascading Carefully

Only enable cascading for permissions that should inherit through hierarchy.

```python
# GOOD - Cascading for management permissions
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="departments.manage",
        description="Manage department settings and members",
        cascades=True  # Organization admin can manage all departments
    )
)

# GOOD - Non-cascading for sensitive operations
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="finance.view_salaries",
        description="View employee salaries",
        cascades=False  # Should not cascade to child groups
    )
)

# BAD - Cascading for sensitive permissions
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="users.delete",
        cascades=True  # Parent group admin can delete users everywhere - too broad
    )
)
```

**Why**: Cascading permissions should be used judiciously for management operations, not for sensitive data access or destructive operations.

### 5. Check Permissions, Not Roles

Check permissions in business logic, not role membership.

```python
# GOOD - Permission-based check
async def delete_user(user_id: UUID, current_user_id: UUID):
    has_permission = await rbac_service.check_permission(
        user_id=current_user_id,
        permission="users.delete"
    )

    if not has_permission:
        raise PermissionDenied("Cannot delete users")

    await user_service.delete_user(user_id)

# BAD - Role-based check
async def delete_user(user_id: UUID, current_user_id: UUID):
    roles = await rbac_service.get_user_global_roles(current_user_id)
    role_names = [r.name for r in roles]

    if "admin" not in role_names and "super_admin" not in role_names:
        raise PermissionDenied("Must be admin")

    await user_service.delete_user(user_id)
    # BAD: Brittle, requires updating code when roles change
```

**Why**: Permission checks are more flexible and maintainable. Role permissions can change without requiring code updates.

### 6. Provide Context in Group Permission Checks

Always provide group context when checking permissions in multi-tenant applications.

```python
# GOOD - Context-aware permission check
async def edit_document(doc_id: UUID, user_id: UUID):
    document = await get_document(doc_id)

    has_permission = await rbac_service.check_permission(
        user_id=user_id,
        permission="documents.edit",
        group_id=document.group_id  # Check in document's group context
    )

    if not has_permission:
        raise PermissionDenied()

    # Edit document

# BAD - No context (checks global only)
async def edit_document(doc_id: UUID, user_id: UUID):
    has_permission = await rbac_service.check_permission(
        user_id=user_id,
        permission="documents.edit"
        # Missing group_id - only checks global permissions
    )

    if not has_permission:
        raise PermissionDenied()
```

**Why**: Context-aware checks respect group-scoped permissions and enable proper multi-tenant isolation.

### 7. Audit Role and Permission Changes

Subscribe to RBAC events for audit logging and compliance.

```python
from portico.kits.rbac import (
    RoleAssignedEvent,
    RoleRevokedEvent,
    GroupRoleAssignedEvent,
    GroupRoleRevokedEvent
)

# Subscribe to role assignment events
@app.events.subscribe(RoleAssignedEvent)
async def log_role_assigned(event: RoleAssignedEvent):
    await audit_service.log(
        action="role.assigned",
        actor_id=event.assigned_by,
        target_id=event.user_id,
        details={
            "role_id": str(event.role_id),
            "scope": "global"
        }
    )

@app.events.subscribe(GroupRoleAssignedEvent)
async def log_group_role_assigned(event: GroupRoleAssignedEvent):
    await audit_service.log(
        action="role.assigned",
        actor_id=event.assigned_by,
        target_id=event.user_id,
        details={
            "role_id": str(event.role_id),
            "group_id": str(event.group_id),
            "scope": "group"
        }
    )
```

**Why**: Audit trails for authorization changes are critical for security, compliance, and troubleshooting.

## Security Considerations

### 1. Validate Role Assignments

Ensure users can only assign roles they have authority to assign.

```python
async def assign_role_with_validation(
    target_user_id: UUID,
    role_name: str,
    assigner_id: UUID,
    group_id: Optional[UUID] = None
):
    # Check if assigner has permission to assign roles
    can_assign = await rbac_service.check_permission(
        user_id=assigner_id,
        permission="roles.assign",
        group_id=group_id
    )

    if not can_assign:
        raise PermissionDenied("Cannot assign roles")

    # Prevent privilege escalation - don't allow assigning roles
    # with permissions the assigner doesn't have
    role = await rbac_service.get_role(role_name)
    assigner_permissions = await rbac_service.get_user_permissions(
        user_id=assigner_id,
        group_id=group_id
    )

    for perm in role.permissions:
        if perm not in assigner_permissions:
            raise PermissionDenied(
                f"Cannot assign role with permission you don't have: {perm}"
            )

    # Assign role
    if group_id:
        await rbac_service.assign_group_role(
            user_id=target_user_id,
            group_id=group_id,
            role_name=role_name,
            assigned_by=assigner_id
        )
    else:
        await rbac_service.assign_global_role(
            user_id=target_user_id,
            role_name=role_name,
            assigned_by=assigner_id
        )
```

### 2. Implement Least Privilege

Grant minimum necessary permissions for each role.

```python
# GOOD - Minimal permissions
viewer_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="document_viewer",
        permissions=["documents.view"]  # Read-only
    )
)

# BAD - Excessive permissions
viewer_role = await rbac_service.create_role(
    CreateRoleRequest(
        name="document_viewer",
        permissions=[
            "documents.view",
            "documents.edit",    # Unnecessary for viewer
            "documents.delete",  # Unnecessary for viewer
        ]
    )
)
```

### 3. Protect System Roles

Prevent modification or deletion of system-defined roles.

```python
async def delete_role_safely(role_name: str, admin_id: UUID):
    role = await rbac_service.get_role(role_name)

    if not role:
        raise NotFoundError(f"Role {role_name} not found")

    if role.is_system:
        raise ValidationError("Cannot delete system roles")

    # Check admin has permission
    can_manage_roles = await rbac_service.check_permission(
        user_id=admin_id,
        permission="roles.delete"
    )

    if not can_manage_roles:
        raise PermissionDenied()

    await rbac_service.repository.delete_role(role.id)
```

### 4. Rate Limit Permission Checks

Cache permission check results to prevent performance degradation or DoS.

```python
from functools import lru_cache
from datetime import datetime, timedelta

permission_cache = {}
cache_ttl = timedelta(minutes=5)

async def check_permission_cached(
    user_id: UUID,
    permission: str,
    group_id: Optional[UUID] = None
) -> bool:
    cache_key = (user_id, permission, group_id)
    now = datetime.utcnow()

    # Check cache
    if cache_key in permission_cache:
        cached_result, cached_time = permission_cache[cache_key]
        if now - cached_time < cache_ttl:
            return cached_result

    # Check permission
    result = await rbac_service.check_permission(user_id, permission, group_id)

    # Update cache
    permission_cache[cache_key] = (result, now)

    return result
```

## FAQs

### 1. When should I use global roles vs group-scoped roles?

**Use Global Roles** for system-wide authority that applies across all contexts:

- System administrators
- Support staff
- Global moderators
- Billing administrators

**Use Group-Scoped Roles** for context-specific authority within groups:

- Project managers (manage specific projects)
- Team leads (manage specific teams)
- Department administrators (manage specific departments)
- Tenant administrators in multi-tenant systems

```python
# Global role - works everywhere
await rbac_service.assign_global_role(support_user.id, "support_agent")

# Group role - works only in specific tenant
await rbac_service.assign_group_role(
    user_id=tenant_admin.id,
    group_id=tenant.id,
    role_name="tenant_admin"
)
```

### 2. How do I implement permission cascading?

Enable cascading by setting both `permission.cascades=True` and `group.permission_cascade_enabled=True`:

```python
# Create cascading permission
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="projects.manage",
        cascades=True  # Step 1: Enable on permission
    )
)

# Enable cascading on parent groups
parent_group = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering Division",
        permission_cascade_enabled=True  # Step 2: Enable on group
    )
)

child_group = await group_service.create_group(
    CreateGroupRequest(
        name="Backend Team",
        parent_ids=[parent_group.id],
        permission_cascade_enabled=True  # Also enable on child
    )
)

# Assign role at parent level
await rbac_service.assign_group_role(
    user_id=manager.id,
    group_id=parent_group.id,
    role_name="manager"  # Has projects.manage permission
)

# Permission check cascades to child
can_manage_child = await rbac_service.check_permission(
    user_id=manager.id,
    permission="projects.manage",
    group_id=child_group.id  # Returns True (cascaded from parent)
)
```

### 3. How do I prevent privilege escalation?

Validate that users can only assign roles with permissions they already have:

```python
async def safe_role_assignment(
    assigner_id: UUID,
    target_user_id: UUID,
    role_name: str,
    group_id: Optional[UUID] = None
):
    # Get role permissions
    role = await rbac_service.get_role(role_name)

    # Get assigner's permissions
    assigner_perms = await rbac_service.get_user_permissions(
        user_id=assigner_id,
        group_id=group_id
    )

    # Check each permission
    for perm in role.permissions:
        if perm not in assigner_perms:
            raise PermissionDenied(
                f"Cannot assign role with permission you don't have: {perm}"
            )

    # Safe to assign
    if group_id:
        await rbac_service.assign_group_role(
            target_user_id, group_id, role_name, assigner_id
        )
    else:
        await rbac_service.assign_global_role(
            target_user_id, role_name, assigner_id
        )
```

### 4. How do I implement custom permission logic?

Extend permission checking with custom business rules:

```python
async def check_permission_with_custom_logic(
    user_id: UUID,
    permission: str,
    resource_id: Optional[UUID] = None,
    group_id: Optional[UUID] = None
) -> bool:
    # Standard RBAC check
    has_rbac_permission = await rbac_service.check_permission(
        user_id=user_id,
        permission=permission,
        group_id=group_id
    )

    if has_rbac_permission:
        return True

    # Custom logic: resource owner can always edit
    if permission == "documents.edit" and resource_id:
        document = await get_document(resource_id)
        if document.owner_id == user_id:
            return True

    # Custom logic: user can edit their own profile
    if permission == "users.edit" and resource_id == user_id:
        return True

    return False
```

### 5. How do I test RBAC authorization?

Use pytest fixtures with test users and roles:

```python
import pytest
from portico.kits.rbac import CreateRoleRequest, CreatePermissionRequest

@pytest.fixture
async def rbac_setup(app):
    rbac_service = app.kits["rbac"].service

    # Create test permissions
    await rbac_service.create_permission(
        CreatePermissionRequest(name="test.read")
    )
    await rbac_service.create_permission(
        CreatePermissionRequest(name="test.write")
    )

    # Create test roles
    await rbac_service.create_role(
        CreateRoleRequest(name="reader", permissions=["test.read"])
    )
    await rbac_service.create_role(
        CreateRoleRequest(name="writer", permissions=["test.read", "test.write"])
    )

    return rbac_service

@pytest.mark.asyncio
async def test_permission_check(rbac_setup, test_user):
    rbac_service = rbac_setup

    # Assign role
    await rbac_service.assign_global_role(test_user.id, "reader")

    # Test permissions
    assert await rbac_service.check_permission(test_user.id, "test.read")
    assert not await rbac_service.check_permission(test_user.id, "test.write")
```

### 6. How do I handle permission changes for active users?

Permission checks are performed in real-time, so changes take effect immediately:

```python
# User currently has admin role
assert await rbac_service.check_permission(user.id, "users.delete")

# Revoke admin role
await rbac_service.revoke_global_role(user.id, "admin")

# Next permission check reflects the change
assert not await rbac_service.check_permission(user.id, "users.delete")

# For cached systems, invalidate cache on role changes
@app.events.subscribe(RoleRevokedEvent)
async def invalidate_permission_cache(event: RoleRevokedEvent):
    # Clear user's permission cache
    clear_user_permission_cache(event.user_id)
```

### 7. How do I implement permission UI/UX?

Show/hide UI elements based on permissions:

```python
from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@app.get("/dashboard")
async def dashboard(request: Request, current_user = deps.current_user):
    rbac_service = app.kits["rbac"].service

    # Check permissions for UI rendering
    can_create_users = await rbac_service.check_permission(
        user_id=current_user.id,
        permission="users.create"
    )

    can_delete_users = await rbac_service.check_permission(
        user_id=current_user.id,
        permission="users.delete"
    )

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user,
        "can_create_users": can_create_users,
        "can_delete_users": can_delete_users,
    })
```

In template:

```html
<!-- dashboard.html -->
{ % if can_create_users % }
    <button>Create User</button>
{ % endif % }

{ % if can_delete_users % }
    <button>Delete User</button>
{ % endif % }
```

### 8. How do I migrate from simple role checks to RBAC?

Gradually migrate by implementing RBAC alongside existing checks:

```python
# Phase 1: Add RBAC in parallel
async def check_admin_access(user_id: UUID) -> bool:
    # Old logic
    user = await user_service.get_user(user_id)
    if user.is_admin:
        return True

    # New RBAC logic (in parallel)
    has_admin_perm = await rbac_service.check_permission(
        user_id=user_id,
        permission="admin.access"
    )

    # Accept either for backwards compatibility
    return user.is_admin or has_admin_perm

# Phase 2: Monitor usage, assign RBAC roles to existing admins

# Phase 3: Remove old logic once all users migrated
async def check_admin_access(user_id: UUID) -> bool:
    return await rbac_service.check_permission(
        user_id=user_id,
        permission="admin.access"
    )
```

### 9. How do I implement time-limited permissions?

Extend the UserRole model with expiration or use scheduled jobs:

```python
# Scheduled job to check and revoke expired roles
async def revoke_expired_roles():
    from datetime import datetime, timedelta

    # Get all user roles (would need to extend model with expires_at)
    # For now, use external tracking
    expired_assignments = await get_expired_role_assignments()

    for assignment in expired_assignments:
        await rbac_service.revoke_global_role(
            user_id=assignment.user_id,
            role_name=assignment.role_name
        )

    await audit_service.log(
        action="roles.auto_revoked",
        details={"count": len(expired_assignments)}
    )

# Run hourly
import asyncio
asyncio.create_task(periodic_task(revoke_expired_roles, hours=1))
```

### 10. How do I implement row-level security with RBAC?

Combine RBAC with resource ownership checks:

```python
async def can_access_resource(
    user_id: UUID,
    resource_id: UUID,
    permission: str
) -> bool:
    resource = await get_resource(resource_id)

    # Resource owner always has access
    if resource.owner_id == user_id:
        return True

    # Check RBAC permission in resource's group context
    if resource.group_id:
        return await rbac_service.check_permission(
            user_id=user_id,
            permission=permission,
            group_id=resource.group_id
        )

    # Check global permission
    return await rbac_service.check_permission(
        user_id=user_id,
        permission=permission
    )
```

## Related Kits

- **User Kit** - User management (required dependency)
- **Group Kit** - Group management for group-scoped permissions
- **Audit Kit** - Audit logging for authorization events

## Architecture Notes

The RBAC Kit is a **stateful kit** that manages authorization through database-persisted roles, permissions, and assignments. It supports two authorization scopes:

1. **Global Scope**: System-wide roles and permissions
2. **Group Scope**: Group-specific roles and permissions with optional hierarchical cascading

**Key Architectural Decisions**:

- **Separation of roles and permissions**: Roles are collections of permissions, enabling flexible permission management
- **Two-tier scoping**: Global and group-scoped roles support both system-wide and multi-tenant authorization
- **Hierarchical cascading**: Permissions can cascade through group hierarchies when enabled on both permission and group
- **Permission-based checks**: Business logic checks permissions, not roles, for flexibility
- **Event publishing**: All authorization changes publish events for audit trails
- **Repository pattern**: Service depends on repository interface, not database models
- **Optional Group Kit dependency**: Group-scoped features require Group Kit, but global roles work standalone

The RBAC Kit follows hexagonal architecture by depending on the repository abstraction and publishing domain events, enabling flexible deployment configurations and comprehensive authorization strategies.
