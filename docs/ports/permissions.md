# Permissions Port

## Overview

The Permissions Port defines the contract for role-based access control (RBAC) in Portico applications.

**Purpose**: Abstract permission and role management to enable flexible authorization systems with global and group-scoped permissions.

**Domain**: Authorization, role-based access control, permission management

**Key Capabilities**:

- Global and group-scoped permissions
- Role definition with permission sets
- User permission checking (global and group contexts)
- System and custom roles/permissions
- Permission registration and discovery
- Category-based permission organization
- Hierarchical group permission inheritance

**Port Type**: Repository

**When to Use**:

- Applications requiring role-based access control
- Multi-tenant systems with group-level permissions
- Systems with hierarchical permission structures
- Applications needing custom permission definitions
- Authorization enforcement at API/service boundaries
- Administrative interfaces with fine-grained access control

## Domain Models

### Permission

Permission domain model. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Permission name (e.g., "users.create", "files.read") |
| `description` | `str` | Yes | - | Human-readable description |
| `scope` | `PermissionScope` | Yes | - | Permission scope (GLOBAL, GROUP, or PERSONAL) |
| `category` | `str` | Yes | - | Category for organization (e.g., "users", "files") |
| `is_system_permission` | `bool` | No | `False` | Whether this is a system-defined permission |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

**Example**:

```python
from portico.ports.permissions import Permission, PermissionScope

# Global permission
perm = Permission(
    name="users.create",
    description="Create new users",
    scope=PermissionScope.GLOBAL,
    category="users",
    is_system_permission=True
)

# Group permission
group_perm = Permission(
    name="documents.edit",
    description="Edit group documents",
    scope=PermissionScope.GROUP,
    category="documents",
    is_system_permission=False
)
```

### GlobalRole

Global role domain model with permissions. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Role name (e.g., "admin", "editor") |
| `description` | `str` | Yes | - | Human-readable description |
| `permissions` | `Set[str]` | Yes | - | Set of permission names this role grants |
| `is_system_role` | `bool` | No | `False` | Whether this is a system-defined role |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

**Example**:

```python
from portico.ports.permissions import GlobalRole

admin_role = GlobalRole(
    name="admin",
    description="System administrator with full access",
    permissions={
        "users.create",
        "users.read",
        "users.update",
        "users.delete",
        "roles.manage"
    },
    is_system_role=True
)

editor_role = GlobalRole(
    name="editor",
    description="Content editor",
    permissions={"content.create", "content.edit", "content.read"},
    is_system_role=False
)
```

### GroupRole

Group role domain model with permissions. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Role name (e.g., "member", "admin") |
| `description` | `str` | Yes | - | Human-readable description |
| `permissions` | `Set[str]` | Yes | - | Set of permission names this role grants |
| `is_system_role` | `bool` | No | `False` | Whether this is a system-defined role |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

**Example**:

```python
from portico.ports.permissions import GroupRole

group_admin = GroupRole(
    name="group_admin",
    description="Group administrator",
    permissions={
        "group.manage",
        "group.members.add",
        "group.members.remove",
        "group.documents.delete"
    },
    is_system_role=True
)

group_member = GroupRole(
    name="member",
    description="Regular group member",
    permissions={"group.documents.read", "group.documents.create"},
    is_system_role=False
)
```

### CreatePermissionRequest

Request for creating a new permission.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Permission name |
| `description` | `str` | Yes | - | Permission description |
| `scope` | `PermissionScope` | Yes | - | Permission scope |
| `category` | `str` | Yes | - | Permission category |

**Example**:

```python
from portico.ports.permissions import CreatePermissionRequest, PermissionScope

request = CreatePermissionRequest(
    name="reports.generate",
    description="Generate financial reports",
    scope=PermissionScope.GLOBAL,
    category="reports"
)
```

### CreateGlobalRoleRequest

Request for creating a global role.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Role name |
| `description` | `str` | Yes | - | Role description |
| `permissions` | `Set[str]` | Yes | - | Permission names to grant |

**Example**:

```python
from portico.ports.permissions import CreateGlobalRoleRequest

request = CreateGlobalRoleRequest(
    name="analyst",
    description="Data analyst with reporting access",
    permissions={"reports.generate", "reports.read", "data.read"}
)
```

### CreateGroupRoleRequest

Request for creating a group role.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Role name |
| `description` | `str` | Yes | - | Role description |
| `permissions` | `Set[str]` | Yes | - | Permission names to grant |

**Example**:

```python
from portico.ports.permissions import CreateGroupRoleRequest

request = CreateGroupRoleRequest(
    name="moderator",
    description="Group moderator",
    permissions={"group.posts.edit", "group.posts.delete", "group.members.warn"}
)
```

## Enumerations

### PermissionScope

Scope of permissions.

| Value | Description |
|-------|-------------|
| `GLOBAL` | Global system-wide permission |
| `GROUP` | Group/organization-scoped permission |
| `PERSONAL` | Personal/user-scoped permission |

**Example**:

```python
from portico.ports.permissions import PermissionScope

# Use enum for type safety
perm = CreatePermissionRequest(
    name="files.upload",
    description="Upload files",
    scope=PermissionScope.GROUP,  # Group-scoped
    category="files"
)
```

## Port Interfaces

### PermissionRepository

The `PermissionRepository` abstract base class defines the contract for permission and role operations.

**Location**: `portico.ports.permissions.PermissionRepository`

#### Key Methods

##### check_permission

```python
async def check_permission(
    user_id: UUID,
    permission: str,
    group_id: Optional[UUID] = None
) -> bool
```

Check if user has permission with optional group context. Primary method for authorization checks.

**Parameters**:

- `user_id`: User identifier
- `permission`: Permission name to check (e.g., "users.create")
- `group_id`: Optional group context for permission check

**Returns**: True if user has the permission, False otherwise.

**Example**:

```python
from portico.ports.permissions import PermissionRepository

# Global permission check
can_create_users = await perm_repo.check_permission(
    user_id=current_user_id,
    permission="users.create"
)

if can_create_users:
    user = await user_service.create_user(request)
else:
    raise AuthorizationError("Insufficient permissions")

# Group permission check
can_edit_docs = await perm_repo.check_permission(
    user_id=current_user_id,
    permission="documents.edit",
    group_id=group_id
)

if can_edit_docs:
    await update_document(doc_id, content)
```

##### get_user_global_permissions

```python
async def get_user_global_permissions(user_id: UUID) -> Set[str]
```

Get user's global permissions. Primary method for retrieving all permissions.

**Parameters**:

- `user_id`: User identifier

**Returns**: Set of global permission names the user has.

**Example**:

```python
# Get all global permissions for user
permissions = await perm_repo.get_user_global_permissions(current_user_id)

print(f"User has {len(permissions)} global permissions:")
for perm in sorted(permissions):
    print(f"  - {perm}")

# Check if user has admin permissions
admin_perms = {"users.delete", "roles.manage", "system.configure"}
is_admin = admin_perms.issubset(permissions)
```

#### Other Methods

##### get_global_role

```python
async def get_global_role(role_name: str) -> Optional[GlobalRole]
```

Get global role by name. Returns GlobalRole if found, None otherwise.

##### get_group_role

```python
async def get_group_role(role_name: str) -> Optional[GroupRole]
```

Get group role by name. Returns GroupRole if found, None otherwise.

##### get_permission

```python
async def get_permission(permission_name: str) -> Optional[Permission]
```

Get permission by name. Returns Permission if found, None otherwise.

##### get_user_group_permissions

```python
async def get_user_group_permissions(user_id: UUID, group_id: UUID) -> Set[str]
```

Get user's permissions within a group hierarchy. Returns set of permission names.

##### register_global_roles

```python
async def register_global_roles(roles: Dict[str, List[str]]) -> None
```

Register custom global roles. Dictionary maps role names to permission lists.

##### register_group_roles

```python
async def register_group_roles(roles: Dict[str, List[str]]) -> None
```

Register custom group roles. Dictionary maps role names to permission lists.

##### register_permissions

```python
async def register_permissions(permissions: List[CreatePermissionRequest]) -> None
```

Register custom permissions. Takes list of permission creation requests.

##### has_global_permission

```python
async def has_global_permission(user_id: UUID, permission: str) -> bool
```

Check if user has global permission. Returns True if user has the permission, False otherwise.

##### has_group_permission

```python
async def has_group_permission(user_id: UUID, permission: str, group_id: UUID) -> bool
```

Check if user has permission within group hierarchy. Returns True if user has the permission in the group, False otherwise.

##### list_global_roles

```python
async def list_global_roles() -> List[GlobalRole]
```

List all global roles. Returns list of GlobalRole objects.

##### list_group_roles

```python
async def list_group_roles() -> List[GroupRole]
```

List all group roles. Returns list of GroupRole objects.

##### list_permissions

```python
async def list_permissions(scope: Optional[PermissionScope] = None) -> List[Permission]
```

List permissions, optionally filtered by scope. Returns list of Permission objects.

## Common Patterns

### Role-Based Authorization

```python
from portico.ports.permissions import PermissionRepository, CreateGlobalRoleRequest
from portico.exceptions import AuthorizationError

async def setup_rbac(perm_repo: PermissionRepository):
    """Initialize roles and permissions."""

    # Register permissions
    from portico.ports.permissions import CreatePermissionRequest, PermissionScope

    permissions = [
        CreatePermissionRequest(
            name="users.create",
            description="Create users",
            scope=PermissionScope.GLOBAL,
            category="users"
        ),
        CreatePermissionRequest(
            name="users.read",
            description="View users",
            scope=PermissionScope.GLOBAL,
            category="users"
        ),
        CreatePermissionRequest(
            name="users.update",
            description="Update users",
            scope=PermissionScope.GLOBAL,
            category="users"
        ),
        CreatePermissionRequest(
            name="users.delete",
            description="Delete users",
            scope=PermissionScope.GLOBAL,
            category="users"
        )
    ]
    await perm_repo.register_permissions(permissions)

    # Register roles
    await perm_repo.register_global_roles({
        "admin": ["users.create", "users.read", "users.update", "users.delete"],
        "user_manager": ["users.create", "users.read", "users.update"],
        "viewer": ["users.read"]
    })

async def require_permission(
    perm_repo: PermissionRepository,
    user_id: UUID,
    permission: str
):
    """Enforce permission requirement."""

    has_permission = await perm_repo.check_permission(user_id, permission)

    if not has_permission:
        raise AuthorizationError(
            f"User {user_id} lacks required permission: {permission}"
        )

# Usage in API endpoints
async def delete_user_endpoint(user_id: UUID, current_user_id: UUID):
    # Check permission
    await require_permission(perm_repo, current_user_id, "users.delete")

    # Permission granted, proceed
    await user_service.delete_user(user_id)
```

### Group-Scoped Authorization

```python
from portico.ports.permissions import PermissionRepository
from uuid import UUID

async def setup_group_permissions(perm_repo: PermissionRepository):
    """Initialize group-scoped permissions and roles."""

    # Register group permissions
    from portico.ports.permissions import CreatePermissionRequest, PermissionScope

    group_permissions = [
        CreatePermissionRequest(
            name="group.documents.create",
            description="Create group documents",
            scope=PermissionScope.GROUP,
            category="documents"
        ),
        CreatePermissionRequest(
            name="group.documents.edit",
            description="Edit group documents",
            scope=PermissionScope.GROUP,
            category="documents"
        ),
        CreatePermissionRequest(
            name="group.documents.delete",
            description="Delete group documents",
            scope=PermissionScope.GROUP,
            category="documents"
        ),
        CreatePermissionRequest(
            name="group.members.add",
            description="Add group members",
            scope=PermissionScope.GROUP,
            category="members"
        )
    ]
    await perm_repo.register_permissions(group_permissions)

    # Register group roles
    await perm_repo.register_group_roles({
        "group_admin": [
            "group.documents.create",
            "group.documents.edit",
            "group.documents.delete",
            "group.members.add"
        ],
        "editor": [
            "group.documents.create",
            "group.documents.edit"
        ],
        "member": [
            "group.documents.create"
        ]
    })

async def can_edit_document(
    perm_repo: PermissionRepository,
    user_id: UUID,
    group_id: UUID,
    document_id: UUID
) -> bool:
    """Check if user can edit a document in a group."""

    # Check group-scoped permission
    can_edit = await perm_repo.has_group_permission(
        user_id=user_id,
        permission="group.documents.edit",
        group_id=group_id
    )

    return can_edit

# Usage
if await can_edit_document(perm_repo, user_id, group_id, doc_id):
    await update_document(doc_id, new_content)
else:
    raise AuthorizationError("Cannot edit this document")
```

## Integration with Kits

The Permissions Port is used by the **RBAC Kit** to provide role and permission management services.

```python
from portico import compose

# Configure RBAC kit
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.user(),
        compose.group(),
        compose.rbac()
    ]
)

await app.initialize()

# Access RBAC service
rbac_service = app.kits["rbac"].service

# Create permission
from portico.ports.permissions import CreatePermissionRequest, PermissionScope

perm = await rbac_service.create_permission(
    CreatePermissionRequest(
        name="reports.export",
        description="Export reports to PDF",
        scope=PermissionScope.GLOBAL,
        category="reports"
    )
)

# Create role
from portico.ports.permissions import CreateGlobalRoleRequest

role = await rbac_service.create_role(
    CreateGlobalRoleRequest(
        name="analyst",
        description="Data analyst",
        permissions={"reports.export", "reports.view", "data.read"}
    )
)

# Assign role to user
await rbac_service.assign_role(user_id, role_name="analyst")

# Check permission
can_export = await rbac_service.check_permission(
    user_id=user_id,
    permission="reports.export"
)

if can_export:
    report_pdf = await generate_report_pdf()
```

The RBAC Kit provides:

- SQLAlchemy-based permission repository
- User role assignment (global and group-scoped)
- Permission checking with caching
- System role/permission initialization
- Event publishing for role changes

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Use Namespaced Permission Names**: Organize permissions by category with dot notation

   ```python
   # ✅ GOOD: Namespaced permissions
   permissions = {
       "users.create",
       "users.read",
       "users.update",
       "users.delete",
       "reports.generate",
       "reports.export"
   }

   # ❌ BAD: Flat permission names
   permissions = {
       "create_user",
       "read_user",
       "generate_report"
   }
   # Harder to organize and understand relationships
   ```

2. **Check Permissions, Not Roles**: Authorize based on permissions, not role names

   ```python
   # ✅ GOOD: Permission-based authorization
   can_delete = await perm_repo.check_permission(user_id, "users.delete")
   if can_delete:
       await delete_user(user_id)

   # ❌ BAD: Role-based authorization
   user_role = await get_user_role(user_id)
   if user_role == "admin":  # Brittle - what if multiple roles can delete?
       await delete_user(user_id)
   ```

3. **Use Group Permissions for Multi-Tenancy**: Scope permissions to groups for isolation

   ```python
   # ✅ GOOD: Group-scoped permissions
   can_edit = await perm_repo.has_group_permission(
       user_id=user_id,
       permission="documents.edit",
       group_id=group_id
   )

   # ❌ BAD: Global permission for group-specific action
   can_edit = await perm_repo.has_global_permission(user_id, "documents.edit")
   # No isolation - user could edit any group's documents!
   ```

4. **Register Permissions at Startup**: Define all permissions during initialization

   ```python
   # ✅ GOOD: Register all permissions at startup
   async def initialize_permissions(perm_repo: PermissionRepository):
       permissions = [
           CreatePermissionRequest(name="users.create", ...),
           CreatePermissionRequest(name="users.read", ...),
           # ... all permissions
       ]
       await perm_repo.register_permissions(permissions)

   # ❌ BAD: Ad-hoc permission creation
   # Create permissions when you discover you need them
   # Leads to inconsistency and missing permissions
   ```

5. **Use Minimal Permissions for Roles**: Grant only necessary permissions (principle of least privilege)

   ```python
   # ✅ GOOD: Minimal permissions
   await perm_repo.register_global_roles({
       "viewer": ["users.read", "reports.read"],  # Read-only
       "editor": ["users.read", "reports.read", "reports.create"],
       "admin": ["users.*", "reports.*", "system.*"]  # Full access
   })

   # ❌ BAD: Over-permissive roles
   await perm_repo.register_global_roles({
       "viewer": ["users.*", "reports.*"],  # Too much access for viewer!
   })
   ```

## FAQs

### What's the difference between global and group permissions?

- **Global permissions** apply system-wide and are checked without group context
- **Group permissions** are scoped to specific groups and require a group_id for checking

```python
# Global permission - system-wide
await perm_repo.check_permission(user_id, "users.create")

# Group permission - within a specific group
await perm_repo.check_permission(user_id, "documents.edit", group_id=group_id)
```

**Use global permissions for**: System administration, user management, global settings

**Use group permissions for**: Group-specific resources, team collaboration, multi-tenant isolation

### How do I create custom permissions?

Use `register_permissions()` to define custom permissions:

```python
from portico.ports.permissions import CreatePermissionRequest, PermissionScope

custom_permissions = [
    CreatePermissionRequest(
        name="invoices.approve",
        description="Approve invoices for payment",
        scope=PermissionScope.GLOBAL,
        category="invoices"
    ),
    CreatePermissionRequest(
        name="team.budget.view",
        description="View team budget",
        scope=PermissionScope.GROUP,
        category="budget"
    )
]

await perm_repo.register_permissions(custom_permissions)
```

### Can a user have multiple roles?

Yes! Users can have multiple global roles and multiple group roles. Permissions are combined:

```python
# User assigned multiple roles
await rbac_service.assign_role(user_id, "editor")
await rbac_service.assign_role(user_id, "analyst")

# User gets union of permissions from both roles
permissions = await perm_repo.get_user_global_permissions(user_id)
# Includes permissions from both "editor" and "analyst" roles
```

### How do group permissions inherit?

Group permissions can inherit from parent groups if your group hierarchy supports it. Implementation depends on the GroupKit configuration:

```python
# If user has "documents.edit" in parent group,
# they may have it in child groups (depends on GroupKit configuration)
can_edit = await perm_repo.has_group_permission(
    user_id=user_id,
    permission="documents.edit",
    group_id=child_group_id
)
```

Check your GroupKit configuration for inheritance rules.

### How do I implement a custom permission repository?

Implement the `PermissionRepository` interface:

```python
from portico.ports.permissions import (
    PermissionRepository,
    Permission,
    GlobalRole,
    GroupRole,
    CreatePermissionRequest
)

class CustomPermissionRepository(PermissionRepository):
    async def check_permission(
        self,
        user_id: UUID,
        permission: str,
        group_id: Optional[UUID] = None
    ) -> bool:
        # Your authorization logic
        if group_id:
            # Check group permission
            group_perms = await your_db.get_group_permissions(user_id, group_id)
            return permission in group_perms
        else:
            # Check global permission
            global_perms = await your_db.get_global_permissions(user_id)
            return permission in global_perms

    async def get_user_global_permissions(self, user_id: UUID) -> Set[str]:
        # Get all global permissions for user
        roles = await your_db.get_user_roles(user_id)
        permissions = set()
        for role in roles:
            role_perms = await your_db.get_role_permissions(role.id)
            permissions.update(role_perms)
        return permissions

    # Implement all other abstract methods...
```

Then use in composition:

```python
def rbac(**config):
    from your_module import CustomPermissionRepository
    from portico.kits.rbac import RBACKit

    def factory(database, events):
        perm_repo = CustomPermissionRepository(database)
        return RBACKit.create(database, events, config, permission_repository=perm_repo)

    return factory
```
