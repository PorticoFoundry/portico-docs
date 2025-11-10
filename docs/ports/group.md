# Group Port

## Overview

The Group Port defines the contract for managing groups, group memberships, and hierarchical organizational structures in Portico applications.

**Purpose**: Provides interfaces and domain models for creating organizational hierarchies, managing group memberships with roles, and implementing group-based access control.

**Domain**: Group management, organizational structure, team hierarchies, role-based membership

**Key Capabilities**:

- Group CRUD operations (create, read, update, delete)
- Hierarchical group structures (parent-child relationships)
- Group membership management with role assignment
- User membership queries across multiple groups
- Hierarchical role and permission inheritance
- Group-specific role and permission management

**Port Type**: Repository (with additional RoleManager interface)

**When to Use**:

- Building multi-tenant applications with organizational hierarchies
- Implementing team-based access control
- Managing user memberships across organizations, teams, and projects
- Creating hierarchical permission structures (organization → team → project)
- Implementing workspace or tenant isolation

## Domain Models

### Group

Core domain model representing a group or organizational unit. Supports hierarchical structures through `parent_ids` with multi-parent support. Immutable snapshot of group state.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | `uuid4()` | Unique group identifier |
| `name` | `str` | Yes | - | Group name (unique within type) |
| `group_type` | `str` | Yes | `"organization"` | Type of group (organization, team, project, etc.) |
| `description` | `Optional[str]` | No | `None` | Group description |
| `parent_ids` | `List[UUID]` | Yes | `[]` | Parent group IDs (supports multiple parents) |
| `is_active` | `bool` | Yes | `True` | Whether the group is active |
| `permission_cascade_enabled` | `bool` | Yes | `True` | Whether permissions cascade through this group |
| `metadata` | `Dict[str, str]` | Yes | `{}` | Custom metadata key-value pairs |
| `created_at` | `datetime` | Yes | `now(UTC)` | Group creation timestamp (UTC) |
| `updated_at` | `datetime` | Yes | `now(UTC)` | Last update timestamp (UTC) |

**Example**:

```python
from portico.ports.group import Group
from datetime import datetime, UTC
from uuid import uuid4

# Root organization
org = Group(
    id=uuid4(),
    name="Acme Corporation",
    group_type="organization",
    description="Root organization",
    parent_ids=[],  # No parents - this is the root
    is_active=True,
    permission_cascade_enabled=True,
    metadata={"industry": "technology", "region": "us-west"},
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC)
)

# Team under organization
team = Group(
    id=uuid4(),
    name="Engineering Team",
    group_type="team",
    description="Software engineering team",
    parent_ids=[org.id],  # Belongs to organization
    is_active=True,
    permission_cascade_enabled=True,
    metadata={"department": "engineering"},
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC)
)

# Cross-functional project with multiple parents (matrix organization)
project = Group(
    id=uuid4(),
    name="Product Launch",
    group_type="project",
    description="Cross-functional product launch project",
    parent_ids=[engineering_team.id, marketing_team.id],  # Multiple parents!
    is_active=True,
    permission_cascade_enabled=True,
    metadata={"priority": "high"},
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC)
)
```

### GroupMembership

Represents a user's membership in a group with a specific role.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | `UUID` | Yes | - | User who is a member |
| `group_id` | `UUID` | Yes | - | Group they belong to |
| `role` | `str` | Yes | - | Role within the group (e.g., "owner", "admin", "member") |
| `joined_at` | `datetime` | Yes | `now(UTC)` | When the user joined |
| `invited_by` | `Optional[UUID]` | No | `None` | User ID who invited this member |
| `is_active` | `bool` | Yes | `True` | Whether membership is active |

**Example**:

```python
from portico.ports.group import GroupMembership
from datetime import datetime, UTC
from uuid import uuid4

membership = GroupMembership(
    user_id=uuid4(),
    group_id=uuid4(),
    role="admin",
    joined_at=datetime.now(UTC),
    invited_by=uuid4(),
    is_active=True
)
```

### CreateGroupRequest

Request model for creating a new group.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Group name |
| `group_type` | `str` | No | `"organization"` | Type of group |
| `description` | `Optional[str]` | No | `None` | Group description |
| `parent_ids` | `List[UUID]` | No | `[]` | Parent group IDs (supports multiple parents) |
| `permission_cascade_enabled` | `bool` | No | `True` | Whether permissions cascade through this group |
| `metadata` | `Dict[str, str]` | No | `{}` | Custom metadata |

**Example**:

```python
from portico.ports.group import CreateGroupRequest

# Create root organization
org_request = CreateGroupRequest(
    name="Acme Corporation",
    group_type="organization",
    description="Our main organization"
)

# Create team under organization
team_request = CreateGroupRequest(
    name="Engineering",
    group_type="team",
    description="Software engineering team",
    parent_ids=[org.id],  # Single parent
    metadata={"department": "engineering"}
)

# Create cross-functional project with multiple parents
project_request = CreateGroupRequest(
    name="Product Launch",
    group_type="project",
    description="Cross-functional project",
    parent_ids=[engineering.id, marketing.id],  # Multiple parents!
    metadata={"priority": "high"}
)
```

### UpdateGroupRequest

Request model for updating an existing group. All fields optional for partial updates.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `Optional[str]` | No | `None` | New group name |
| `description` | `Optional[str]` | No | `None` | New description |
| `is_active` | `Optional[bool]` | No | `None` | New active status |
| `metadata` | `Optional[Dict[str, str]]` | No | `None` | New metadata |

**Example**:

```python
from portico.ports.group import UpdateGroupRequest

# Update description only
request = UpdateGroupRequest(description="Updated team description")

# Deactivate group
request = UpdateGroupRequest(is_active=False)
```

### GroupMembershipRequest

Request model for group membership operations (add member, update role).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | `UUID` | Yes | - | User to add/modify |
| `group_id` | `UUID` | Yes | - | Target group |
| `role` | `str` | Yes | - | Role to assign |

**Example**:

```python
from portico.ports.group import GroupMembershipRequest

# Add user as admin
request = GroupMembershipRequest(
    user_id=user.id,
    group_id=team.id,
    role="admin"
)
```

## Port Interfaces

### GroupRepository

Abstract interface for group persistence operations.

**Location**: `portico.ports.group.GroupRepository`

#### Key Methods

##### create

```python
async def create(group_data: CreateGroupRequest) -> Group
```

Create a new group in the system.

**Parameters**:

- `group_data: CreateGroupRequest` - Group creation data

**Returns**: Created Group object

**Raises**:

- `ValidationError` - If group data is invalid
- `ConflictError` - If group name already exists for that type

**Example**:

```python
from portico.ports.group import GroupRepository, CreateGroupRequest

group = await repository.create(
    CreateGroupRequest(
        name="Engineering Team",
        group_type="team",
        description="Our engineering team"
    )
)
```

##### get_by_id

```python
async def get_by_id(group_id: UUID) -> Optional[Group]
```

Retrieve a group by its unique ID.

**Parameters**:

- `group_id: UUID` - Group identifier

**Returns**: Group if found, None otherwise

**Example**:

```python
group = await repository.get_by_id(group_id)
if group:
    print(f"Found group: {group.name}")
```

##### get_by_name

```python
async def get_by_name(name: str, group_type: str) -> Optional[Group]
```

Retrieve a group by name within a specific type.

**Parameters**:

- `name: str` - Group name
- `group_type: str` - Group type (e.g., "organization", "team")

**Returns**: Group if found, None otherwise

**Note**: Name uniqueness is scoped to group type, so you can have "Engineering" as both an organization and a team.

**Example**:

```python
org = await repository.get_by_name("Acme Corp", "organization")
team = await repository.get_by_name("Acme Corp", "team")  # Different group!
```

##### get_group_hierarchy

```python
async def get_group_hierarchy(group_id: UUID) -> List[Group]
```

Get all parent groups up the hierarchy from a given group.

**Parameters**:

- `group_id: UUID` - Starting group ID

**Returns**: List of parent groups from immediate parent to root

**Example**:

```python
# Get hierarchy: Project -> Team -> Organization
hierarchy = await repository.get_group_hierarchy(project.id)
# Returns: [team, organization]
```

##### get_child_groups

```python
async def get_child_groups(group_id: UUID) -> List[Group]
```

Get direct children of a group.

**Parameters**:

- `group_id: UUID` - Parent group ID

**Returns**: List of child Group objects

**Example**:

```python
# Get all teams in an organization
teams = await repository.get_child_groups(org.id)
```

##### get_user_roles_in_hierarchy

```python
async def get_user_roles_in_hierarchy(
    user_id: UUID,
    group_id: UUID
) -> Dict[UUID, str]
```

Get user's roles in a group and all its parent groups.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_id: UUID` - Starting group ID

**Returns**: Dictionary mapping group IDs to role names

**Example**:

```python
# User might be "member" in project, "admin" in team, "owner" in org
roles = await repository.get_user_roles_in_hierarchy(user.id, project.id)
# Returns: {project.id: "member", team.id: "admin", org.id: "owner"}
```

#### Other Methods

##### update

```python
async def update(
    group_id: UUID,
    update_data: UpdateGroupRequest
) -> Optional[Group]
```

Update an existing group. Performs partial update - only non-None fields are updated.

##### delete

```python
async def delete(group_id: UUID) -> bool
```

Delete a group by ID. Returns True if deleted, False if not found.

##### list_groups

```python
async def list_groups(
    group_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Group]
```

List groups with optional filtering by type and pagination.

### GroupMembershipRepository

Abstract interface for group membership persistence operations.

**Location**: `portico.ports.group.GroupMembershipRepository`

#### Key Methods

##### add_membership

```python
async def add_membership(
    membership: GroupMembershipRequest
) -> GroupMembership
```

Add a user to a group with a specific role.

**Parameters**:

- `membership: GroupMembershipRequest` - Membership details

**Returns**: Created GroupMembership object

**Example**:

```python
from portico.ports.group import GroupMembershipRequest

membership = await membership_repo.add_membership(
    GroupMembershipRequest(
        user_id=user.id,
        group_id=team.id,
        role="admin"
    )
)
```

##### remove_membership

```python
async def remove_membership(user_id: UUID, group_id: UUID) -> bool
```

Remove a user from a group.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_id: UUID` - Group identifier

**Returns**: True if membership was removed, False if not found

##### update_membership_role

```python
async def update_membership_role(
    user_id: UUID,
    group_id: UUID,
    new_role: str
) -> Optional[GroupMembership]
```

Update a user's role in a group.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_id: UUID` - Group identifier
- `new_role: str` - New role to assign

**Returns**: Updated GroupMembership if found, None otherwise

##### get_user_memberships

```python
async def get_user_memberships(user_id: UUID) -> List[GroupMembership]
```

Get all groups a user belongs to.

**Parameters**:

- `user_id: UUID` - User identifier

**Returns**: List of GroupMembership objects for the user

##### get_group_memberships

```python
async def get_group_memberships(group_id: UUID) -> List[GroupMembership]
```

Get all members of a group.

**Parameters**:

- `group_id: UUID` - Group identifier

**Returns**: List of GroupMembership objects for the group

##### get_user_groups_by_type

```python
async def get_user_groups_by_type(
    user_id: UUID,
    group_type: str
) -> List[Group]
```

Get all groups of a specific type that a user belongs to.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_type: str` - Group type filter

**Returns**: List of Group objects of the specified type

**Example**:

```python
# Get all organizations user belongs to
orgs = await membership_repo.get_user_groups_by_type(user.id, "organization")

# Get all teams user belongs to
teams = await membership_repo.get_user_groups_by_type(user.id, "team")
```

### GroupRoleManager

Abstract interface for group-specific role and permission management.

**Location**: `portico.ports.group.GroupRoleManager`

**Note**: This is a synchronous interface for in-memory role management.

#### Key Methods

##### define_group_role

```python
def define_group_role(
    group_type: str,
    role_name: str,
    permissions: Set[str]
) -> None
```

Define a role for a specific group type.

**Parameters**:

- `group_type: str` - Group type (e.g., "organization", "team")
- `role_name: str` - Role name (e.g., "owner", "admin", "member")
- `permissions: Set[str]` - Set of permission strings

**Example**:

```python
# Define organization-level roles
role_manager.define_group_role(
    "organization",
    "owner",
    {"org.manage", "org.delete", "team.create", "user.invite", "user.remove"}
)

role_manager.define_group_role(
    "organization",
    "member",
    {"org.view", "team.view"}
)

# Define team-level roles
role_manager.define_group_role(
    "team",
    "lead",
    {"team.manage", "task.assign", "user.invite"}
)
```

##### get_group_role_permissions

```python
def get_group_role_permissions(
    group_type: str,
    role_name: str
) -> Set[str]
```

Get permissions for a specific role within a group type.

**Parameters**:

- `group_type: str` - Group type
- `role_name: str` - Role name

**Returns**: Set of permission strings for the role

##### user_has_group_permission

```python
def user_has_group_permission(
    user_id: UUID,
    group_id: UUID,
    permission: str
) -> bool
```

Check if a user has a specific permission within a group.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_id: UUID` - Group identifier
- `permission: str` - Permission string to check

**Returns**: True if user has the permission, False otherwise

##### user_has_group_role

```python
def user_has_group_role(
    user_id: UUID,
    group_id: UUID,
    role: str
) -> bool
```

Check if a user has a specific role within a group.

**Parameters**:

- `user_id: UUID` - User identifier
- `group_id: UUID` - Group identifier
- `role: str` - Role name to check

**Returns**: True if user has the role, False otherwise

## Common Patterns

### Building Hierarchical Organizations

```python
from portico import compose
from portico.ports.group import CreateGroupRequest, GroupMembershipRequest

# Initialize application
app = compose.webapp(
    database_url="sqlite+aiosqlite:///app.db",
    kits=[compose.group(), compose.user()]
)
await app.initialize()

group_service = app.kits["group"].service

# Create organization (root)
org = await group_service.create_group(
    CreateGroupRequest(
        name="Acme Corporation",
        group_type="organization"
    )
)

# Create teams under organization
eng_team = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering",
        group_type="team",
        parent_id=org.id
    )
)

sales_team = await group_service.create_group(
    CreateGroupRequest(
        name="Sales",
        group_type="team",
        parent_id=org.id
    )
)

# Create project under team
project = await group_service.create_group(
    CreateGroupRequest(
        name="Product Launch",
        group_type="project",
        parent_id=eng_team.id
    )
)

# Hierarchy: org -> eng_team -> project
```

### Managing Group Memberships

```python
from portico.ports.group import GroupMembershipRequest

# Add user to organization as owner
await group_service.add_member(
    GroupMembershipRequest(
        user_id=user.id,
        group_id=org.id,
        role="owner"
    ),
    invited_by=admin_user.id
)

# Add user to team as admin
await group_service.add_member(
    GroupMembershipRequest(
        user_id=user.id,
        group_id=eng_team.id,
        role="admin"
    ),
    invited_by=org_owner.id
)

# List all members of a team
members = await group_service.get_group_members(eng_team.id)
for membership in members:
    print(f"User {membership.user_id} has role: {membership.role}")

# Get all groups a user belongs to
user_groups = await group_service.get_user_memberships(user.id)
```

### Hierarchical Permission Checking

```python
# Check if user has permission in a group or any parent group
has_permission = group_service.user_has_permission_in_hierarchy(
    user_id=user.id,
    group_id=project.id,
    permission="task.create"
)

# This checks:
# 1. Does user have permission in project?
# 2. Does user have permission in eng_team (parent)?
# 3. Does user have permission in org (grandparent)?

if has_permission:
    await create_task(project.id, task_data)
else:
    raise AuthorizationError("Insufficient permissions")
```

### Multi-Tenant Isolation

```python
# Each tenant gets their own organization
async def create_tenant(tenant_name: str, owner_id: UUID):
    # Create organization for tenant
    org = await group_service.create_group(
        CreateGroupRequest(
            name=tenant_name,
            group_type="organization"
        )
    )

    # Make creator the owner
    await group_service.add_member(
        GroupMembershipRequest(
            user_id=owner_id,
            group_id=org.id,
            role="owner"
        )
    )

    return org

# Query user's organizations to enforce tenant isolation
user_orgs = await group_service.repository.get_user_groups_by_type(
    user.id,
    "organization"
)

# Only show data from organizations user belongs to
accessible_org_ids = [org.id for org in user_orgs]
```

## Integration with Kits

The Group Port is used by the **Group Kit** to provide group management services.

```python
from portico import compose

# Configure Group Kit
app = compose.webapp(
    kits=[compose.group()]
)

# Access Group Service
group_service = app.kits["group"].service

# Create group
group = await group_service.create_group(
    CreateGroupRequest(name="Engineering", group_type="team")
)

# Add member
await group_service.add_member(
    GroupMembershipRequest(user_id=user.id, group_id=group.id, role="admin"),
    invited_by=owner.id
)

# Check permissions
has_perm = group_service.user_has_permission_in_hierarchy(
    user.id, group.id, "team.manage"
)
```

The Group Kit provides:

- Event publishing (GroupCreatedEvent, MemberAddedEvent, etc.)
- Validation before repository calls
- Hierarchical permission checking
- Convenience methods for common operations

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Hierarchical Design**: Use parent-child relationships to model organizational structures

   ```python
   # ✅ GOOD - Clear hierarchy
   organization (root)
     └── team (parent: org)
           └── project (parent: team)

   # ❌ BAD - Flat structure loses context
   organization, team, project (all separate, no relationships)
   ```

2. **Group Type Consistency**: Use consistent group types across your application

   ```python
   # ✅ GOOD - Standard types
   group_types = ["organization", "team", "project"]

   # ❌ BAD - Inconsistent naming
   group_types = ["org", "organization", "Organisation", "teams", "project"]
   ```

3. **Role Naming**: Use consistent role names within group types

   ```python
   # ✅ GOOD - Standard roles
   organization_roles = ["owner", "admin", "member"]
   team_roles = ["lead", "member", "viewer"]

   # ❌ BAD - Inconsistent
   roles = ["owner", "Owner", "administrator", "adm", "usr"]
   ```

4. **Membership Queries**: Use specialized queries instead of filtering in memory

   ```python
   # ✅ GOOD - Direct query
   teams = await repository.get_user_groups_by_type(user.id, "team")

   # ❌ BAD - Get all then filter
   all_memberships = await repository.get_user_memberships(user.id)
   teams = [m for m in all_memberships if m.group.group_type == "team"]
   ```

5. **Hierarchy Traversal**: Let the repository handle hierarchy queries

   ```python
   # ✅ GOOD - Repository handles it
   hierarchy = await repository.get_group_hierarchy(project.id)

   # ❌ BAD - Manual recursion
   parents = []
   current = await repository.get_by_id(project.id)
   while current.parent_id:
       parent = await repository.get_by_id(current.parent_id)
       parents.append(parent)
       current = parent
   ```

6. **Immutability**: Group and GroupMembership models are immutable

   ```python
   # ✅ GOOD
   updated_group = await repository.update(
       group.id,
       UpdateGroupRequest(description="New description")
   )

   # ❌ BAD
   group.description = "New description"  # Raises FrozenInstanceError!
   ```

## FAQs

### What's the difference between a group and an organization?

A "group" is the generic term for any organizational unit. An "organization" is a specific `group_type`. Other common types include "team", "project", "workspace", etc. The `group_type` field lets you model different kinds of groupings with the same underlying infrastructure.

### Can a group have multiple parents?

No, the current design supports single-parent hierarchies through the `parent_id` field. Each group can have at most one parent, forming a tree structure. For graph-based relationships (multiple parents), consider using metadata or a separate relationship table.

### How do I implement role inheritance?

Use the `get_user_roles_in_hierarchy()` method to get a user's roles across all parent groups. Check permissions at each level:

```python
roles = await repository.get_user_roles_in_hierarchy(user.id, project.id)
# Returns: {project.id: "member", team.id: "admin", org.id: "owner"}

# Check if user is admin in any parent group
is_admin_somewhere = any(
    role_manager.get_group_role_permissions(group_type, role).contains("admin.permission")
    for role in roles.values()
)
```

### Should I use group-level or global roles?

Use both appropriately:

- **Global roles** (from User Port): System-wide permissions (e.g., "super_admin", "user")
- **Group roles** (from Group Port): Context-specific permissions (e.g., "organization owner", "team member")

A user might be a regular "user" globally but an "owner" within their organization.

### Can users belong to multiple groups of the same type?

Yes! A user can be a member of multiple teams, multiple organizations, multiple projects, etc. Use `get_user_groups_by_type()` to get all groups of a specific type the user belongs to.

### How do I delete a group with children?

The port doesn't enforce cascading deletion - that's a business logic decision. Either:

1. Prevent deletion if children exist (recommended)
2. Cascade delete children (requires iteration)
3. Orphan children by setting their `parent_id` to None

```python
# Check for children before deleting
children = await repository.get_child_groups(group.id)
if children:
    raise ValidationError("Cannot delete group with children")
await repository.delete(group.id)
```

### How do I implement custom adapters for GroupRepository?

Implement all abstract methods from both `GroupRepository` and `GroupMembershipRepository`:

```python
from portico.ports.group import GroupRepository, Group, CreateGroupRequest

class CustomGroupRepository(GroupRepository):
    async def create(self, group_data: CreateGroupRequest) -> Group:
        # Your implementation
        pass

    async def get_by_id(self, group_id: UUID) -> Optional[Group]:
        # Your implementation
        pass

    # ... implement all other abstract methods
```

Then inject it through composition:

```python
def group(**config):
    from your_module import CustomGroupRepository
    from portico.kits.group import GroupKit

    def factory(database, events):
        repository = CustomGroupRepository(config["connection"])
        return GroupKit(database, events, config, repository=repository)

    return factory
```
