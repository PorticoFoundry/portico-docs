# Group Kit

## Overview

**Purpose**: Provide flexible group and membership management with support for hierarchical structures, role-based membership, and event-driven notifications for team collaboration and organizational modeling.

**Key Features**:

- Create and manage groups (organizations, teams, departments, etc.)
- Member management with custom roles
- Hierarchical group structures (multiple parent support)
- Membership tracking with invitation history
- Active/inactive states for groups and members
- Permission cascade support
- Event publishing for all operations
- Flexible metadata storage

**Dependencies**:

- **Injected services**: None
- **Port dependencies**: None (uses repository pattern)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose
from portico.kits.group import CreateGroupRequest

# Basic configuration
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(),
        compose.group(
            max_groups_per_user=100,
            allow_public_groups=True
        ),
    ]
)

# Access the group service
group_service = app.kits["group"].service

# Create a group
group = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering Team",
        group_type="team",
        description="Software engineering team"
    ),
    created_by=user_id
)

# Add a member
member = await group_service.add_member(
    group_id=group.id,
    user_id=member_user_id,
    role="member",
    invited_by=admin_user_id
)
```

## Core Concepts

### Groups

Groups represent collections of users with a common purpose:

```python
from portico.kits.group import CreateGroupRequest

# Create different types of groups
# Organization (top-level)
org = await group_service.create_group(
    CreateGroupRequest(
        name="Acme Corp",
        group_type="organization",
        description="Main organization"
    )
)

# Team within organization
team = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering",
        group_type="team",
        description="Engineering team",
        parent_ids=[org.id],  # Hierarchical structure
        permission_cascade_enabled=True
    )
)

# Sub-team (nested hierarchy)
subteam = await group_service.create_group(
    CreateGroupRequest(
        name="Backend Team",
        group_type="team",
        parent_ids=[team.id],  # Can have multiple parents
        metadata={"location": "San Francisco"}
    )
)
```

**Group types** can be customized for your application:

- `"organization"` - Top-level entity
- `"team"` - Team or department
- `"project"` - Project group
- `"channel"` - Communication channel
- Custom types as needed

### Memberships

Users join groups with specific roles:

```python
# Add member with role
member = await group_service.add_member(
    group_id=team.id,
    user_id=user_id,
    role="admin",  # Custom roles
    invited_by=admin_id
)

# Common role patterns (customizable):
# - "owner" - Group owner
# - "admin" - Administrative access
# - "member" - Regular member
# - "viewer" - Read-only access
# - Custom roles for your app

# Update member's role
updated = await group_service.update_member_role(
    group_id=team.id,
    user_id=user_id,
    new_role="owner"
)

# Remove member
removed = await group_service.remove_member(
    group_id=team.id,
    user_id=user_id
)
```

### Hierarchical Groups

Groups can have multiple parents, enabling flexible organizational structures:

```python
# Create matrix organization structure
backend_team = await group_service.create_group(
    CreateGroupRequest(
        name="Backend Team",
        parent_ids=[engineering_dept.id]
    )
)

platform_project = await group_service.create_group(
    CreateGroupRequest(
        name="Platform Project",
        parent_ids=[backend_team.id, infrastructure_dept.id]
        # Member of both Backend Team and Infrastructure Dept
    )
)

# Permission cascade allows inheriting permissions from parents
```

### Membership Queries

Find groups and members efficiently:

```python
# List all members in a group
members = await group_service.list_members(group_id)

# Find all groups a user belongs to
user_groups = await group_service.list_user_groups(user_id)

# Check specific membership
member = await group_service.get_member(group_id, user_id)
if member:
    print(f"User is {member.role} in group")

# List groups by type
teams = await group_service.list_groups(
    group_type="team",
    limit=50,
    offset=0
)
```

### Group Lifecycle

Manage group states and updates:

```python
# Update group information
from portico.kits.group import UpdateGroupRequest

updated = await group_service.update_group(
    group_id=group.id,
    updates=UpdateGroupRequest(
        name="New Team Name",
        description="Updated description",
        is_active=True,
        metadata={"department": "Engineering"}
    )
)

# Deactivate a group (soft delete)
await group_service.update_group(
    group_id=group.id,
    updates=UpdateGroupRequest(is_active=False)
)

# Delete a group (hard delete)
deleted = await group_service.delete_group(group_id)
```

## Configuration

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_groups_per_user` | `int` | `100` | Maximum groups a user can create |
| `allow_public_groups` | `bool` | `True` | Whether to allow public groups |

**Example Configurations:**

```python
from portico import compose

# Default configuration
compose.group()

# Limit groups per user
compose.group(max_groups_per_user=50)

# Restrict to private groups only
compose.group(allow_public_groups=False)

# Custom limits
compose.group(
    max_groups_per_user=200,
    allow_public_groups=True
)
```

## Usage Examples

### Example 1: Organization Management System

```python
from portico.kits.group import CreateGroupRequest
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@app.post("/organizations")
async def create_organization(
    name: str,
    description: str,
    user = deps.current_user
):
    group_service = deps.webapp.kits["group"].service

    # Create top-level organization
    org = await group_service.create_group(
        CreateGroupRequest(
            name=name,
            group_type="organization",
            description=description
        ),
        created_by=user.id
    )

    # Add creator as owner
    await group_service.add_member(
        group_id=org.id,
        user_id=user.id,
        role="owner"
    )

    return {
        "organization_id": str(org.id),
        "name": org.name,
        "created_at": org.created_at.isoformat()
    }
```

### Example 2: Team Invitation System

```python
@app.post("/teams/{team_id}/invite")
async def invite_to_team(
    team_id: UUID,
    user_email: str,
    role: str,
    current_user = deps.current_user
):
    group_service = deps.webapp.kits["group"].service
    user_service = deps.user_service

    # Verify current user is admin
    member = await group_service.get_member(team_id, current_user.id)
    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(403, "Only admins can invite")

    # Find user to invite
    invited_user = await user_service.get_by_email(user_email)
    if not invited_user:
        raise HTTPException(404, "User not found")

    # Add to group
    new_member = await group_service.add_member(
        group_id=team_id,
        user_id=invited_user.id,
        role=role,
        invited_by=current_user.id
    )

    return {
        "member_id": str(new_member.id),
        "user_email": user_email,
        "role": role,
        "invited_by": current_user.email
    }
```

### Example 3: User Dashboard with Groups

```python
@app.get("/dashboard/groups")
async def list_my_groups(current_user = deps.current_user):
    group_service = deps.webapp.kits["group"].service

    # Get all groups user belongs to
    memberships = await group_service.list_user_groups(current_user.id)

    # Load full group details
    groups_data = []
    for membership in memberships:
        group = await group_service.get_group(membership.group_id)
        if group:
            groups_data.append({
                "group_id": str(group.id),
                "name": group.name,
                "type": group.group_type,
                "role": membership.role,
                "joined_at": membership.joined_at.isoformat()
            })

    return {"groups": groups_data, "count": len(groups_data)}
```

### Example 4: Hierarchical Team Structure

```python
@app.post("/teams/{parent_id}/subteams")
async def create_subteam(
    parent_id: UUID,
    name: str,
    description: str,
    user = deps.current_user
):
    group_service = deps.webapp.kits["group"].service

    # Verify parent exists
    parent = await group_service.get_group(parent_id)
    if not parent:
        raise HTTPException(404, "Parent team not found")

    # Verify user has permission in parent
    member = await group_service.get_member(parent_id, user.id)
    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(403, "Need admin access to create subteam")

    # Create subteam
    subteam = await group_service.create_group(
        CreateGroupRequest(
            name=name,
            group_type="team",
            description=description,
            parent_ids=[parent_id],
            permission_cascade_enabled=True
        ),
        created_by=user.id
    )

    # Add creator as admin
    await group_service.add_member(
        group_id=subteam.id,
        user_id=user.id,
        role="admin"
    )

    return {
        "subteam_id": str(subteam.id),
        "name": subteam.name,
        "parent_id": str(parent_id)
    }
```

### Example 5: Group Member Management

```python
@app.get("/teams/{team_id}/members")
async def list_team_members(
    team_id: UUID,
    user = deps.current_user
):
    group_service = deps.webapp.kits["group"].service

    # Verify user is member
    user_member = await group_service.get_member(team_id, user.id)
    if not user_member:
        raise HTTPException(403, "Not a member of this team")

    # List all members
    members = await group_service.list_members(team_id)

    # Get user details for each member
    user_service = deps.user_service
    members_data = []
    for member in members:
        user_info = await user_service.get_by_id(member.user_id)
        if user_info:
            members_data.append({
                "user_id": str(member.user_id),
                "email": user_info.email,
                "username": user_info.username,
                "role": member.role,
                "joined_at": member.joined_at.isoformat()
            })

    return {"members": members_data, "count": len(members_data)}
```

## Domain Models

### Group

Represents a group entity.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | Auto | Unique group identifier |
| `name` | `str` | - | Group name |
| `group_type` | `str` | `"organization"` | Type of group (customizable) |
| `description` | `str \| None` | `None` | Optional description |
| `parent_ids` | `List[UUID]` | `[]` | Parent group IDs (hierarchical) |
| `is_active` | `bool` | `True` | Whether group is active |
| `permission_cascade_enabled` | `bool` | `True` | Whether to cascade permissions to children |
| `metadata` | `Dict[str, str]` | `{}` | Additional metadata |
| `created_at` | `datetime` | Auto | When group was created (UTC) |
| `updated_at` | `datetime` | Auto | When group was last updated (UTC) |

### GroupMember

Represents a user's membership in a group.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | Auto | Unique membership identifier |
| `group_id` | `UUID` | - | Group ID |
| `user_id` | `UUID` | - | User ID |
| `role` | `str` | - | Member's role in group |
| `joined_at` | `datetime` | Auto | When user joined (UTC) |
| `invited_by` | `UUID \| None` | `None` | Who invited this member |
| `is_active` | `bool` | `True` | Whether membership is active |

### CreateGroupRequest

Request model for creating a group.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | - | Group name |
| `group_type` | `str` | `"organization"` | Type of group |
| `description` | `str \| None` | `None` | Optional description |
| `parent_ids` | `List[UUID]` | `[]` | Parent group IDs |
| `permission_cascade_enabled` | `bool` | `True` | Enable permission cascade |
| `metadata` | `Dict[str, str]` | `{}` | Additional metadata |

### UpdateGroupRequest

Request model for updating a group.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str \| None` | New name |
| `description` | `str \| None` | New description |
| `parent_ids` | `List[UUID] \| None` | New parent IDs |
| `is_active` | `bool \| None` | Change active status |
| `permission_cascade_enabled` | `bool \| None` | Change cascade setting |
| `metadata` | `Dict[str, str] \| None` | New metadata |

### AddMemberRequest

Request model for adding a member.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `UUID` | User to add |
| `role` | `str` | Role for member |
| `invited_by` | `UUID \| None` | Who is inviting |

## Database Models

### KitGroupModel

**Table**: `kit_groups`

**Columns**:

- `id`: UUID, primary key
- `name`: String(255)
- `group_type`: String(50), default "organization"
- `description`: Text, nullable
- `is_active`: Boolean, default True
- `permission_cascade_enabled`: Boolean, default True
- `meta`: JSON (dictionary)
- `created_at`: DateTime with timezone
- `updated_at`: DateTime with timezone

### KitGroupMemberModel

**Table**: `kit_group_members`

**Columns**:

- `id`: UUID, primary key
- `group_id`: UUID, foreign key to `kit_groups.id` (cascade delete)
- `user_id`: UUID, foreign key to `users.id` (cascade delete)
- `role`: String(50), default "member"
- `joined_at`: DateTime with timezone
- `invited_by`: UUID, foreign key to `users.id` (set null on delete), nullable
- `is_active`: Boolean, default True

### KitGroupParentModel

**Table**: `kit_group_parents` (junction table)

**Columns**:

- `group_id`: UUID, foreign key to `kit_groups.id` (cascade delete), primary key
- `parent_id`: UUID, foreign key to `kit_groups.id` (cascade delete), primary key

## Events

This kit publishes the following events:

### `GroupCreatedEvent`

**Triggered when**: A group is created.

**Payload**:

```python
{
    "group_id": UUID,
    "name": str,
    "group_type": str,
    "created_by": UUID | None,
    "timestamp": datetime
}
```

**Use cases**: Audit logging, notifications, provisioning resources.

### `GroupUpdatedEvent`

**Triggered when**: A group is updated.

**Payload**:

```python
{
    "group_id": UUID,
    "fields_changed": List[str],  # ["name", "description", ...]
    "timestamp": datetime
}
```

**Use cases**: Audit logging, change notifications, cache invalidation.

### `GroupDeletedEvent`

**Triggered when**: A group is deleted.

**Payload**:

```python
{
    "group_id": UUID,
    "timestamp": datetime
}
```

**Use cases**: Cleanup, audit logging, cascade deletions.

### `MemberAddedEvent`

**Triggered when**: A member is added to a group.

**Payload**:

```python
{
    "group_id": UUID,
    "user_id": UUID,
    "role": str,
    "invited_by": UUID | None,
    "timestamp": datetime
}
```

**Use cases**: Welcome emails, notifications, access provisioning.

### `MemberRemovedEvent`

**Triggered when**: A member is removed from a group.

**Payload**:

```python
{
    "group_id": UUID,
    "user_id": UUID,
    "timestamp": datetime
}
```

**Use cases**: Access revocation, notifications, audit logging.

### `MemberRoleChangedEvent`

**Triggered when**: A member's role is changed.

**Payload**:

```python
{
    "group_id": UUID,
    "user_id": UUID,
    "old_role": str,
    "new_role": str,
    "timestamp": datetime
}
```

**Use cases**: Permission updates, notifications, audit logging.

## Best Practices

### 1. Use Meaningful Group Types

Define clear group types for your application:

```python
# ✅ GOOD - Clear, consistent types
CreateGroupRequest(name="Engineering", group_type="department")
CreateGroupRequest(name="Project Alpha", group_type="project")
CreateGroupRequest(name="Backend Team", group_type="team")

# ❌ BAD - Inconsistent types
CreateGroupRequest(name="Engineering", group_type="eng")
CreateGroupRequest(name="Project Alpha", group_type="proj")
CreateGroupRequest(name="Backend", group_type="TEAM")  # Inconsistent case
```

### 2. Implement Role-Based Access Control

Check roles before allowing operations:

```python
# ✅ GOOD - Role-based permission check
async def delete_team(team_id: UUID, user_id: UUID):
    member = await group_service.get_member(team_id, user_id)
    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(403, "Only owners/admins can delete teams")

    await group_service.delete_group(team_id)

# ❌ BAD - No permission check
async def delete_team(team_id: UUID, user_id: UUID):
    await group_service.delete_group(team_id)
    # Any member can delete!
```

### 3. Use Hierarchies for Organization Structure

Model real organizational structures with parent-child relationships:

```python
# ✅ GOOD - Clear hierarchy
company = await group_service.create_group(
    CreateGroupRequest(name="Acme Corp", group_type="organization")
)

engineering = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering",
        group_type="department",
        parent_ids=[company.id]
    )
)

backend_team = await group_service.create_group(
    CreateGroupRequest(
        name="Backend Team",
        group_type="team",
        parent_ids=[engineering.id]
    )
)

# ❌ BAD - Flat structure when hierarchy makes sense
# All teams at same level, no organization
```

### 4. Track Invitation History

Use `invited_by` to maintain audit trail:

```python
# ✅ GOOD - Track who invited
await group_service.add_member(
    group_id=team_id,
    user_id=new_user_id,
    role="member",
    invited_by=admin_id  # Audit trail
)

# ❌ BAD - No invitation tracking
await group_service.add_member(
    group_id=team_id,
    user_id=new_user_id,
    role="member",
    invited_by=None  # Lost context
)
```

### 5. Use Metadata for Custom Attributes

Store additional information in metadata:

```python
# ✅ GOOD - Structured metadata
CreateGroupRequest(
    name="Project Alpha",
    group_type="project",
    metadata={
        "budget": "500000",
        "start_date": "2024-01-01",
        "manager_email": "manager@example.com",
        "status": "active"
    }
)

# ❌ BAD - Putting structured data in description
CreateGroupRequest(
    name="Project Alpha",
    description="Budget: 500000, Start: 2024-01-01, ..."
    # Hard to query or parse
)
```

### 6. Handle Member Removal Gracefully

Check membership before operations:

```python
# ✅ GOOD - Verify membership exists
member = await group_service.get_member(group_id, user_id)
if member:
    await group_service.remove_member(group_id, user_id)
    return {"success": True}
else:
    return {"success": False, "error": "Not a member"}

# ❌ BAD - Assume removal always works
await group_service.remove_member(group_id, user_id)
return {"success": True}  # But was user even a member?
```

### 7. Use Soft Deletes When Appropriate

Deactivate instead of deleting for audit trail:

```python
# ✅ GOOD - Soft delete preserves history
await group_service.update_group(
    group_id,
    UpdateGroupRequest(is_active=False)
)
# Group still exists in database for audit

# ❌ BAD - Hard delete loses all history
await group_service.delete_group(group_id)
# All references lost, breaks foreign keys
```

## Security Considerations

### Access Control

Always verify membership and roles before operations:

- Check if user is member before showing group data
- Verify role (owner/admin) before allowing destructive operations
- Implement permission checks at application layer

### Membership Validation

Validate invitations and role assignments:

- Verify inviter has permission to add members
- Validate that assigned roles are valid for your application
- Prevent self-promotion to higher roles

### Hierarchical Permissions

Consider permission inheritance in hierarchies:

- Use `permission_cascade_enabled` to control inheritance
- Be careful with multi-parent groups
- Document your permission model clearly

## FAQs

### Q: How do I implement admin roles that can manage any group?

A: Implement at application layer with global role check:

```python
async def can_manage_group(user_id: UUID, group_id: UUID) -> bool:
    # Check if user is global admin
    user = await user_service.get_by_id(user_id)
    if user.global_role == "admin":
        return True

    # Check if user is group owner/admin
    member = await group_service.get_member(group_id, user_id)
    return member and member.role in ["owner", "admin"]
```

### Q: Can a user be in multiple groups?

A: Yes! Users can belong to unlimited groups. Use `list_user_groups(user_id)` to get all memberships.

### Q: How do I prevent users from creating too many groups?

A: Check count before creation:

```python
user_groups = await group_service.list_user_groups(user_id)
if len(user_groups) >= config.max_groups_per_user:
    raise ValidationError("Maximum groups reached")
```

### Q: What happens to members when a group is deleted?

A: Members are cascade deleted (foreign key constraint). Consider soft delete instead to preserve history.

### Q: How do I implement group invitations with pending status?

A: Use `is_active=False` for pending invitations:

```python
# Create pending invitation
member = await group_service.add_member(
    group_id=group_id,
    user_id=user_id,
    role="member"
)
await group_service.repository.update_member_active_status(
    group_id, user_id, is_active=False
)

# User accepts invitation
await group_service.repository.update_member_active_status(
    group_id, user_id, is_active=True
)
```

### Q: Can groups have multiple parents?

A: Yes! `parent_ids` is a list supporting multiple parents for matrix organizations or cross-functional teams.

### Q: How do I search for groups by name?

A: Use `get_group_by_name(name, group_type)` for exact match. For partial search, extend the repository with custom queries.

### Q: What's the difference between `group_type` and metadata?

A: `group_type` is for categorization (organization, team, project). Metadata is for custom attributes specific to your application.

### Q: How do I implement group-level permissions?

A: Integrate with RBAC Kit:

```python
# Check group permission
has_permission = await rbac_service.check_group_permission(
    user_id=user.id,
    group_id=group.id,
    permission="files.write"
)
```

### Q: Can I rename a group?

A: Yes, use `update_group()` with new name:

```python
await group_service.update_group(
    group_id,
    UpdateGroupRequest(name="New Team Name")
)
```
