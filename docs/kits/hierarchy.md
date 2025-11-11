# Hierarchy Kit

## Overview

The Hierarchy Kit provides services for building and querying hierarchical group structures in Portico applications. It builds on the [Group Port](../ports/group.md) to offer tree-building, ancestor/descendant queries, and hierarchy traversal with multi-parent support.

**Purpose**: Transform flat group data into hierarchical tree structures, query relationships between groups in a hierarchy, and support complex organizational charts with matrix structures.

**Domain**: Organizational hierarchies, tree structures, ancestor/descendant relationships

**Key Capabilities**:

- Build complete hierarchy trees from root groups
- Query ancestors (parents, grandparents, etc.) of any group
- Query descendants (children, grandchildren, etc.) of any group
- Get subtrees starting from specific nodes
- Support multi-parent groups (matrix organizations)
- Configurable depth limits for large hierarchies
- Cycle detection for multi-parent graphs

**Kit Type**: Stateless (no database models, wraps GroupKit)

**When to Use**:

- Displaying organizational charts or team structures
- Finding all parent organizations for permission inheritance
- Showing all teams under a division
- Building navigation trees for nested workspaces
- Implementing cascading permissions down a hierarchy
- Supporting matrix organizations with multi-parent groups

## Dependencies

The Hierarchy Kit depends on the [Group Kit](../ports/group.md) to access group data. When composing your application, ensure Group Kit is included before Hierarchy Kit:

```python
from portico import compose

app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.group(),      # Required - provides group data
        compose.hierarchy(),  # Uses GroupKit
    ]
)
```

## Service Methods

The `HierarchyService` provides methods for building and querying hierarchical structures.

### build_hierarchy()

Build complete hierarchy tree(s) from root groups.

```python
async def build_hierarchy(
    self,
    root_ids: Optional[List[UUID]] = None,
    max_depth: Optional[int] = None
) -> List[HierarchyNode]:
    """Build complete hierarchy tree from root groups.

    Args:
        root_ids: Optional list of specific root group IDs to start from.
                  If None, finds all groups without parents.
        max_depth: Optional maximum depth to traverse. None means unlimited.

    Returns:
        List of HierarchyNode trees, one per root group.
    """
```

**Example - Build complete organization hierarchy:**

```python
from portico import compose

app = compose.webapp(kits=[compose.group(), compose.hierarchy()])

# Build entire hierarchy from all roots
hierarchy_service = app.kits["hierarchy"].service
roots = await hierarchy_service.build_hierarchy()

# Roots is a list of HierarchyNode objects
for root in roots:
    print(f"Root: {root.name} ({root.group_type})")
    print(f"  Children: {len(root.children)}")
    print(f"  Depth: {root.depth}")
```

**Example - Build from specific root:**

```python
# Build hierarchy starting from a specific division
division_id = UUID("...")
roots = await hierarchy_service.build_hierarchy(root_ids=[division_id])

# Returns hierarchy tree rooted at that division
division_tree = roots[0]
```

**Example - Limit depth for performance:**

```python
# Build only 2 levels deep (root + immediate children)
roots = await hierarchy_service.build_hierarchy(max_depth=2)

# Useful for large organizations - prevents loading entire tree
```

### get_ancestors()

Get all ancestor groups walking up the parent hierarchy.

```python
async def get_ancestors(
    self,
    group_id: UUID,
    include_self: bool = False
) -> List[Group]:
    """Get all ancestors of a group (walking up parent hierarchy).

    Uses breadth-first search to handle multi-parent groups.

    Args:
        group_id: Group to find ancestors for
        include_self: Whether to include the group itself in results

    Returns:
        List of ancestor Group objects (parents, grandparents, etc.)
        Ordered by distance: immediate parents first, then grandparents, etc.
    """
```

**Example - Find all parent organizations:**

```python
# Get all ancestors of a team
team_id = UUID("...")
ancestors = await hierarchy_service.get_ancestors(team_id)

# ancestors might be: [Division, Company]
for ancestor in ancestors:
    print(f"Parent: {ancestor.name} ({ancestor.group_type})")
```

**Example - Include group itself:**

```python
# Get group + all ancestors
ancestors = await hierarchy_service.get_ancestors(team_id, include_self=True)

# ancestors might be: [Team, Division, Company]
```

**Example - Check permission inheritance:**

```python
# Find all groups where permissions might be inherited from
async def get_permission_chain(group_id: UUID) -> List[Group]:
    """Get groups to check for cascading permissions."""
    hierarchy_service = app.kits["hierarchy"].service

    # Check this group + all ancestors
    return await hierarchy_service.get_ancestors(
        group_id,
        include_self=True
    )
```

### get_descendants()

Get all descendant groups walking down the child hierarchy.

```python
async def get_descendants(
    self,
    group_id: UUID,
    include_self: bool = False
) -> List[Group]:
    """Get all descendants of a group (walking down child hierarchy).

    Uses breadth-first search.

    Args:
        group_id: Group to find descendants for
        include_self: Whether to include the group itself in results

    Returns:
        List of descendant Group objects (children, grandchildren, etc.)
        Ordered by distance: immediate children first, then grandchildren, etc.
    """
```

**Example - Find all teams under a division:**

```python
# Get all descendants of a division
division_id = UUID("...")
descendants = await hierarchy_service.get_descendants(division_id)

# descendants might include all teams, projects, sub-groups
for descendant in descendants:
    print(f"Child: {descendant.name} ({descendant.group_type})")
```

**Example - Count total groups in subtree:**

```python
# Get count of all groups under a division
division_id = UUID("...")
descendants = await hierarchy_service.get_descendants(
    division_id,
    include_self=True
)

print(f"Total groups in division: {len(descendants)}")
```

### get_subtree()

Get a subtree rooted at a specific group, returned as a HierarchyNode tree.

```python
async def get_subtree(
    self,
    root_id: UUID,
    max_depth: Optional[int] = None
) -> HierarchyNode:
    """Get a subtree starting from a specific group.

    Args:
        root_id: Group ID to use as root of subtree
        max_depth: Optional maximum depth to traverse

    Returns:
        HierarchyNode tree rooted at the specified group

    Raises:
        ResourceNotFoundError: If group not found
    """
```

**Example - Get division subtree:**

```python
# Get tree view of a division and its teams
division_id = UUID("...")
subtree = await hierarchy_service.get_subtree(division_id)

print(f"Division: {subtree.name}")
for team in subtree.children:
    print(f"  Team: {team.name}")
    for project in team.children:
        print(f"    Project: {project.name}")
```

**Example - Render org chart for a specific division:**

```python
async def render_division_chart(division_id: UUID, max_levels: int = 3):
    """Render organizational chart for a division."""
    hierarchy_service = app.kits["hierarchy"].service

    # Get subtree with depth limit
    subtree = await hierarchy_service.get_subtree(division_id, max_depth=max_levels)

    # Render as HTML/JSON for frontend
    return subtree.dict()
```

## Domain Models

### HierarchyNode

Represents a node in a hierarchy tree with children.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | - | Group identifier |
| `name` | `str` | Yes | - | Group name |
| `group_type` | `str` | Yes | - | Group type (company, division, team, etc.) |
| `description` | `Optional[str]` | No | `None` | Group description |
| `metadata` | `Dict[str, Any]` | Yes | `{}` | Custom metadata |
| `parent_ids` | `List[UUID]` | Yes | `[]` | Parent group IDs (supports multi-parent) |
| `children` | `List[HierarchyNode]` | Yes | `[]` | Child nodes in tree |
| `depth` | `int` | Yes | `0` | Depth in tree (0 for roots) |

**Example:**

```python
from portico.kits.hierarchy import HierarchyNode

# Build hierarchy returns trees of HierarchyNode objects
roots = await hierarchy_service.build_hierarchy()

root = roots[0]
print(f"Root: {root.name}")
print(f"Type: {root.group_type}")
print(f"Children: {len(root.children)}")
print(f"Depth: {root.depth}")

# Recursively traverse tree
def print_tree(node: HierarchyNode, indent: int = 0):
    print("  " * indent + f"- {node.name} ({node.group_type})")
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(root)
```

## Configuration

The Hierarchy Kit has minimal configuration:

```python
from dataclasses import dataclass

@dataclass
class HierarchyKitConfig:
    """Configuration for Hierarchy Kit."""
    # Currently no configuration options
    # Reserved for future features (caching, performance tuning)
```

Currently, the Hierarchy Kit has no configuration options but the config class is available for future extensibility.

## Multi-Parent Support

The Hierarchy Kit fully supports multi-parent groups, enabling matrix organizational structures.

### What are Multi-Parent Groups?

Multi-parent groups can belong to multiple parent groups simultaneously. This is common in matrix organizations where employees report to multiple managers, or projects span multiple departments.

**Example: Cross-functional project team:**

```python
from portico.kits.group.models import CreateGroupRequest

group_service = app.kits["group"].service

# Create company
company = await group_service.create_group(
    CreateGroupRequest(name="TechCorp", group_type="company")
)

# Create divisions
engineering = await group_service.create_group(
    CreateGroupRequest(
        name="Engineering",
        group_type="division",
        parent_ids=[company.id]
    )
)

marketing = await group_service.create_group(
    CreateGroupRequest(
        name="Marketing",
        group_type="division",
        parent_ids=[company.id]
    )
)

# Cross-functional project with multiple parents
launch_project = await group_service.create_group(
    CreateGroupRequest(
        name="Product Launch 2024",
        group_type="project",
        parent_ids=[engineering.id, marketing.id]  # Multiple parents!
    )
)
```

### How Multi-Parent Traversal Works

When traversing hierarchies with multi-parent groups:

- **Breadth-first search (BFS)** ensures closest ancestors/descendants are returned first
- **Cycle detection** prevents infinite loops
- **Deduplication** ensures each group appears once in results
- **Tree building** shows multi-parent groups under each parent

**Example hierarchy with multi-parent:**

```
Company
├── Engineering
│   └── Product Launch (multi-parent)
└── Marketing
    └── Product Launch (same group!)
```

When you call `build_hierarchy()`, the Product Launch project appears under both Engineering and Marketing. When you call `get_ancestors(launch_project.id)`, you get both Engineering and Marketing, plus Company.

## Use Cases

### Use Case 1: Displaying Organizational Chart

```python
@app.get("/org/chart")
async def get_org_chart():
    """Get complete organizational hierarchy."""
    hierarchy_service = app.kits["hierarchy"].service

    # Build full tree
    roots = await hierarchy_service.build_hierarchy()

    # Return as JSON for frontend rendering
    return {"hierarchy": [root.dict() for root in roots]}
```

### Use Case 2: Permission Inheritance

```python
async def check_inherited_permission(
    user_id: UUID,
    permission: str,
    group_id: UUID
) -> bool:
    """Check if user has permission in group or any ancestor."""
    rbac_service = app.kits["rbac"].service
    hierarchy_service = app.kits["hierarchy"].service

    # Check group + all ancestors
    groups_to_check = await hierarchy_service.get_ancestors(
        group_id,
        include_self=True
    )

    for group in groups_to_check:
        if await rbac_service.check_permission(user_id, permission, group.id):
            return True

    return False
```

**Note:** With Portico's cascading permissions system (see [Cascading Permissions](#cascading-permissions)), you don't need to manually walk the hierarchy - the RBAC service does it automatically when permissions have `cascades=True`.

### Use Case 3: Finding All Teams Under a Division

```python
@app.get("/divisions/{division_id}/teams")
async def get_division_teams(division_id: UUID):
    """Get all teams under a division."""
    hierarchy_service = app.kits["hierarchy"].service

    # Get all descendants
    descendants = await hierarchy_service.get_descendants(division_id)

    # Filter to only teams
    teams = [g for g in descendants if g.group_type == "team"]

    return {"teams": teams}
```

### Use Case 4: Breadcrumb Navigation

```python
@app.get("/groups/{group_id}/breadcrumbs")
async def get_breadcrumbs(group_id: UUID):
    """Get breadcrumb navigation for a group."""
    hierarchy_service = app.kits["hierarchy"].service
    group_service = app.kits["group"].service

    # Get path from root to this group
    ancestors = await hierarchy_service.get_ancestors(group_id, include_self=True)

    # Reverse to get root → group order
    ancestors.reverse()

    # Format as breadcrumbs
    breadcrumbs = [
        {"id": str(g.id), "name": g.name, "type": g.group_type}
        for g in ancestors
    ]

    return {"breadcrumbs": breadcrumbs}
```

## Cascading Permissions

The Hierarchy Kit works seamlessly with Portico's RBAC cascading permissions system. When permissions have `cascades=True`, they automatically flow down the hierarchy without manual traversal.

### How Cascading Works

1. **Permission Level**: Set `cascades=True` when creating permissions
2. **Group Level**: Groups have `permission_cascade_enabled=True` by default
3. **Automatic Inheritance**: RBAC service walks hierarchy automatically

**Example:**

```python
from portico.kits.rbac.models import CreatePermissionRequest

rbac_service = app.kits["rbac"].service

# Create permission that cascades down hierarchy
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="documents.read",
        description="Read documents",
        cascades=True  # Flows down hierarchy
    )
)

# Assign permission at company level
await rbac_service.assign_group_role(
    user_id=ceo_user_id,
    group_id=company_id,
    role_id=admin_role_id
)

# CEO automatically has documents.read in all child groups
can_read = await rbac_service.check_permission(
    ceo_user_id,
    "documents.read",
    team_id  # Deep in hierarchy - still returns True!
)
```

The RBAC service internally uses `HierarchyService.get_ancestors()` to walk up the hierarchy when checking cascading permissions.

### When to Use Manual Hierarchy Traversal

Use manual `get_ancestors()` / `get_descendants()` calls when:

- **Building UIs** that display hierarchy structures
- **Custom business logic** that needs to inspect the hierarchy
- **Non-permission queries** like finding all teams under a division
- **Breadcrumbs** and navigation trees

Use cascading permissions when:

- **Checking permissions** - Let RBAC service handle it
- **Access control** - Simpler and more maintainable
- **Standard inheritance patterns** - Less code to maintain

## Performance Considerations

### Large Hierarchies

For organizations with hundreds or thousands of groups:

1. **Use depth limits** when building trees:
   ```python
   # Only load 3 levels
   roots = await hierarchy_service.build_hierarchy(max_depth=3)
   ```

2. **Load subtrees** instead of full hierarchy:
   ```python
   # Only load division and below
   subtree = await hierarchy_service.get_subtree(division_id)
   ```

3. **Cache results** in your application:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=100)
   async def get_cached_hierarchy(root_id: str):
       return await hierarchy_service.get_subtree(UUID(root_id))
   ```

### Multi-Parent Groups

Multi-parent groups increase traversal complexity:

- BFS ensures efficient traversal
- Cycle detection prevents infinite loops
- Each group still visited once (deduplication)

For very large multi-parent graphs, consider:
- Limiting depth
- Loading on-demand (lazy loading)
- Caching subtrees at application level

## Testing

The Hierarchy Kit is easy to test since it depends on the Group Kit:

```python
import pytest
from portico import compose
from portico.kits.group.models import CreateGroupRequest

@pytest.mark.asyncio
async def test_build_simple_hierarchy():
    """Test building a 3-level hierarchy."""
    app = compose.webapp(
        database_url="sqlite+aiosqlite:///:memory:",
        kits=[compose.group(), compose.hierarchy()]
    )

    group_service = app.kits["group"].service
    hierarchy_service = app.kits["hierarchy"].service

    # Create hierarchy: company > division > team
    company = await group_service.create_group(
        CreateGroupRequest(name="ACME Corp", group_type="company")
    )
    division = await group_service.create_group(
        CreateGroupRequest(
            name="Engineering",
            group_type="division",
            parent_ids=[company.id]
        )
    )
    team = await group_service.create_group(
        CreateGroupRequest(
            name="Backend",
            group_type="team",
            parent_ids=[division.id]
        )
    )

    # Build hierarchy
    roots = await hierarchy_service.build_hierarchy()

    # Verify structure
    assert len(roots) == 1
    assert roots[0].id == company.id
    assert len(roots[0].children) == 1
    assert roots[0].children[0].id == division.id
    assert len(roots[0].children[0].children) == 1
    assert roots[0].children[0].children[0].id == team.id
```

## Integration with Other Kits

### RBAC Kit

Hierarchy Kit + RBAC Kit enable cascading permissions:

```python
app = compose.webapp(
    kits=[
        compose.user(),
        compose.group(),
        compose.hierarchy(),  # Provides hierarchy traversal
        compose.rbac(group_kit=group_kit),  # Uses hierarchy for cascading
    ]
)
```

RBAC service automatically uses HierarchyService for permission inheritance when checking cascading permissions.

### Group Kit

Hierarchy Kit depends on Group Kit for group data:

```python
# Group Kit manages CRUD
group_service = app.kits["group"].service
new_group = await group_service.create_group(...)

# Hierarchy Kit queries relationships
hierarchy_service = app.kits["hierarchy"].service
ancestors = await hierarchy_service.get_ancestors(new_group.id)
```

## Summary

The Hierarchy Kit provides:

- **Tree building** from flat group data
- **Ancestor/descendant queries** for relationship traversal
- **Multi-parent support** for matrix organizations
- **Integration with RBAC** for cascading permissions
- **Performance controls** via depth limits and subtrees

Use Hierarchy Kit when you need to:
- Display organizational charts
- Implement permission inheritance
- Build navigation trees
- Query group relationships
- Support complex organizational structures

Next steps:
- [Group Port](../ports/group.md) - Understand group data model
- [Compose](../compose.md) - Learn how to wire kits together
- [Philosophy](../philosophy.md) - Understand hexagonal architecture
