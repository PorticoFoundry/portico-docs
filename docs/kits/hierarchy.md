# Hierarchy Kit

## Overview

**Purpose**: Transform flat group data into hierarchical tree structures and query ancestor/descendant relationships for organizational charts, permission inheritance, and navigation.

**Key Features**:

- Build complete hierarchy trees from root groups
- Query ancestors (parents, grandparents, etc.) of any group
- Query descendants (children, grandchildren, etc.) of any group
- Support multi-parent groups (matrix organizations)
- Configurable depth limits for large hierarchies
- Cycle detection for multi-parent graphs

**Dependencies**:

- **Injected services**: GroupService (for accessing group data)
- **Port dependencies**: None (uses GroupService methods)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose

# Basic configuration
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.group(),      # Required - provides group data
        compose.hierarchy(),  # Uses GroupService
    ]
)

# Access the hierarchy kit
hierarchy_kit = app.kits["hierarchy"]
hierarchy_service = hierarchy_kit.service

# Build complete hierarchy
roots = await hierarchy_service.build_hierarchy()

# Get ancestors of a group
ancestors = await hierarchy_service.get_ancestors(group_id)

# Get descendants of a group
descendants = await hierarchy_service.get_descendants(group_id)
```

## Core Concepts

### Tree Building

The Hierarchy Kit builds tree structures from flat group data by following parent-child relationships. Each node in the tree is represented as a `HierarchyNode` containing the group data plus its children.

```python
# Build entire hierarchy from all roots
roots = await hierarchy_service.build_hierarchy()

# Build from specific root
division_id = UUID("...")
roots = await hierarchy_service.build_hierarchy(root_ids=[division_id])

# Limit depth for performance
roots = await hierarchy_service.build_hierarchy(max_depth=2)
```

### Ancestor/Descendant Queries

Efficiently query relationships in the hierarchy using breadth-first search:

```python
# Get all parent groups (walking up the hierarchy)
ancestors = await hierarchy_service.get_ancestors(team_id)
# Returns: [Division, Company]

# Get all child groups (walking down the hierarchy)
descendants = await hierarchy_service.get_descendants(division_id)
# Returns: [Team1, Team2, Project1, Project2, ...]

# Include the group itself in results
ancestors = await hierarchy_service.get_ancestors(team_id, include_self=True)
# Returns: [Team, Division, Company]
```

### Multi-Parent Support

The Hierarchy Kit supports matrix organizations where groups can have multiple parents:

```python
# Create cross-functional project with multiple parents
launch_project = await group_service.create_group(
    CreateGroupRequest(
        name="Product Launch 2024",
        group_type="project",
        parent_ids=[engineering.id, marketing.id]  # Multiple parents
    )
)

# When building hierarchy, the project appears under both parents
# When getting ancestors, both parents are returned
```

Multi-parent traversal uses BFS with cycle detection and deduplication to ensure each group appears once in results.

## Configuration

The Hierarchy Kit has no configuration options:

```python
from portico import compose

app = compose.webapp(
    kits=[
        compose.group(),
        compose.hierarchy(),  # No configuration needed
    ]
)
```

## Usage Examples

### Example 1: Displaying Organizational Chart

```python
@app.get("/org/chart")
async def get_org_chart():
    """Get complete organizational hierarchy."""
    hierarchy_service = app.kits["hierarchy"].service

    # Build full tree with depth limit
    roots = await hierarchy_service.build_hierarchy(max_depth=3)

    # Return as JSON for frontend rendering
    return {"hierarchy": [root.dict() for root in roots]}
```

### Example 2: Permission Inheritance Chain

```python
async def get_permission_chain(group_id: UUID) -> List[Group]:
    """Get groups to check for cascading permissions."""
    hierarchy_service = app.kits["hierarchy"].service

    # Check this group + all ancestors
    return await hierarchy_service.get_ancestors(
        group_id,
        include_self=True
    )

# Usage with RBAC
groups_to_check = await get_permission_chain(team_id)
for group in groups_to_check:
    if await rbac_service.check_permission(user_id, "documents.read", group.id):
        return True
```

### Example 3: Breadcrumb Navigation

```python
@app.get("/groups/{group_id}/breadcrumbs")
async def get_breadcrumbs(group_id: UUID):
    """Get breadcrumb navigation for a group."""
    hierarchy_service = app.kits["hierarchy"].service

    # Get path from root to this group
    ancestors = await hierarchy_service.get_ancestors(group_id, include_self=True)
    ancestors.reverse()  # Root → group order

    return {
        "breadcrumbs": [
            {"id": str(g.id), "name": g.name, "type": g.group_type}
            for g in ancestors
        ]
    }
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
# Build hierarchy returns trees of HierarchyNode objects
roots = await hierarchy_service.build_hierarchy()

# Recursively traverse tree
def print_tree(node: HierarchyNode, indent: int = 0):
    print("  " * indent + f"- {node.name} ({node.group_type})")
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(roots[0])
```

## Best Practices

### 1. Use Depth Limits for Large Hierarchies

For organizations with hundreds of groups, always use depth limits to prevent loading entire trees:

```python
# ✅ GOOD - Limit depth for performance
roots = await hierarchy_service.build_hierarchy(max_depth=3)
subtree = await hierarchy_service.get_subtree(division_id, max_depth=2)

# ❌ BAD - Loading entire hierarchy without limits
roots = await hierarchy_service.build_hierarchy()  # Could load 1000+ groups
```

### 2. Load Subtrees Instead of Full Hierarchy

When displaying a specific section of the org chart, load only the needed subtree:

```python
# ✅ GOOD - Load only relevant subtree
subtree = await hierarchy_service.get_subtree(division_id)

# ❌ BAD - Load full hierarchy and filter client-side
roots = await hierarchy_service.build_hierarchy()
# Then search for division node...
```

### 3. Cache Hierarchy Results

Hierarchy queries can be expensive for large organizations. Cache results in your application:

```python
# ✅ GOOD - Cache hierarchy results
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_cached_hierarchy(root_id: str):
    return await hierarchy_service.get_subtree(UUID(root_id))

# ❌ BAD - Rebuild hierarchy on every request
@app.get("/org/chart")
async def get_org_chart():
    roots = await hierarchy_service.build_hierarchy()  # Expensive!
    return {"hierarchy": [r.dict() for r in roots]}
```

### 4. Use Cascading Permissions Instead of Manual Traversal

For permission checks, use RBAC's cascading permissions instead of manually walking the hierarchy:

```python
# ✅ GOOD - Let RBAC handle cascading
can_read = await rbac_service.check_permission(
    user_id,
    "documents.read",
    team_id  # Automatically checks ancestors
)

# ❌ BAD - Manually walk hierarchy for permissions
groups = await hierarchy_service.get_ancestors(team_id, include_self=True)
for group in groups:
    if await rbac_service.check_permission(user_id, "documents.read", group.id):
        return True
```

### 5. Include Self When Building Permission Chains

When checking permissions, always include the group itself in the ancestor list:

```python
# ✅ GOOD - Include group in permission chain
groups = await hierarchy_service.get_ancestors(group_id, include_self=True)

# ❌ BAD - Missing direct group permissions
groups = await hierarchy_service.get_ancestors(group_id)  # Skips group itself
```

### 6. Handle Multi-Parent Groups Carefully

When working with multi-parent groups, remember that ancestors/descendants may contain duplicates from different paths:

```python
# ✅ GOOD - BFS ensures closest ancestors first, no duplicates
ancestors = await hierarchy_service.get_ancestors(multi_parent_group_id)
# Returns each unique ancestor once, ordered by distance

# ❌ BAD - Assuming single-parent structure
parent_id = group.parent_ids[0]  # Breaks for multi-parent groups!
```

### 7. Use get_subtree for Tree Rendering

When rendering hierarchical UIs, use `get_subtree()` to get pre-built tree structures:

```python
# ✅ GOOD - Get tree structure ready for rendering
subtree = await hierarchy_service.get_subtree(division_id)
return subtree.dict()  # Frontend can render directly

# ❌ BAD - Get descendants and rebuild tree manually
descendants = await hierarchy_service.get_descendants(division_id)
# Then manually reconstruct tree structure...
```

## Integration with RBAC

### Cascading Permissions

The Hierarchy Kit works seamlessly with RBAC's cascading permissions. When permissions have `cascades=True`, they automatically flow down the hierarchy:

```python
from portico.kits.rbac.models import CreatePermissionRequest

rbac_service = app.kits["rbac"].service

# Create cascading permission
await rbac_service.create_permission(
    CreatePermissionRequest(
        name="documents.read",
        cascades=True  # Flows down hierarchy
    )
)

# Assign at company level
await rbac_service.assign_group_role(
    user_id=ceo_user_id,
    group_id=company_id,
    role_id=admin_role_id
)

# CEO automatically has permission in all child groups
can_read = await rbac_service.check_permission(
    ceo_user_id,
    "documents.read",
    team_id  # Deep in hierarchy - returns True!
)
```

The RBAC service internally uses `HierarchyService.get_ancestors()` to walk up the hierarchy when checking cascading permissions.

### When to Use Manual Traversal

Use manual `get_ancestors()` / `get_descendants()` calls for:

- Building UIs that display hierarchy structures
- Custom business logic that needs to inspect the hierarchy
- Non-permission queries (e.g., finding all teams under a division)
- Breadcrumbs and navigation trees

Use cascading permissions for:

- Checking permissions (let RBAC handle it)
- Access control (simpler and more maintainable)
- Standard inheritance patterns

## FAQs

### Q: What's the difference between get_descendants and get_subtree?

A: `get_descendants()` returns a flat list of all child groups (children, grandchildren, etc.) ordered by distance. `get_subtree()` returns a tree structure (`HierarchyNode`) with nested children, ready for rendering hierarchical UIs.

```python
# Flat list of all descendants
descendants = await hierarchy_service.get_descendants(division_id)
# Returns: [Team1, Team2, Project1, Project2]

# Tree structure
subtree = await hierarchy_service.get_subtree(division_id)
# Returns: HierarchyNode with nested children property
```

### Q: How do I handle multi-parent groups?

A: The Hierarchy Kit fully supports multi-parent groups using breadth-first search with cycle detection. Each group appears once in results, ordered by distance from the starting point. When building trees, multi-parent groups appear under each parent.

### Q: What happens if there's a cycle in the hierarchy?

A: The Hierarchy Kit detects cycles during traversal and prevents infinite loops. Each group is visited only once, even if it appears multiple times in different paths.

### Q: How do I improve performance for large hierarchies?

A: Use depth limits (`max_depth`) when building trees, load subtrees instead of full hierarchies, and cache results in your application. For very large organizations (1000+ groups), consider lazy loading and pagination.

### Q: Can I customize how the tree is built?

A: The Hierarchy Kit uses a fixed BFS algorithm for consistency and correctness. For custom tree structures, use `get_descendants()` to get all groups and build your own tree structure.

### Q: Do I need to manually walk the hierarchy for permission checks?

A: No! Use RBAC's cascading permissions instead. When a permission has `cascades=True`, the RBAC service automatically checks ancestors. Manual traversal is only needed for non-permission use cases like building UIs.

### Q: How do I find the root groups in my hierarchy?

A: Call `build_hierarchy()` without parameters - it automatically finds all groups without parents and uses them as roots:

```python
roots = await hierarchy_service.build_hierarchy()
# Returns HierarchyNode objects for each root group
```

### Q: What's the order of results from get_ancestors and get_descendants?

A: Both methods return results ordered by distance from the starting group. `get_ancestors()` returns immediate parents first, then grandparents, etc. `get_descendants()` returns immediate children first, then grandchildren, etc.
