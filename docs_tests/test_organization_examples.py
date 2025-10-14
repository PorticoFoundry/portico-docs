"""Test examples for Organization documentation.

This module tests code examples from organization-related documentation to ensure
they remain accurate and working.
"""

from uuid import uuid4

# --8<-- [start:imports]
from portico.ports.organization import (
    HierarchyNode,
    OrganizationalUnit,
    PermissionAnalysis,
    PermissionMatrixEntry,
    UserPermissionInfo,
)

# --8<-- [end:imports]


# --8<-- [start:hierarchy-node-basic]
def test_hierarchy_node_basic():
    """Basic hierarchy node creation."""
    company_id = uuid4()

    company = HierarchyNode(
        id=company_id,
        name="Acme Corp",
        type="company",
        description="Parent company",
        parent_id=None,  # Root node
        members_count=0,
    )

    assert company.id == company_id
    assert company.name == "Acme Corp"
    assert company.type == "company"
    assert company.parent_id is None
    assert company.members_count == 0
    assert company.children == []


# --8<-- [end:hierarchy-node-basic]


# --8<-- [start:hierarchy-node-with-children]
def test_hierarchy_node_with_children():
    """Hierarchy node with children."""
    company_id = uuid4()
    engineering_id = uuid4()
    sales_id = uuid4()

    # Create child nodes
    engineering = HierarchyNode(
        id=engineering_id,
        name="Engineering",
        type="department",
        parent_id=company_id,
        members_count=25,
    )

    sales = HierarchyNode(
        id=sales_id,
        name="Sales",
        type="department",
        parent_id=company_id,
        members_count=15,
    )

    # Create parent with children
    company = HierarchyNode(
        id=company_id,
        name="Acme Corp",
        type="company",
        parent_id=None,
        children=[engineering, sales],
        members_count=0,
    )

    assert len(company.children) == 2
    assert company.children[0].name == "Engineering"
    assert company.children[1].name == "Sales"
    assert company.children[0].parent_id == company_id


# --8<-- [end:hierarchy-node-with-children]


# --8<-- [start:hierarchy-node-metadata]
def test_hierarchy_node_metadata():
    """Hierarchy node with metadata."""
    team_id = uuid4()

    team = HierarchyNode(
        id=team_id,
        name="Platform Team",
        type="team",
        description="Backend platform development",
        metadata={
            "location": "San Francisco",
            "cost_center": "ENG-001",
            "manager_id": str(uuid4()),
            "budget": 500000,
            "tags": ["engineering", "platform", "backend"],
        },
        members_count=8,
    )

    assert team.metadata["location"] == "San Francisco"
    assert team.metadata["cost_center"] == "ENG-001"
    assert team.metadata["budget"] == 500000
    assert "platform" in team.metadata["tags"]


# --8<-- [end:hierarchy-node-metadata]


# --8<-- [start:hierarchy-tree-structure]
def test_hierarchy_tree_structure():
    """Multi-level hierarchy tree."""
    company_id = uuid4()
    division_id = uuid4()
    dept_id = uuid4()
    team_id = uuid4()

    # Level 4: Team
    team = HierarchyNode(
        id=team_id,
        name="Backend Team",
        type="team",
        parent_id=dept_id,
        members_count=5,
    )

    # Level 3: Department
    department = HierarchyNode(
        id=dept_id,
        name="Engineering",
        type="department",
        parent_id=division_id,
        children=[team],
        members_count=20,
    )

    # Level 2: Division
    division = HierarchyNode(
        id=division_id,
        name="Technology",
        type="division",
        parent_id=company_id,
        children=[department],
        members_count=50,
    )

    # Level 1: Company (root)
    company = HierarchyNode(
        id=company_id,
        name="Acme Corp",
        type="company",
        parent_id=None,
        children=[division],
        members_count=100,
    )

    # Verify hierarchy
    assert company.parent_id is None
    assert len(company.children) == 1
    assert company.children[0].name == "Technology"
    assert company.children[0].children[0].name == "Engineering"
    assert company.children[0].children[0].children[0].name == "Backend Team"


# --8<-- [end:hierarchy-tree-structure]


# --8<-- [start:hierarchy-node-direct-members]
def test_hierarchy_node_direct_members():
    """Hierarchy node with direct members."""
    team_id = uuid4()
    user1_id = uuid4()
    user2_id = uuid4()

    team = HierarchyNode(
        id=team_id,
        name="Frontend Team",
        type="team",
        members_count=2,
        direct_members=[
            {
                "user_id": str(user1_id),
                "username": "alice",
                "email": "alice@example.com",
                "role": "senior_engineer",
            },
            {
                "user_id": str(user2_id),
                "username": "bob",
                "email": "bob@example.com",
                "role": "engineer",
            },
        ],
    )

    assert team.members_count == 2
    assert len(team.direct_members) == 2
    assert team.direct_members[0]["username"] == "alice"
    assert team.direct_members[1]["role"] == "engineer"


# --8<-- [end:hierarchy-node-direct-members]


# --8<-- [start:user-permission-info-basic]
def test_user_permission_info_basic():
    """Basic user permission information."""
    user_id = uuid4()

    user_perms = UserPermissionInfo(
        user_id=user_id,
        username="alice",
        email="alice@example.com",
        global_role="user",
        direct_permissions=["documents.read", "documents.write"],
        effective_permissions=["documents.read", "documents.write", "users.read"],
    )

    assert user_perms.user_id == user_id
    assert user_perms.username == "alice"
    assert user_perms.global_role == "user"
    assert len(user_perms.direct_permissions) == 2
    assert len(user_perms.effective_permissions) == 3
    assert "documents.read" in user_perms.direct_permissions


# --8<-- [end:user-permission-info-basic]


# --8<-- [start:user-permission-info-with-groups]
def test_user_permission_info_with_groups():
    """User permission info with group memberships."""
    user_id = uuid4()
    group1_id = uuid4()
    group2_id = uuid4()

    user_perms = UserPermissionInfo(
        user_id=user_id,
        username="bob",
        email="bob@example.com",
        global_role="user",
        group_memberships=[
            {
                "group_id": str(group1_id),
                "group_name": "Engineering",
                "role": "member",
                "permissions": ["code.read", "code.write"],
            },
            {
                "group_id": str(group2_id),
                "group_name": "Admins",
                "role": "admin",
                "permissions": ["users.manage", "groups.manage"],
            },
        ],
        direct_permissions=["documents.read"],
        inherited_permissions=[
            "code.read",
            "code.write",
            "users.manage",
            "groups.manage",
        ],
        effective_permissions=[
            "documents.read",
            "code.read",
            "code.write",
            "users.manage",
            "groups.manage",
        ],
    )

    assert len(user_perms.group_memberships) == 2
    assert user_perms.group_memberships[0]["group_name"] == "Engineering"
    assert user_perms.group_memberships[1]["role"] == "admin"
    assert len(user_perms.inherited_permissions) == 4
    assert len(user_perms.effective_permissions) == 5


# --8<-- [end:user-permission-info-with-groups]


# --8<-- [start:permission-inheritance]
def test_permission_inheritance():
    """Permission inheritance tracking."""
    user_id = uuid4()

    user_perms = UserPermissionInfo(
        user_id=user_id,
        username="charlie",
        email="charlie@example.com",
        global_role="user",
        direct_permissions=["documents.read"],
        inherited_permissions=["projects.read", "projects.write"],
        effective_permissions=["documents.read", "projects.read", "projects.write"],
    )

    # Verify separation of direct vs inherited
    assert "documents.read" in user_perms.direct_permissions
    assert "documents.read" not in user_perms.inherited_permissions

    assert "projects.read" in user_perms.inherited_permissions
    assert "projects.read" not in user_perms.direct_permissions

    # All permissions appear in effective
    assert all(
        perm in user_perms.effective_permissions
        for perm in user_perms.direct_permissions + user_perms.inherited_permissions
    )


# --8<-- [end:permission-inheritance]


# --8<-- [start:permission-matrix-entry]
def test_permission_matrix_entry():
    """Permission matrix entry."""
    user_id = uuid4()
    group_id = uuid4()

    entry = PermissionMatrixEntry(
        user_id=user_id,
        user_name="Alice Smith",
        group_id=group_id,
        group_name="Engineering",
        role="senior_engineer",
        permissions=["code.read", "code.write", "code.review"],
        is_inherited=False,
    )

    assert entry.user_name == "Alice Smith"
    assert entry.group_name == "Engineering"
    assert entry.role == "senior_engineer"
    assert len(entry.permissions) == 3
    assert entry.is_inherited is False
    assert entry.inheritance_path is None


# --8<-- [end:permission-matrix-entry]


# --8<-- [start:permission-matrix-inherited]
def test_permission_matrix_inherited():
    """Permission matrix entry with inheritance."""
    user_id = uuid4()
    group_id = uuid4()

    entry = PermissionMatrixEntry(
        user_id=user_id,
        user_name="Bob Jones",
        group_id=group_id,
        group_name="Platform Team",
        role="member",
        permissions=["projects.read", "projects.write"],
        is_inherited=True,
        inheritance_path=["Engineering", "Backend Division", "Platform Team"],
    )

    assert entry.is_inherited is True
    assert entry.inheritance_path is not None
    assert len(entry.inheritance_path) == 3
    assert entry.inheritance_path[0] == "Engineering"
    assert entry.inheritance_path[-1] == "Platform Team"


# --8<-- [end:permission-matrix-inherited]


# --8<-- [start:organizational-unit]
def test_organizational_unit():
    """Basic organizational unit."""
    unit_id = uuid4()
    parent_id = uuid4()

    unit = OrganizationalUnit(
        id=unit_id,
        name="Backend Team",
        type="team",
        parent_id=parent_id,
        level=2,
    )

    assert unit.id == unit_id
    assert unit.name == "Backend Team"
    assert unit.type == "team"
    assert unit.parent_id == parent_id
    assert unit.level == 2


# --8<-- [end:organizational-unit]


# --8<-- [start:organizational-unit-root]
def test_organizational_unit_root():
    """Root organizational unit."""
    company_id = uuid4()

    company = OrganizationalUnit(
        id=company_id,
        name="Acme Corp",
        type="company",
        parent_id=None,
        level=0,
    )

    assert company.parent_id is None
    assert company.level == 0


# --8<-- [end:organizational-unit-root]


# --8<-- [start:permission-analysis]
def test_permission_analysis():
    """Permission analysis results."""
    user1_id = uuid4()
    user2_id = uuid4()
    group1_id = uuid4()
    group2_id = uuid4()

    analysis = PermissionAnalysis(
        total_users=2,
        total_groups=2,
        permission_matrix=[
            PermissionMatrixEntry(
                user_id=user1_id,
                user_name="Alice",
                group_id=group1_id,
                group_name="Engineering",
                role="admin",
                permissions=["code.read", "code.write", "users.manage"],
            ),
            PermissionMatrixEntry(
                user_id=user2_id,
                user_name="Bob",
                group_id=group2_id,
                group_name="Sales",
                role="member",
                permissions=["deals.read", "deals.write"],
            ),
        ],
        unique_permissions=[
            "code.read",
            "code.write",
            "users.manage",
            "deals.read",
            "deals.write",
        ],
        role_distribution={"admin": 1, "member": 1},
        inheritance_stats={
            "total_inherited_permissions": 5,
            "users_with_inherited_perms": 1,
            "avg_inheritance_depth": 2.0,
        },
    )

    assert analysis.total_users == 2
    assert analysis.total_groups == 2
    assert len(analysis.permission_matrix) == 2
    assert len(analysis.unique_permissions) == 5
    assert analysis.role_distribution["admin"] == 1
    assert analysis.inheritance_stats["total_inherited_permissions"] == 5


# --8<-- [end:permission-analysis]


# --8<-- [start:permission-analysis-role-distribution]
def test_permission_analysis_role_distribution():
    """Analyze role distribution."""
    analysis = PermissionAnalysis(
        total_users=10,
        total_groups=3,
        permission_matrix=[],
        unique_permissions=[],
        role_distribution={
            "admin": 2,
            "senior_engineer": 3,
            "engineer": 4,
            "member": 1,
        },
    )

    # Calculate percentages
    total = analysis.total_users
    admin_pct = (analysis.role_distribution["admin"] / total) * 100
    engineer_pct = (analysis.role_distribution["engineer"] / total) * 100

    assert admin_pct == 20.0
    assert engineer_pct == 40.0
    assert sum(analysis.role_distribution.values()) == total


# --8<-- [end:permission-analysis-role-distribution]


# --8<-- [start:organizational-types]
def test_organizational_types():
    """Different organizational unit types."""
    company = HierarchyNode(
        id=uuid4(),
        name="Acme Corp",
        type="company",
        parent_id=None,
    )

    division = HierarchyNode(
        id=uuid4(),
        name="Technology",
        type="division",
        parent_id=company.id,
    )

    department = HierarchyNode(
        id=uuid4(),
        name="Engineering",
        type="department",
        parent_id=division.id,
    )

    team = HierarchyNode(
        id=uuid4(),
        name="Platform",
        type="team",
        parent_id=department.id,
    )

    project = HierarchyNode(
        id=uuid4(),
        name="Auth Service",
        type="project",
        parent_id=team.id,
    )

    assert company.type == "company"
    assert division.type == "division"
    assert department.type == "department"
    assert team.type == "team"
    assert project.type == "project"


# --8<-- [end:organizational-types]


# --8<-- [start:empty-hierarchy-node]
def test_empty_hierarchy_node():
    """Hierarchy node with minimal fields."""
    node = HierarchyNode(
        id=uuid4(),
        name="New Unit",
        type="team",
    )

    # Default values
    assert node.description is None
    assert node.metadata == {}
    assert node.parent_id is None
    assert node.children == []
    assert node.members_count == 0
    assert node.direct_members == []


# --8<-- [end:empty-hierarchy-node]


# --8<-- [start:user-permission-defaults]
def test_user_permission_defaults():
    """User permission info with defaults."""
    user_id = uuid4()

    user_perms = UserPermissionInfo(
        user_id=user_id,
        username="newuser",
        email="newuser@example.com",
        global_role="user",
    )

    # Default empty lists
    assert user_perms.group_memberships == []
    assert user_perms.effective_permissions == []
    assert user_perms.inherited_permissions == []
    assert user_perms.direct_permissions == []


# --8<-- [end:user-permission-defaults]


# --8<-- [start:permission-matrix-multiple-groups]
def test_permission_matrix_multiple_groups():
    """User with permissions in multiple groups."""
    user_id = uuid4()
    eng_group_id = uuid4()
    admin_group_id = uuid4()

    matrix_entries = [
        PermissionMatrixEntry(
            user_id=user_id,
            user_name="Alice",
            group_id=eng_group_id,
            group_name="Engineering",
            role="member",
            permissions=["code.read", "code.write"],
            is_inherited=False,
        ),
        PermissionMatrixEntry(
            user_id=user_id,
            user_name="Alice",
            group_id=admin_group_id,
            group_name="Admins",
            role="admin",
            permissions=["users.manage", "groups.manage", "system.admin"],
            is_inherited=False,
        ),
    ]

    # Same user, different groups
    assert matrix_entries[0].user_id == matrix_entries[1].user_id
    assert matrix_entries[0].group_id != matrix_entries[1].group_id

    # Different permissions per group
    assert len(matrix_entries[0].permissions) == 2
    assert len(matrix_entries[1].permissions) == 3

    # Collect all permissions
    all_perms = matrix_entries[0].permissions + matrix_entries[1].permissions
    assert len(all_perms) == 5


# --8<-- [end:permission-matrix-multiple-groups]


# --8<-- [start:hierarchy-depth-calculation]
def test_hierarchy_depth_calculation():
    """Calculate hierarchy depth."""
    # Root level
    company = OrganizationalUnit(id=uuid4(), name="Company", type="company", level=0)

    # Level 1
    division = OrganizationalUnit(
        id=uuid4(),
        name="Division",
        type="division",
        parent_id=company.id,
        level=1,
    )

    # Level 2
    department = OrganizationalUnit(
        id=uuid4(),
        name="Department",
        type="department",
        parent_id=division.id,
        level=2,
    )

    # Level 3
    team = OrganizationalUnit(
        id=uuid4(),
        name="Team",
        type="team",
        parent_id=department.id,
        level=3,
    )

    units = [company, division, department, team]
    max_depth = max(unit.level for unit in units)

    assert max_depth == 3
    assert company.level == 0
    assert team.level == 3


# --8<-- [end:hierarchy-depth-calculation]
