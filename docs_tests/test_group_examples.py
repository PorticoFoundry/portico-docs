"""Test examples for Group port documentation."""

from datetime import UTC, datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from portico.ports.group import (
    CreateGroupRequest,
    Group,
    GroupMembership,
    GroupMembershipRepository,
    GroupMembershipRequest,
    GroupRepository,
    UpdateGroupRequest,
)


class MockGroupRepository(GroupRepository):
    """Mock group repository for testing."""

    def __init__(self):
        self.groups: dict[UUID, Group] = {}
        self.name_index: dict[tuple[str, str], UUID] = {}  # (name, type) -> id

    async def create(self, group_data: CreateGroupRequest) -> Group:
        """Create a new group."""
        group = Group(
            id=uuid4(),
            name=group_data.name,
            group_type=group_data.group_type,
            description=group_data.description,
            parent_id=group_data.parent_id,
            metadata=group_data.metadata,
        )

        self.groups[group.id] = group
        self.name_index[(group.name, group.group_type)] = group.id
        return group

    async def get_by_id(self, group_id: UUID) -> Optional[Group]:
        """Retrieve a group by ID."""
        return self.groups.get(group_id)

    async def get_by_name(self, name: str, group_type: str) -> Optional[Group]:
        """Retrieve a group by name within a specific type."""
        group_id = self.name_index.get((name, group_type))
        return self.groups.get(group_id) if group_id else None

    async def update(
        self, group_id: UUID, update_data: UpdateGroupRequest
    ) -> Optional[Group]:
        """Update an existing group."""
        group = self.groups.get(group_id)
        if not group:
            return None

        # Create updated group (Group is frozen/immutable)
        updated_group = group.model_copy(
            update={
                "name": update_data.name if update_data.name else group.name,
                "description": update_data.description
                if update_data.description is not None
                else group.description,
                "is_active": update_data.is_active
                if update_data.is_active is not None
                else group.is_active,
                "metadata": update_data.metadata
                if update_data.metadata is not None
                else group.metadata,
                "updated_at": datetime.now(UTC),
            }
        )

        # Update name index if name changed
        if update_data.name and update_data.name != group.name:
            del self.name_index[(group.name, group.group_type)]
            self.name_index[(update_data.name, group.group_type)] = group_id

        self.groups[group_id] = updated_group
        return updated_group

    async def delete(self, group_id: UUID) -> bool:
        """Delete a group by ID."""
        group = self.groups.get(group_id)
        if not group:
            return False

        del self.groups[group_id]
        del self.name_index[(group.name, group.group_type)]
        return True

    async def list_groups(
        self, group_type: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Group]:
        """List groups with optional filtering by type and pagination."""
        all_groups = list(self.groups.values())

        if group_type:
            all_groups = [g for g in all_groups if g.group_type == group_type]

        return all_groups[offset : offset + limit]

    async def get_group_hierarchy(self, group_id: UUID) -> List[Group]:
        """Get all parent groups up the hierarchy."""
        hierarchy = []
        current_id = group_id

        while current_id:
            group = self.groups.get(current_id)
            if not group:
                break

            hierarchy.append(group)
            current_id = group.parent_id

        return hierarchy

    async def get_child_groups(self, group_id: UUID) -> List[Group]:
        """Get direct children of a group."""
        return [g for g in self.groups.values() if g.parent_id == group_id]

    async def get_user_roles_in_hierarchy(
        self, user_id: UUID, group_id: UUID
    ) -> Dict[UUID, str]:
        """Get user's roles in group and all parent groups."""
        # This would typically query membership repository
        # For mock, return empty dict
        return {}


class MockGroupMembershipRepository(GroupMembershipRepository):
    """Mock group membership repository for testing."""

    def __init__(self):
        self.memberships: dict[
            tuple[UUID, UUID], GroupMembership
        ] = {}  # (user_id, group_id) -> membership

    async def add_membership(
        self, membership: GroupMembershipRequest
    ) -> GroupMembership:
        """Add a user to a group with a specific role."""
        group_membership = GroupMembership(
            user_id=membership.user_id,
            group_id=membership.group_id,
            role=membership.role,
        )

        key = (membership.user_id, membership.group_id)
        self.memberships[key] = group_membership
        return group_membership

    async def remove_membership(self, user_id: UUID, group_id: UUID) -> bool:
        """Remove a user from a group."""
        key = (user_id, group_id)
        if key in self.memberships:
            del self.memberships[key]
            return True
        return False

    async def update_membership_role(
        self, user_id: UUID, group_id: UUID, new_role: str
    ) -> Optional[GroupMembership]:
        """Update a user's role in a group."""
        key = (user_id, group_id)
        membership = self.memberships.get(key)

        if not membership:
            return None

        updated = membership.model_copy(update={"role": new_role})
        self.memberships[key] = updated
        return updated

    async def get_membership(
        self, user_id: UUID, group_id: UUID
    ) -> Optional[GroupMembership]:
        """Get a specific membership."""
        key = (user_id, group_id)
        return self.memberships.get(key)

    async def get_user_memberships(self, user_id: UUID) -> List[GroupMembership]:
        """Get all groups a user belongs to."""
        return [
            m
            for (uid, gid), m in self.memberships.items()
            if uid == user_id and m.is_active
        ]

    async def get_group_memberships(self, group_id: UUID) -> List[GroupMembership]:
        """Get all members of a group."""
        return [
            m
            for (uid, gid), m in self.memberships.items()
            if gid == group_id and m.is_active
        ]

    async def get_user_groups_by_type(
        self, user_id: UUID, group_type: str
    ) -> List[Group]:
        """Get all groups of a specific type that a user belongs to."""
        # For mock, would need access to group repository
        # Return empty list for simplicity
        return []


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_group_creation():
    """Create a basic group."""
    repo = MockGroupRepository()

    # Create group
    group = await repo.create(
        CreateGroupRequest(
            name="Engineering",
            group_type="organization",
            description="Engineering department",
        )
    )

    assert group is not None
    assert group.name == "Engineering"
    assert group.group_type == "organization"
    assert group.is_active is True


# --8<-- [end:basic-usage]


# --8<-- [start:group-metadata]
@pytest.mark.asyncio
async def test_group_with_metadata():
    """Create a group with custom metadata."""
    repo = MockGroupRepository()

    # Create group with metadata
    group = await repo.create(
        CreateGroupRequest(
            name="Sales Team",
            group_type="team",
            metadata={
                "region": "US-West",
                "budget_code": "SALES-2024",
                "manager_email": "manager@example.com",
            },
        )
    )

    assert group.metadata["region"] == "US-West"
    assert group.metadata["budget_code"] == "SALES-2024"


# --8<-- [end:group-metadata]


# --8<-- [start:hierarchical-groups]
@pytest.mark.asyncio
async def test_hierarchical_groups():
    """Create hierarchical group structure."""
    repo = MockGroupRepository()

    # Create parent group
    company = await repo.create(
        CreateGroupRequest(name="Acme Corp", group_type="organization")
    )

    # Create child department
    engineering = await repo.create(
        CreateGroupRequest(
            name="Engineering", group_type="department", parent_id=company.id
        )
    )

    # Create sub-team
    backend_team = await repo.create(
        CreateGroupRequest(
            name="Backend Team", group_type="team", parent_id=engineering.id
        )
    )

    # Get hierarchy
    hierarchy = await repo.get_group_hierarchy(backend_team.id)
    assert len(hierarchy) == 3
    assert hierarchy[0].name == "Backend Team"
    assert hierarchy[1].name == "Engineering"
    assert hierarchy[2].name == "Acme Corp"


# --8<-- [end:hierarchical-groups]


# --8<-- [start:add-member]
@pytest.mark.asyncio
async def test_add_group_member():
    """Add a user to a group with a role."""
    group_repo = MockGroupRepository()
    membership_repo = MockGroupMembershipRepository()

    # Create group
    group = await group_repo.create(
        CreateGroupRequest(name="Project Alpha", group_type="project")
    )

    # Add member
    user_id = uuid4()
    membership = await membership_repo.add_membership(
        GroupMembershipRequest(user_id=user_id, group_id=group.id, role="member")
    )

    assert membership is not None
    assert membership.user_id == user_id
    assert membership.group_id == group.id
    assert membership.role == "member"


# --8<-- [end:add-member]


# --8<-- [start:update-role]
@pytest.mark.asyncio
async def test_update_member_role():
    """Update a user's role in a group."""
    group_repo = MockGroupRepository()
    membership_repo = MockGroupMembershipRepository()

    # Setup
    group = await group_repo.create(
        CreateGroupRequest(name="Team A", group_type="team")
    )
    user_id = uuid4()
    await membership_repo.add_membership(
        GroupMembershipRequest(user_id=user_id, group_id=group.id, role="member")
    )

    # Update role
    updated = await membership_repo.update_membership_role(user_id, group.id, "admin")

    assert updated is not None
    assert updated.role == "admin"


# --8<-- [end:update-role]


# --8<-- [start:list-members]
@pytest.mark.asyncio
async def test_list_group_members():
    """List all members of a group."""
    group_repo = MockGroupRepository()
    membership_repo = MockGroupMembershipRepository()

    # Create group
    group = await group_repo.create(
        CreateGroupRequest(name="Team B", group_type="team")
    )

    # Add multiple members
    for i in range(5):
        await membership_repo.add_membership(
            GroupMembershipRequest(
                user_id=uuid4(), group_id=group.id, role="member" if i < 4 else "admin"
            )
        )

    # List members
    members = await membership_repo.get_group_memberships(group.id)
    assert len(members) == 5

    # Count admins
    admins = [m for m in members if m.role == "admin"]
    assert len(admins) == 1


# --8<-- [end:list-members]


# --8<-- [start:user-groups]
@pytest.mark.asyncio
async def test_get_user_groups():
    """Get all groups a user belongs to."""
    group_repo = MockGroupRepository()
    membership_repo = MockGroupMembershipRepository()

    user_id = uuid4()

    # Create multiple groups and add user to each
    for i in range(3):
        group = await group_repo.create(
            CreateGroupRequest(name=f"Team {i}", group_type="team")
        )
        await membership_repo.add_membership(
            GroupMembershipRequest(user_id=user_id, group_id=group.id, role="member")
        )

    # Get user's groups
    memberships = await membership_repo.get_user_memberships(user_id)
    assert len(memberships) == 3


# --8<-- [end:user-groups]


# --8<-- [start:remove-member]
@pytest.mark.asyncio
async def test_remove_group_member():
    """Remove a user from a group."""
    group_repo = MockGroupRepository()
    membership_repo = MockGroupMembershipRepository()

    # Setup
    group = await group_repo.create(
        CreateGroupRequest(name="Team C", group_type="team")
    )
    user_id = uuid4()
    await membership_repo.add_membership(
        GroupMembershipRequest(user_id=user_id, group_id=group.id, role="member")
    )

    # Remove member
    removed = await membership_repo.remove_membership(user_id, group.id)
    assert removed is True

    # Verify removal
    membership = await membership_repo.get_membership(user_id, group.id)
    assert membership is None


# --8<-- [end:remove-member]


# --8<-- [start:list-by-type]
@pytest.mark.asyncio
async def test_list_groups_by_type():
    """List groups filtered by type."""
    repo = MockGroupRepository()

    # Create groups of different types
    await repo.create(CreateGroupRequest(name="Org 1", group_type="organization"))
    await repo.create(CreateGroupRequest(name="Org 2", group_type="organization"))
    await repo.create(CreateGroupRequest(name="Team 1", group_type="team"))
    await repo.create(CreateGroupRequest(name="Project 1", group_type="project"))

    # List only organizations
    orgs = await repo.list_groups(group_type="organization")
    assert len(orgs) == 2

    # List only teams
    teams = await repo.list_groups(group_type="team")
    assert len(teams) == 1


# --8<-- [end:list-by-type]


# --8<-- [start:child-groups]
@pytest.mark.asyncio
async def test_get_child_groups():
    """Get all child groups of a parent."""
    repo = MockGroupRepository()

    # Create parent
    parent = await repo.create(
        CreateGroupRequest(name="Engineering", group_type="department")
    )

    # Create children
    for i in range(3):
        await repo.create(
            CreateGroupRequest(name=f"Team {i}", group_type="team", parent_id=parent.id)
        )

    # Get children
    children = await repo.get_child_groups(parent.id)
    assert len(children) == 3


# --8<-- [end:child-groups]
