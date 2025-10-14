"""Test examples for User port documentation."""

from datetime import UTC, datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest

from portico.ports.user import (
    CreateUserRequest,
    SetPasswordRequest,
    UpdateUserRequest,
    User,
    UserRepository,
)


class MockUserRepository(UserRepository):
    """Mock user repository for testing."""

    def __init__(self):
        self.users: dict[UUID, User] = {}
        self.email_index: dict[str, UUID] = {}
        self.username_index: dict[str, UUID] = {}

    async def create(self, user_data: CreateUserRequest) -> Optional[User]:
        """Create a new user. Returns None if email/username exists."""
        # Check for duplicates
        if user_data.email in self.email_index:
            return None
        if user_data.username in self.username_index:
            return None

        user = User(
            id=uuid4(),
            email=user_data.email,
            username=user_data.username,
            global_role=user_data.global_role,
            password_hash=user_data.password_hash,
            password_changed_at=datetime.now(UTC) if user_data.password_hash else None,
        )

        self.users[user.id] = user
        self.email_index[user.email] = user.id
        self.username_index[user.username] = user.id
        return user

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Retrieve a user by ID."""
        return self.users.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by email."""
        user_id = self.email_index.get(email)
        return self.users.get(user_id) if user_id else None

    async def get_by_username(self, username: str) -> Optional[User]:
        """Retrieve a user by username."""
        user_id = self.username_index.get(username)
        return self.users.get(user_id) if user_id else None

    async def update(
        self, user_id: UUID, update_data: UpdateUserRequest
    ) -> Optional[User]:
        """Update an existing user."""
        user = self.users.get(user_id)
        if not user:
            return None

        # Create updated user (User is frozen/immutable)
        updated_user = user.model_copy(
            update={
                "email": update_data.email if update_data.email else user.email,
                "username": update_data.username
                if update_data.username
                else user.username,
                "is_active": update_data.is_active
                if update_data.is_active is not None
                else user.is_active,
                "updated_at": datetime.now(UTC),
            }
        )

        # Update indexes if email/username changed
        if update_data.email and update_data.email != user.email:
            del self.email_index[user.email]
            self.email_index[update_data.email] = user_id

        if update_data.username and update_data.username != user.username:
            del self.username_index[user.username]
            self.username_index[update_data.username] = user_id

        self.users[user_id] = updated_user
        return updated_user

    async def delete(self, user_id: UUID) -> bool:
        """Delete a user by ID."""
        user = self.users.get(user_id)
        if not user:
            return False

        del self.users[user_id]
        del self.email_index[user.email]
        del self.username_index[user.username]
        return True

    async def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List users with pagination."""
        all_users = list(self.users.values())
        return all_users[offset : offset + limit]

    async def set_password(
        self, user_id: UUID, password_request: SetPasswordRequest
    ) -> Optional[User]:
        """Set or update a user's password."""
        user = self.users.get(user_id)
        if not user:
            return None

        updated_user = user.model_copy(
            update={
                "password_hash": password_request.password_hash,
                "password_changed_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        )

        self.users[user_id] = updated_user
        return updated_user

    async def authenticate_user(
        self, username_or_email: str, password_hash: str
    ) -> Optional[User]:
        """Authenticate a user by username/email and password hash."""
        # Try email first
        user = await self.get_by_email(username_or_email)
        if not user:
            # Try username
            user = await self.get_by_username(username_or_email)

        if user and user.password_hash == password_hash and user.is_active:
            return user
        return None

    async def get_user_password_hash(self, user_id: UUID) -> Optional[str]:
        """Get a user's password hash for verification."""
        user = self.users.get(user_id)
        return user.password_hash if user else None

    async def search_users(self, query: str, limit: int = 10) -> List[User]:
        """Search users by email or username."""
        results = []
        query_lower = query.lower()
        for user in self.users.values():
            if (
                query_lower in user.email.lower()
                or query_lower in user.username.lower()
            ):
                results.append(user)
                if len(results) >= limit:
                    break
        return results


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_user_creation():
    """Create a basic user."""
    repo = MockUserRepository()

    # Create user
    user = await repo.create(
        CreateUserRequest(
            email="alice@example.com", username="alice", global_role="user"
        )
    )

    assert user is not None
    assert user.email == "alice@example.com"
    assert user.username == "alice"
    assert user.is_active is True
    assert user.global_role == "user"


# --8<-- [end:basic-usage]


# --8<-- [start:password-hash]
@pytest.mark.asyncio
async def test_create_user_with_password():
    """Create a user with password hash."""
    repo = MockUserRepository()

    # Create user with password hash (would be hashed by auth service)
    user = await repo.create(
        CreateUserRequest(
            email="bob@example.com",
            username="bob",
            password_hash="$argon2id$v=19$m=65536,t=3,p=4$...",
        )
    )

    assert user is not None
    assert user.has_password() is True
    assert user.password_changed_at is not None


# --8<-- [end:password-hash]


# --8<-- [start:retrieve-user]
@pytest.mark.asyncio
async def test_retrieve_user():
    """Retrieve users by ID, email, or username."""
    repo = MockUserRepository()

    # Create user
    created = await repo.create(
        CreateUserRequest(email="charlie@example.com", username="charlie")
    )

    # Retrieve by ID
    user = await repo.get_by_id(created.id)
    assert user is not None
    assert user.email == "charlie@example.com"

    # Retrieve by email
    user = await repo.get_by_email("charlie@example.com")
    assert user is not None
    assert user.username == "charlie"

    # Retrieve by username
    user = await repo.get_by_username("charlie")
    assert user is not None
    assert user.id == created.id


# --8<-- [end:retrieve-user]


# --8<-- [start:update-user]
@pytest.mark.asyncio
async def test_update_user():
    """Update user information."""
    repo = MockUserRepository()

    # Create user
    user = await repo.create(
        CreateUserRequest(email="dave@example.com", username="dave")
    )

    # Update email and username
    updated = await repo.update(
        user.id,
        UpdateUserRequest(email="dave.new@example.com", username="dave_updated"),
    )

    assert updated is not None
    assert updated.email == "dave.new@example.com"
    assert updated.username == "dave_updated"

    # Verify retrieval with new email
    found = await repo.get_by_email("dave.new@example.com")
    assert found is not None
    assert found.id == user.id


# --8<-- [end:update-user]


# --8<-- [start:deactivate-user]
@pytest.mark.asyncio
async def test_deactivate_user():
    """Deactivate a user account."""
    repo = MockUserRepository()

    # Create user
    user = await repo.create(CreateUserRequest(email="eve@example.com", username="eve"))

    # Deactivate
    updated = await repo.update(user.id, UpdateUserRequest(is_active=False))

    assert updated is not None
    assert updated.is_active is False


# --8<-- [end:deactivate-user]


# --8<-- [start:delete-user]
@pytest.mark.asyncio
async def test_delete_user():
    """Delete a user account."""
    repo = MockUserRepository()

    # Create user
    user = await repo.create(
        CreateUserRequest(email="frank@example.com", username="frank")
    )

    # Delete user
    deleted = await repo.delete(user.id)
    assert deleted is True

    # Verify user is gone
    found = await repo.get_by_id(user.id)
    assert found is None

    # Try deleting again
    deleted_again = await repo.delete(user.id)
    assert deleted_again is False


# --8<-- [end:delete-user]


# --8<-- [start:list-users]
@pytest.mark.asyncio
async def test_list_users():
    """List users with pagination."""
    repo = MockUserRepository()

    # Create multiple users
    for i in range(15):
        await repo.create(
            CreateUserRequest(email=f"user{i}@example.com", username=f"user{i}")
        )

    # Get first page
    page1 = await repo.list_users(limit=10, offset=0)
    assert len(page1) == 10

    # Get second page
    page2 = await repo.list_users(limit=10, offset=10)
    assert len(page2) == 5


# --8<-- [end:list-users]


# --8<-- [start:set-password]
@pytest.mark.asyncio
async def test_set_password():
    """Set or update a user's password."""
    repo = MockUserRepository()

    # Create user without password
    user = await repo.create(
        CreateUserRequest(email="grace@example.com", username="grace")
    )
    assert user.has_password() is False

    # Set password
    updated = await repo.set_password(
        user.id, SetPasswordRequest(password_hash="$argon2id$v=19$...")
    )

    assert updated is not None
    assert updated.has_password() is True
    assert updated.password_changed_at is not None


# --8<-- [end:set-password]


# --8<-- [start:authenticate]
@pytest.mark.asyncio
async def test_authenticate_user():
    """Authenticate a user with username/email and password."""
    repo = MockUserRepository()
    password_hash = "$argon2id$v=19$m=65536,t=3,p=4$..."

    # Create user with password
    user = await repo.create(
        CreateUserRequest(
            email="henry@example.com", username="henry", password_hash=password_hash
        )
    )

    # Authenticate with email
    authenticated = await repo.authenticate_user("henry@example.com", password_hash)
    assert authenticated is not None
    assert authenticated.id == user.id

    # Authenticate with username
    authenticated = await repo.authenticate_user("henry", password_hash)
    assert authenticated is not None

    # Wrong password
    authenticated = await repo.authenticate_user("henry", "wrong_password_hash")
    assert authenticated is None


# --8<-- [end:authenticate]


# --8<-- [start:duplicate-email]
@pytest.mark.asyncio
async def test_duplicate_email_prevention():
    """Prevent duplicate email addresses."""
    repo = MockUserRepository()

    # Create first user
    user1 = await repo.create(
        CreateUserRequest(email="duplicate@example.com", username="user1")
    )
    assert user1 is not None

    # Try to create user with same email
    user2 = await repo.create(
        CreateUserRequest(email="duplicate@example.com", username="user2")
    )
    assert user2 is None  # Creation should fail


# --8<-- [end:duplicate-email]
