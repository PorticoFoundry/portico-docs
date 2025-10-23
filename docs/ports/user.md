# User Port

## Overview

**Purpose**: Defines interfaces and domain models for user management, authentication, and authorization in Portico applications.

**Domain**: User identity, authentication, role-based access control, and user lifecycle management

**Key Capabilities**:

- User CRUD operations (create, read, update, delete)
- Password-based authentication with secure hashing
- Global role-based authorization
- User search and listing with pagination
- Email and username uniqueness enforcement

**Port Type**: Repository

## When to Use

Use this port when you need to:

- Implement user registration and account management
- Authenticate users with username/email and password
- Store and manage user profiles
- Implement role-based access control at the global level
- Track user activity with created/updated timestamps

## Architecture Role

**Import Location**: `portico.ports.user`

The User port provides the foundation for authentication and authorization in Portico applications. It defines immutable domain models and repository interfaces that separate business logic from persistence implementation.

## Domain Models

### User

**Description**: Core domain model representing an authenticated user in the system. This is an immutable snapshot of user state at a point in time.

**Location**: `portico.ports.user.User`

**Attributes**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | `uuid4()` | Unique user identifier |
| `email` | `str` | Yes | - | User's email address (unique) |
| `username` | `str` | Yes | - | Username for login (unique) |
| `is_active` | `bool` | Yes | `True` | Whether the account is active |
| `global_role` | `str` | Yes | `"user"` | Global system role (e.g., "user", "admin", "moderator") |
| `password_hash` | `Optional[str]` | No | `None` | Serialized password hash in format: `algorithm$hash$salt$params` |
| `password_changed_at` | `Optional[datetime]` | No | `None` | Timestamp of last password change (UTC) |
| `created_at` | `datetime` | Yes | `now(UTC)` | Account creation timestamp (UTC) |
| `updated_at` | `datetime` | Yes | `now(UTC)` | Last update timestamp (UTC) |

**Configuration**:

- `model_config = ConfigDict(frozen=True)` - User model is immutable (cannot be modified after creation)

**Methods**:

```python
def has_password(self) -> bool:
    """Check if user has a password set.

    Returns:
        True if password_hash is not None, False otherwise
    """
```

**Example**:

```python
from portico.ports.user import User
from datetime import datetime, UTC
from uuid import uuid4

# Create user instance
user = User(
    id=uuid4(),
    email="john@example.com",
    username="john_doe",
    is_active=True,
    global_role="admin",
    password_hash="$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUe",
    password_changed_at=datetime.now(UTC),
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC)
)

# Check if user can authenticate with password
if user.has_password():
    print("User has password authentication enabled")

# User is immutable - this raises an error
# user.email = "new@example.com"  # FrozenInstanceError!
```

### CreateUserRequest

**Description**: Request model for creating a new user. Supports both raw password (to be hashed) and pre-hashed password workflows.

**Location**: `portico.ports.user.CreateUserRequest`

**Attributes**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `email` | `str` | Yes | - | User's email address |
| `username` | `str` | Yes | - | User's username |
| `global_role` | `str` | No | `"user"` | Initial global role |
| `password_hash` | `Optional[str]` | No | `None` | Pre-hashed password (for admin-created users) |
| `password` | `Optional[str]` | No | `None` | Raw password (will be hashed by service) |

**Validation**:

- Email must be unique across all users
- Username must be unique across all users
- Must provide either `password` (raw) or `password_hash` (pre-hashed), or neither for SSO users
- If `password` is provided, it will be hashed by the service layer
- If `password_hash` is provided, it's assumed to be already hashed

**Example**:

```python
from portico.ports.user import CreateUserRequest

# User self-registration with raw password
request = CreateUserRequest(
    email="john@example.com",
    username="john_doe",
    password="securePassword123"
)

# Admin creating user with pre-hashed password
admin_request = CreateUserRequest(
    email="admin@example.com",
    username="admin_user",
    password_hash="$2b$12$...",  # Pre-computed bcrypt hash
    global_role="admin"
)

# SSO user without password
sso_request = CreateUserRequest(
    email="sso_user@example.com",
    username="sso_user"
    # No password or password_hash - will use external auth
)
```

### UpdateUserRequest

**Description**: Request model for updating an existing user. All fields are optional to support partial updates.

**Location**: `portico.ports.user.UpdateUserRequest`

**Attributes**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `email` | `Optional[str]` | No | `None` | New email address |
| `username` | `Optional[str]` | No | `None` | New username |
| `is_active` | `Optional[bool]` | No | `None` | New active status |
| `global_role` | `Optional[str]` | No | `None` | New global role |

**Validation**:

- All fields are optional
- Omitted fields will NOT be updated
- If `email` is provided, it must not conflict with another user
- If `username` is provided, it must not conflict with another user

**Example**:

```python
from portico.ports.user import UpdateUserRequest

# Update email only
request = UpdateUserRequest(email="newemail@example.com")

# Deactivate user
request = UpdateUserRequest(is_active=False)

# Update multiple fields
request = UpdateUserRequest(
    email="updated@example.com",
    username="new_username",
    global_role="moderator"
)

# Empty request is valid (no-op)
request = UpdateUserRequest()
```

### SetPasswordRequest

**Description**: Request model for setting or updating a user's password hash.

**Location**: `portico.ports.user.SetPasswordRequest`

**Attributes**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `password_hash` | `str` | Yes | - | Serialized password hash |

**Example**:

```python
from portico.ports.user import SetPasswordRequest

# Called by service after hashing password
request = SetPasswordRequest(
    password_hash="$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUe"
)
```

## Port Interfaces

### UserRepository

**Description**: Abstract interface for user persistence operations. Provides CRUD operations, authentication, and search capabilities.

**Type**: Repository

**Location**: `portico.ports.user.UserRepository`

**Abstract Methods**:

##### create

```python
async def create(user_data: CreateUserRequest) -> Optional[User]
```

Create a new user.

**Parameters**:
- `user_data: CreateUserRequest` - User creation data

**Returns**:
- `Optional[User]` - Created User object, or None if creation failed

**Raises**:
- `ConflictError` - If email or username already exists
- `ValidationError` - If data validation fails

**Example**:

```python
from portico.ports.user import UserRepository, CreateUserRequest

class MyUserRepository(UserRepository):
    async def create(self, user_data: CreateUserRequest) -> Optional[User]:
        # Check for existing email
        if await self.get_by_email(user_data.email):
            raise ConflictError("Email already exists")

        # Check for existing username
        if await self.get_by_username(user_data.username):
            raise ConflictError("Username already exists")

        # Create user in database
        user_model = UserModel(...)
        session.add(user_model)
        await session.commit()

        return user_model.to_domain()
```

##### get_by_id

```python
async def get_by_id(user_id: UUID) -> Optional[User]
```

Retrieve a user by their unique ID.

**Parameters**:
- `user_id: UUID` - User ID to retrieve

**Returns**:
- `Optional[User]` - User object if found, None otherwise

**Example**:

```python
user = await repository.get_by_id(UUID('550e8400-e29b-41d4-a716-446655440000'))
if user:
    print(f"Found user: {user.email}")
```

##### get_by_email

```python
async def get_by_email(email: str) -> Optional[User]
```

Retrieve a user by their email address.

**Parameters**:
- `email: str` - Email address to look up

**Returns**:
- `Optional[User]` - User object if found, None otherwise

**Example**:

```python
user = await repository.get_by_email("john@example.com")
if user:
    print(f"User ID: {user.id}")
```

##### get_by_username

```python
async def get_by_username(username: str) -> Optional[User]
```

Retrieve a user by their username.

**Parameters**:
- `username: str` - Username to look up

**Returns**:
- `Optional[User]` - User object if found, None otherwise

**Example**:

```python
user = await repository.get_by_username("john_doe")
if user:
    print(f"User email: {user.email}")
```

##### update

```python
async def update(user_id: UUID, update_data: UpdateUserRequest) -> Optional[User]
```

Update an existing user. Performs partial update - only non-None fields are updated.

**Parameters**:
- `user_id: UUID` - ID of user to update
- `update_data: UpdateUserRequest` - Fields to update

**Returns**:
- `Optional[User]` - Updated User object, or None if user not found

**Raises**:
- `ResourceNotFoundError` - If user does not exist
- `ConflictError` - If email/username conflicts with another user

**Example**:

```python
updated_user = await repository.update(
    user_id,
    UpdateUserRequest(email="newemail@example.com", is_active=False)
)
```

##### delete

```python
async def delete(user_id: UUID) -> bool
```

Delete a user by ID.

**Parameters**:
- `user_id: UUID` - ID of user to delete

**Returns**:
- `bool` - True if user was deleted, False if user not found

**Example**:

```python
deleted = await repository.delete(user_id)
if deleted:
    print("User deleted successfully")
else:
    print("User not found")
```

##### list_users

```python
async def list_users(limit: int = 100, offset: int = 0) -> List[User]
```

List users with pagination.

**Parameters**:
- `limit: int` - Maximum number of users to return (default: 100)
- `offset: int` - Number of users to skip (default: 0)

**Returns**:
- `List[User]` - List of User objects

**Example**:

```python
# Get first page (users 0-99)
users = await repository.list_users(limit=100, offset=0)

# Get second page (users 100-199)
users = await repository.list_users(limit=100, offset=100)
```

##### set_password

```python
async def set_password(user_id: UUID, password_request: SetPasswordRequest) -> Optional[User]
```

Set or update a user's password hash.

**Parameters**:
- `user_id: UUID` - ID of user
- `password_request: SetPasswordRequest` - Password hash data

**Returns**:
- `Optional[User]` - Updated User object with new password_hash and password_changed_at

**Example**:

```python
updated_user = await repository.set_password(
    user_id,
    SetPasswordRequest(password_hash="$2b$12$...")
)
```

##### authenticate_user

```python
async def authenticate_user(username_or_email: str, password_hash: str) -> Optional[User]
```

Authenticate a user by username/email and password hash.

**Parameters**:
- `username_or_email: str` - Username or email address
- `password_hash: str` - Password hash to verify

**Returns**:
- `Optional[User]` - User if authentication succeeds, None otherwise

**Example**:

```python
# Service layer would hash the provided password first
user = await repository.authenticate_user(
    "john@example.com",
    hashed_password
)
if user:
    print("Authentication successful")
```

##### get_user_password_hash

```python
async def get_user_password_hash(user_id: UUID) -> Optional[str]
```

Get a user's password hash for verification purposes.

**Parameters**:
- `user_id: UUID` - User ID

**Returns**:
- `Optional[str]` - Password hash string, or None if user not found

**Example**:

```python
password_hash = await repository.get_user_password_hash(user_id)
if password_hash:
    # Verify against provided password
    is_valid = verify_password(provided_password, password_hash)
```

##### search_users

```python
async def search_users(query: str, limit: int = 10) -> List[User]
```

Search users by email or username.

**Parameters**:
- `query: str` - Search query string
- `limit: int` - Maximum results to return (default: 10)

**Returns**:
- `List[User]` - Users matching the search query

**Example**:

```python
# Search for users with "john" in email or username
results = await repository.search_users("john", limit=20)
for user in results:
    print(f"{user.username} - {user.email}")
```

### RolePermissionManager

**Description**: Abstract interface for role and permission management at the global level. Used for authorization and RBAC.

**Type**: Service

**Location**: `portico.ports.user.RolePermissionManager`

**Abstract Methods**:

##### define_role

```python
def define_role(role_name: str, permissions: Set[str]) -> None
```

Define a role with its associated permissions.

**Parameters**:
- `role_name: str` - Name of the role (e.g., "admin", "moderator")
- `permissions: Set[str]` - Set of permission strings

**Example**:

```python
manager.define_role("editor", {"create_post", "edit_post", "delete_own_post"})
manager.define_role("admin", {"create_post", "edit_post", "delete_any_post", "manage_users"})
```

##### get_role_permissions

```python
def get_role_permissions(role_name: str) -> Set[str]
```

Get all permissions for a specific role.

**Parameters**:
- `role_name: str` - Name of the role

**Returns**:
- `Set[str]` - Set of permission strings

**Example**:

```python
permissions = manager.get_role_permissions("editor")
print(permissions)  # {"create_post", "edit_post", "delete_own_post"}
```

##### user_has_permission

```python
def user_has_permission(user: User, permission: str) -> bool
```

Check if a user has a specific permission based on their global role.

**Parameters**:
- `user: User` - User to check
- `permission: str` - Permission string to check

**Returns**:
- `bool` - True if user has the permission

**Example**:

```python
if manager.user_has_permission(user, "delete_post"):
    await delete_post(post_id)
else:
    raise PermissionDeniedError()
```

##### user_has_role

```python
def user_has_role(user: User, role: str) -> bool
```

Check if a user has a specific role.

**Parameters**:
- `user: User` - User to check
- `role: str` - Role name to check

**Returns**:
- `bool` - True if user has the role

**Example**:

```python
if manager.user_has_role(user, "admin"):
    # Show admin panel
    pass
```

##### get_all_roles

```python
def get_all_roles() -> Dict[str, Set[str]]
```

Get all defined roles and their permissions.

**Returns**:
- `Dict[str, Set[str]]` - Dictionary mapping role names to permission sets

**Example**:

```python
all_roles = manager.get_all_roles()
for role_name, permissions in all_roles.items():
    print(f"{role_name}: {', '.join(permissions)}")
```

## Usage Patterns

### Creating Users

```python
from portico import compose
from portico.ports.user import CreateUserRequest

# Initialize application
app = compose.webapp(
    database_url="sqlite+aiosqlite:///app.db",
    kits=[compose.user(password_min_length=8)]
)
await app.initialize()

# Create a user
user = await app.kits["user"].service.create_user(
    CreateUserRequest(
        email="john@example.com",
        username="john_doe",
        password="securePassword123"
    )
)
print(f"Created user: {user.id}")
```

### User Lookups

```python
# Get by ID
user = await app.kits["user"].service.get_user(user_id)

# Get by email
user = await app.kits["user"].service.get_user_by_email("john@example.com")

# Get by username
user = await app.kits["user"].service.get_user_by_username("john_doe")

# List with pagination
users = await app.kits["user"].service.list_users(limit=50, offset=0)
```

### Updating Users

```python
from portico.ports.user import UpdateUserRequest

# Update email and role
updated = await app.kits["user"].service.update_user(
    user.id,
    UpdateUserRequest(
        email="newemail@example.com",
        global_role="moderator"
    )
)

# Deactivate user
await app.kits["user"].service.update_user(
    user.id,
    UpdateUserRequest(is_active=False)
)
```

### Password Management

```python
# Change password
await app.kits["user"].service.update_user(
    user.id,
    UpdateUserRequest(password="newSecurePassword456")
)

# Verify password
try:
    is_valid = await app.kits["user"].service.verify_password(
        user.id,
        "securePassword123"
    )
    if is_valid:
        print("Password correct!")
except AuthenticationError:
    print("User not found")
```

### Working with Events

```python
from portico.kits.user.events import UserCreatedEvent, UserUpdatedEvent

async def on_user_created(event: UserCreatedEvent):
    print(f"New user: {event.email}")
    # Send welcome email, setup defaults, etc.

async def on_user_updated(event: UserUpdatedEvent):
    print(f"User {event.user_id} updated")
    print(f"Changed fields: {event.fields_changed}")

# Subscribe to events
app.events.subscribe(UserCreatedEvent, on_user_created)
app.events.subscribe(UserUpdatedEvent, on_user_updated)
```

## Integration with Kits

### User Kit

The User Kit provides the service layer implementation that uses the User port:

```python
from portico.kits.user import UserKit, UserService

# UserService depends on UserRepository port
class UserService:
    def __init__(
        self,
        user_repository: UserRepository,  # Port interface
        events: EventBus,
        config: UserKitConfig
    ):
        self.repository = user_repository
        self.events = events
        self.config = config
```

**Key Features**:
- Password hashing with bcrypt
- Password policy enforcement (configurable minimum length)
- Event publishing (UserCreatedEvent, UserUpdatedEvent, UserDeletedEvent)
- Validation before repository calls
- Automatic timestamp updates

### Auth Kit

The Auth Kit builds on the User port for authentication:

```python
# Auth Kit depends on User Kit
class AuthKit:
    def __init__(self, user_kit: UserKit, ...):
        self.user_kit = user_kit

    async def login(self, username_or_email: str, password: str):
        # Uses UserRepository via UserKit
        user = await self.user_kit.service.get_user_by_email(username_or_email)
        if user and await self.user_kit.service.verify_password(user.id, password):
            # Create session...
            pass
```

### RBAC Kit

The RBAC Kit extends user authorization with group-based roles:

```python
# RBAC Kit uses User port models
from portico.ports.user import User

class AuthorizationService:
    def check_permission(self, user: User, permission: str, group_id: Optional[UUID] = None):
        # Check global role permissions
        if self.role_manager.user_has_permission(user, permission):
            return True

        # Check group-specific permissions...
        pass
```

## Best Practices

### Immutability

The User model is immutable - treat it as a snapshot:

```python
# ✅ CORRECT - Create new instance
updated_user = await repository.update(user.id, UpdateUserRequest(email="new@example.com"))

# ❌ WRONG - Trying to modify
user.email = "new@example.com"  # Raises FrozenInstanceError!
```

### Password Handling

Never store plain text passwords:

```python
# ✅ CORRECT - Use CreateUserRequest with raw password
request = CreateUserRequest(
    email="user@example.com",
    username="username",
    password="plaintext"  # Will be hashed by service
)

# ✅ ALSO CORRECT - Use pre-hashed password
request = CreateUserRequest(
    email="user@example.com",
    username="username",
    password_hash="$2b$12$..."  # Already hashed
)

# ❌ WRONG - Storing plain password in User model
user = User(
    ...,
    password_hash="plaintext"  # Never do this!
)
```

### Unique Constraints

Handle conflicts gracefully:

```python
from portico.exceptions import ConflictError

try:
    user = await repository.create(CreateUserRequest(...))
except ConflictError as e:
    print(f"User already exists: {e}")
    # Show error to user or handle differently
```

### Pagination

Always paginate when listing users:

```python
# ✅ CORRECT - Use pagination
page_size = 50
page = 0
users = await repository.list_users(limit=page_size, offset=page * page_size)

# ❌ WRONG - Loading all users
users = await repository.list_users(limit=999999)  # Slow and memory-intensive!
```

### Search Performance

Use indexed lookups when possible:

```python
# ✅ FAST - Direct lookup (indexed)
user = await repository.get_by_email("john@example.com")

# ❌ SLOW - Search for exact match
users = await repository.search_users("john@example.com")
user = users[0] if users else None
```

## FAQs

### Q: Why is the User model immutable (frozen)?

A: Immutability ensures data consistency in async operations. Since multiple coroutines might reference the same User object, immutability prevents race conditions where one coroutine modifies the object while another is reading it. When you need to update a user, you get a new User instance from the repository.

### Q: Should I hash passwords in my application code?

A: No, the UserService in the User Kit handles password hashing automatically. When you provide a `password` field in CreateUserRequest, the service hashes it before calling the repository. Only provide `password_hash` if you're doing admin operations or migrations with pre-hashed passwords.

### Q: Can I extend the User model with custom fields?

A: The User port model is intentionally minimal. For custom fields, you have two options:

1. **Store in metadata**: Add a JSONB column in your database model that maps to app-specific data
2. **Create extension models**: Define your own domain model that includes User + custom fields

Example extension pattern:

```python
from portico.ports.user import User

class ExtendedUser(BaseModel):
    user: User  # Core user from port
    profile_picture_url: Optional[str] = None
    bio: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
```

### Q: What's the difference between UserRepository and RolePermissionManager?

A: `UserRepository` handles persistence (CRUD operations, authentication). `RolePermissionManager` handles authorization logic (checking roles and permissions). They're separated because permissions can be defined independently of the database.

### Q: How do I implement a custom adapter for UserRepository?

A: Implement all abstract methods defined in the `UserRepository` interface:

```python
from portico.ports.user import UserRepository, User, CreateUserRequest
from typing import Optional, List
from uuid import UUID

class CustomUserRepository(UserRepository):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize your storage backend

    async def create(self, user_data: CreateUserRequest) -> Optional[User]:
        # Your implementation - store user in your backend
        # Return User domain model
        pass

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        # Your implementation - fetch from your backend
        # Convert to User domain model
        pass

    # ... implement all other abstract methods
```

Then use it in composition:

```python
def user(**config):
    from your_module import CustomUserRepository
    from portico.kits.user import UserKit

    def factory(database, events):
        repository = CustomUserRepository(config["connection_string"])
        return UserKit(database, events, config, user_repository=repository)

    return factory
```

### Q: How do password hashes work in the User model?

A: The `password_hash` field stores hashes in a structured format:

```
algorithm$hash$salt$params
```

Example (bcrypt):
```
$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUe
```

The UserService uses bcrypt by default with these parameters:
- Algorithm: bcrypt (`$2b$`)
- Work factor: 12 rounds
- Auto-generated salt (included in hash)

### Q: Can users exist without passwords?

A: Yes! Set `password_hash` to `None` for SSO users or users who authenticate through external providers:

```python
user = CreateUserRequest(
    email="sso@example.com",
    username="sso_user"
    # No password or password_hash
)

# Check if password auth is available
if not user.has_password():
    print("User must authenticate via SSO")
```

### Q: How do I handle email verification?

A: The base User port doesn't include verification status. You can:

1. Use the `is_active` field (set to False until verified)
2. Extend the User model with a verification status field
3. Use the Audit port to track verification events

Example with `is_active`:

```python
# Create unverified user
user = await service.create_user(
    CreateUserRequest(..., is_active=False)
)

# After email verification
await service.update_user(
    user.id,
    UpdateUserRequest(is_active=True)
)
```
