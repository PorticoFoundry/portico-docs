# User Port

## Overview

The User Port defines the contract for user management, authentication, and authorization in Portico applications.

**Purpose**: Provides interfaces and domain models for user identity, password-based authentication, and global role-based access control.

**Domain**: User management, authentication, authorization

**Key Capabilities**:

- User CRUD operations (create, read, update, delete)
- Password-based authentication with secure hashing
- Global role-based authorization
- User search and listing with pagination
- Email and username uniqueness enforcement

**Port Type**: Repository

**When to Use**:

- Implementing user registration and account management
- Authenticating users with username/email and password
- Managing user profiles and lifecycle
- Implementing role-based access control at the global level

## Domain Models

### User

Core domain model representing an authenticated user. Immutable snapshot of user state.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | `uuid4()` | Unique user identifier |
| `email` | `str` | Yes | - | User's email address (unique) |
| `username` | `str` | Yes | - | Username for login (unique) |
| `is_active` | `bool` | Yes | `True` | Whether the account is active |
| `global_role` | `str` | Yes | `"user"` | Global system role (e.g., "user", "admin") |
| `password_hash` | `Optional[str]` | No | `None` | Serialized password hash |
| `password_changed_at` | `Optional[datetime]` | No | `None` | Last password change timestamp (UTC) |
| `created_at` | `datetime` | Yes | `now(UTC)` | Account creation timestamp (UTC) |
| `updated_at` | `datetime` | Yes | `now(UTC)` | Last update timestamp (UTC) |

**Methods**:

- `has_password() -> bool` - Returns True if user has a password set

**Example**:

```python
from portico.ports.user import User
from datetime import datetime, UTC
from uuid import uuid4

user = User(
    id=uuid4(),
    email="john@example.com",
    username="john_doe",
    is_active=True,
    global_role="admin",
    password_hash="$2b$12$...",
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC)
)

# User is immutable (frozen)
if user.has_password():
    print("User has password authentication enabled")
```

### CreateUserRequest

Request model for creating a new user. Supports both raw password (to be hashed) and pre-hashed password workflows.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `email` | `str` | Yes | - | User's email address |
| `username` | `str` | Yes | - | User's username |
| `global_role` | `str` | No | `"user"` | Initial global role |
| `password_hash` | `Optional[str]` | No | `None` | Pre-hashed password |
| `password` | `Optional[str]` | No | `None` | Raw password (will be hashed by service) |

**Example**:

```python
from portico.ports.user import CreateUserRequest

# User registration with raw password
request = CreateUserRequest(
    email="john@example.com",
    username="john_doe",
    password="securePassword123"
)

# SSO user without password
sso_request = CreateUserRequest(
    email="sso@example.com",
    username="sso_user"
)
```

### UpdateUserRequest

Request model for updating an existing user. All fields optional for partial updates.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `email` | `Optional[str]` | No | `None` | New email address |
| `username` | `Optional[str]` | No | `None` | New username |
| `is_active` | `Optional[bool]` | No | `None` | New active status |
| `global_role` | `Optional[str]` | No | `None` | New global role |

**Example**:

```python
from portico.ports.user import UpdateUserRequest

# Update email only
request = UpdateUserRequest(email="newemail@example.com")

# Deactivate user
request = UpdateUserRequest(is_active=False)
```

### SetPasswordRequest

Request model for setting or updating a user's password hash.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `password_hash` | `str` | Yes | - | Serialized password hash |

**Example**:

```python
from portico.ports.user import SetPasswordRequest

request = SetPasswordRequest(
    password_hash="$2b$12$..."
)
```

## Port Interfaces

### UserRepository

Abstract interface for user persistence operations.

**Location**: `portico.ports.user.UserRepository`

#### Key Methods

##### create

```python
async def create(user_data: CreateUserRequest) -> Optional[User]
```

Create a new user in the system.

**Parameters**:

- `user_data: CreateUserRequest` - User creation data

**Returns**: Created User object, or None if creation failed

**Raises**:

- `ConflictError` - If email or username already exists
- `ValidationError` - If data validation fails

**Example**:

```python
from portico.ports.user import UserRepository, CreateUserRequest

user = await repository.create(
    CreateUserRequest(
        email="john@example.com",
        username="john_doe",
        password="securePassword123"
    )
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

**Returns**: User if authentication succeeds, None otherwise

**Example**:

```python
# Service layer hashes the provided password first
user = await repository.authenticate_user(
    "john@example.com",
    hashed_password
)
if user:
    print("Authentication successful")
```

#### Other Methods

##### get_by_id

```python
async def get_by_id(user_id: UUID) -> Optional[User]
```

Retrieve a user by their unique ID.

##### get_by_email

```python
async def get_by_email(email: str) -> Optional[User]
```

Retrieve a user by their email address.

##### get_by_username

```python
async def get_by_username(username: str) -> Optional[User]
```

Retrieve a user by their username.

##### update

```python
async def update(user_id: UUID, update_data: UpdateUserRequest) -> Optional[User]
```

Update an existing user. Performs partial update - only non-None fields are updated.

##### delete

```python
async def delete(user_id: UUID) -> bool
```

Delete a user by ID. Returns True if deleted, False if not found.

##### list_users

```python
async def list_users(limit: int = 100, offset: int = 0) -> List[User]
```

List users with pagination.

##### set_password

```python
async def set_password(user_id: UUID, password_request: SetPasswordRequest) -> Optional[User]
```

Set or update a user's password hash.

##### get_user_password_hash

```python
async def get_user_password_hash(user_id: UUID) -> Optional[str]
```

Get a user's password hash for verification purposes.

##### search_users

```python
async def search_users(query: str, limit: int = 10) -> List[User]
```

Search users by email or username.

### RolePermissionManager

Abstract interface for role and permission management at the global level.

**Location**: `portico.ports.user.RolePermissionManager`

##### define_role

```python
def define_role(role_name: str, permissions: Set[str]) -> None
```

Define a role with its associated permissions.

##### get_role_permissions

```python
def get_role_permissions(role_name: str) -> Set[str]
```

Get all permissions for a specific role.

##### user_has_permission

```python
def user_has_permission(user: User, permission: str) -> bool
```

Check if a user has a specific permission based on their global role.

##### user_has_role

```python
def user_has_role(user: User, role: str) -> bool
```

Check if a user has a specific role.

##### get_all_roles

```python
def get_all_roles() -> Dict[str, Set[str]]
```

Get all defined roles and their permissions.

## Common Patterns

### Creating and Authenticating Users

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

# Authenticate user (via Auth Kit)
authenticated = await app.kits["auth"].service.login(
    username_or_email="john@example.com",
    password="securePassword123"
)

if authenticated:
    print(f"Logged in as: {authenticated.email}")
```

### Role-Based Authorization

```python
from portico.ports.user import User

# Define roles
manager = app.kits["user"].role_manager
manager.define_role("editor", {"create_post", "edit_post", "delete_own_post"})
manager.define_role("admin", {"create_post", "edit_post", "delete_any_post", "manage_users"})

# Check permissions
user = await app.kits["user"].service.get_user(user_id)
if manager.user_has_permission(user, "delete_post"):
    await delete_post(post_id)
else:
    raise AuthorizationError("Permission denied")
```

## Integration with Kits

The User Port is used by the **User Kit** to provide user management services.

```python
from portico import compose

# Configure User Kit
app = compose.webapp(
    kits=[
        compose.user(
            password_min_length=8,
            password_max_age_days=90
        )
    ]
)

# Access User Service
user_service = app.kits["user"].service

# Create user
user = await user_service.create_user(
    CreateUserRequest(email="user@example.com", username="user", password="password123")
)

# Update user
updated = await user_service.update_user(
    user.id,
    UpdateUserRequest(global_role="admin")
)
```

The User Kit provides:

- Password hashing with bcrypt
- Password policy enforcement
- Event publishing (UserCreatedEvent, UserUpdatedEvent, UserDeletedEvent)
- Validation before repository calls

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Immutability**: User model is immutable - always get fresh instances from repository

   ```python
   # ✅ GOOD
   updated_user = await repository.update(user.id, UpdateUserRequest(email="new@example.com"))

   # ❌ BAD
   user.email = "new@example.com"  # Raises FrozenInstanceError!
   ```

2. **Password Security**: Never store plain text passwords. Use CreateUserRequest with `password` field for automatic hashing

   ```python
   # ✅ GOOD
   CreateUserRequest(email="user@example.com", username="user", password="plaintext")

   # ❌ BAD
   User(..., password_hash="plaintext")  # Never store plain passwords!
   ```

3. **Unique Constraints**: Handle email/username conflicts gracefully with ConflictError

   ```python
   from portico.exceptions import ConflictError

   try:
       user = await repository.create(CreateUserRequest(...))
   except ConflictError:
       print("User already exists")
   ```

4. **Pagination**: Always paginate when listing users to avoid memory issues

   ```python
   # ✅ GOOD
   users = await repository.list_users(limit=50, offset=0)

   # ❌ BAD
   users = await repository.list_users(limit=999999)
   ```

5. **Indexed Lookups**: Use direct lookups (get_by_email) instead of search for better performance

   ```python
   # ✅ FAST - Direct lookup (indexed)
   user = await repository.get_by_email("john@example.com")

   # ❌ SLOW - Search for exact match
   users = await repository.search_users("john@example.com")
   ```

## FAQs

### Why is the User model immutable (frozen)?

Immutability ensures data consistency in async operations. Multiple coroutines might reference the same User object - immutability prevents race conditions. When you update a user, you get a new User instance from the repository.

### Should I hash passwords in my application code?

No, the UserService in the User Kit handles password hashing automatically. Provide a `password` field in CreateUserRequest - the service hashes it before calling the repository. Only provide `password_hash` for admin operations with pre-hashed passwords.

### Can I extend the User model with custom fields?

The User port model is intentionally minimal. For custom fields:

1. **Store in metadata**: Add a JSONB column in your database model
2. **Create extension models**: Define your own model that includes User + custom fields

```python
from portico.ports.user import User

class ExtendedUser(BaseModel):
    user: User  # Core user from port
    profile_picture_url: Optional[str] = None
    bio: Optional[str] = None
```

### Can users exist without passwords?

Yes! Set `password_hash` to `None` for SSO users or users who authenticate through external providers:

```python
user = CreateUserRequest(
    email="sso@example.com",
    username="sso_user"
    # No password or password_hash
)

if not user.has_password():
    print("User must authenticate via SSO")
```

### How do I implement a custom adapter for UserRepository?

Implement all abstract methods defined in the `UserRepository` interface:

```python
from portico.ports.user import UserRepository, User, CreateUserRequest

class CustomUserRepository(UserRepository):
    async def create(self, user_data: CreateUserRequest) -> Optional[User]:
        # Your implementation
        pass

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        # Your implementation
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
