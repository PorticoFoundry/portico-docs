# Auth Kit

## Overview

**Purpose**: Provide secure user authentication, password hashing with bcrypt, and session management for web applications.

**Key Features**:

- User authentication with username/email and password
- Secure password hashing using bcrypt
- Session creation and validation with configurable expiration
- Token-based session management
- Automatic session cleanup for expired sessions
- Event publishing for login/logout tracking
- Stateless or stateful session modes

**Dependencies**:

- **Injected services**: UserKit (for user data access)
- **Port dependencies**: None (uses UserRepository from UserKit)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose

# Basic configuration
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(password_min_length=8),
        compose.auth(
            session_secret="your-secret-key-minimum-32-characters-long",
            session_timeout_minutes=60,
        ),
    ]
)

# Access the auth service
auth_service = app.kits["auth"].service

# Authenticate a user
from portico.exceptions import AuthenticationError

try:
    result = await auth_service.authenticate(
        username="user@example.com",
        password="user_password"
    )
    # result.user, result.token, result.expires_at
except AuthenticationError:
    # Invalid credentials
    pass
```

## Core Concepts

### Authentication Flow

The Auth Kit provides a complete authentication workflow from credentials to session:

```python
from portico.exceptions import AuthenticationError

# Step 1: Authenticate with credentials
try:
    result = await auth_service.authenticate(
        username="user@example.com",  # Can be email or username
        password="secure_password"
    )

    # result contains:
    # - result.user: User domain model
    # - result.token: Session token (opaque string)
    # - result.expires_at: Expiration datetime

    # Store token in cookie or response header
    response.set_cookie("session_token", result.token, httponly=True)

except AuthenticationError as e:
    # Credentials invalid - same error for user not found or wrong password
    return {"error": str(e)}
```

The authentication process:

1. **Lookup**: Finds user by email or username
2. **Verify**: Checks password against bcrypt hash
3. **Create Session**: Generates secure token and stores session
4. **Publish Event**: Emits `UserLoggedInEvent` for audit/tracking
5. **Return Result**: Provides user, token, and expiration

### Password Hashing

Passwords are hashed using bcrypt with automatic salt generation:

```python
# Hash a password (typically done by UserKit during registration)
hashed = AuthenticationService.hash_password("user_password")
# Returns: "$2b$12$..." (bcrypt hash with embedded salt)

# Verify a password (done automatically during authentication)
is_valid = await auth_service.verify_password(user_id, "user_password")
# Returns: True if password matches hash
```

Bcrypt properties:

- **Adaptive**: Configurable work factor (rounds) for future-proofing
- **Salted**: Unique salt per password prevents rainbow table attacks
- **Slow**: Intentionally slow to resist brute-force attacks

### Session Management

Sessions store authenticated user state with expiration:

```python
# Create a session (done automatically during authentication)
session = await auth_service.create_session(user_id)
# session.token, session.expires_at

# Validate a session (e.g., on each authenticated request)
user = await auth_service.validate_session(token)
if user:
    # Session valid, user authenticated
    pass
else:
    # Session invalid, expired, or not found
    pass

# Logout (invalidate session)
success = await auth_service.logout(token)
```

Sessions are:

- **Time-limited**: Expire after `session_timeout_minutes`
- **Secure tokens**: Generated with `secrets.token_urlsafe(32)` (256 bits)
- **Database-backed**: Stored in `auth_sessions` table (unless stateless mode)
- **User-scoped**: Each session belongs to one user

### Stateless vs Stateful Mode

The Auth Kit supports two session modes:

```python
# Stateful (default) - sessions stored in database
compose.auth(
    session_secret="...",
    enable_session_storage=True  # Default
)

# Stateless - no database storage (requires JWT or similar)
compose.auth(
    session_secret="...",
    enable_session_storage=False
)
```

**Stateful mode** (recommended):

- Sessions stored in `auth_sessions` table
- Can validate and invalidate sessions
- Supports logout functionality
- Requires database queries for validation

**Stateless mode**:

- No database storage
- Cannot validate sessions (returns None)
- No logout support
- Requires external token system (JWT)

## Configuration

### Required Settings

| Setting | Type | Description | Example |
|---------|------|-------------|---------|
| `session_secret` | `str` | Secret key for signing tokens (min 32 chars) | `"your-secret-key-minimum-32-characters-long"` |

Generate a secure secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `session_timeout_minutes` | `int` | `30` | Session expiration time in minutes |
| `enable_session_storage` | `bool` | `True` | Whether to store sessions in database |

**Example:**

```python
from portico import compose

app = compose.webapp(
    kits=[
        compose.user(),
        compose.auth(
            session_secret="your-secret-key-minimum-32-characters-long",
            session_timeout_minutes=1440,  # 24 hours
            enable_session_storage=True,
        ),
    ]
)
```

## Usage Examples

### Example 1: Login Endpoint

```python
from fastapi import Form, Response
from portico.exceptions import AuthenticationError

@app.post("/login")
async def login(
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
):
    auth_service = app.kits["auth"].service

    try:
        result = await auth_service.authenticate(
            username=email,  # Can be email or username
            password=password
        )

        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=result.token,
            httponly=True,  # Prevent JavaScript access
            secure=True,     # HTTPS only
            samesite="lax",  # CSRF protection
            max_age=60 * 60, # 1 hour
        )

        return {
            "success": True,
            "user_id": str(result.user.id),
            "email": result.user.email,
            "expires_at": result.expires_at.isoformat(),
        }

    except AuthenticationError as e:
        return {"success": False, "error": str(e)}, 401
```

### Example 2: Protected Endpoint with Session Validation

```python
from fastapi import Cookie, HTTPException

@app.get("/profile")
async def get_profile(session_token: str = Cookie(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    auth_service = app.kits["auth"].service
    user = await auth_service.validate_session(session_token)

    if not user:
        raise HTTPException(status_code=401, detail="Session invalid or expired")

    return {
        "user_id": str(user.id),
        "email": user.email,
        "username": user.username,
    }
```

### Example 3: Logout Endpoint

```python
@app.post("/logout")
async def logout(
    response: Response,
    session_token: str = Cookie(None)
):
    if session_token:
        auth_service = app.kits["auth"].service
        await auth_service.logout(session_token)

    # Clear cookie
    response.delete_cookie("session_token")

    return {"success": True}
```

### Example 4: Using FastAPI Dependency Injection

```python
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@app.get("/dashboard")
async def dashboard(current_user = deps.current_user):
    # current_user is automatically validated from session
    # Raises 401 if session invalid
    return {
        "welcome": f"Hello, {current_user.email}!",
        "user_id": str(current_user.id),
    }

@app.get("/public")
async def public_page(user = deps.optional_user):
    # user is None if not authenticated, otherwise validated User
    if user:
        return {"message": f"Welcome back, {user.email}"}
    else:
        return {"message": "Welcome, guest"}
```

## Domain Models

### Session

Represents an active user session.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | Auto | Unique session identifier |
| `user_id` | `UUID` | Yes | - | User who owns this session |
| `token` | `str` | Yes | - | Session token (opaque string for client) |
| `expires_at` | `datetime` | Yes | - | When the session expires (UTC) |
| `created_at` | `datetime` | Yes | Auto | When the session was created (UTC) |

### AuthResult

Result of successful authentication.

| Field | Type | Description |
|-------|------|-------------|
| `user` | `User` | Authenticated user (from UserKit) |
| `token` | `str` | Session token to return to client |
| `expires_at` | `datetime` | When the session expires (UTC) |

## Database Models

### SessionModel

**Table**: `auth_sessions`

**Columns**:

- `id`: UUID, primary key
- `user_id`: UUID, foreign key to `users.id` (cascade delete)
- `token`: String(255), unique, indexed
- `expires_at`: DateTime with timezone
- `created_at`: DateTime with timezone

**Indexes**:

- `idx_sessions_token`: On `token` column for fast lookups

**Relationships**:

- Belongs to: User (via `user_id`)

## Events

This kit publishes the following events:

### `UserLoggedInEvent`

**Triggered when**: A user successfully authenticates and creates a session.

**Payload**:

```python
{
    "user_id": UUID,
    "session_id": UUID,
    "timestamp": datetime
}
```

**Listeners**: Audit logging, security monitoring, usage analytics.

**Example listener:**

```python
from portico.kits.auth.events import UserLoggedInEvent

async def log_login(event: UserLoggedInEvent):
    print(f"User {event.user_id} logged in at {event.timestamp}")

events = app.events
await events.subscribe(UserLoggedInEvent, log_login)
```

### `UserLoggedOutEvent`

**Triggered when**: A user explicitly logs out (session invalidated).

**Payload**:

```python
{
    "user_id": UUID,
    "session_id": UUID,
    "timestamp": datetime
}
```

**Listeners**: Audit logging, session cleanup handlers.

### `SessionExpiredEvent`

**Triggered when**: A session expires (checked during validation).

**Payload**:

```python
{
    "user_id": UUID,
    "session_id": UUID,
    "expired_at": datetime
}
```

**Listeners**: Session cleanup, security monitoring.

## Best Practices

### 1. Use Secure Session Cookies

Always set security flags on session cookies:

```python
# ✅ GOOD - Secure cookie configuration
response.set_cookie(
    key="session_token",
    value=result.token,
    httponly=True,   # Prevent XSS attacks
    secure=True,      # HTTPS only
    samesite="lax",   # CSRF protection
    max_age=3600,     # 1 hour
)

# ❌ BAD - Insecure cookie
response.set_cookie("session_token", result.token)
# Vulnerable to XSS, works over HTTP, no CSRF protection
```

### 2. Store Session Secret in Environment Variables

Never hardcode session secrets in source code:

```python
import os

# ✅ GOOD - Load from environment
app = compose.webapp(
    kits=[
        compose.auth(
            session_secret=os.environ["SESSION_SECRET"],
            session_timeout_minutes=60,
        ),
    ]
)

# ❌ BAD - Hardcoded secret
app = compose.webapp(
    kits=[
        compose.auth(
            session_secret="my-secret-123",  # In version control!
        ),
    ]
)
```

Generate and store securely:

```bash
# Generate secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env file (never commit this file)
echo "SESSION_SECRET=<generated-secret>" >> .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Implement Session Cleanup

Schedule periodic cleanup of expired sessions:

```python
# ✅ GOOD - Scheduled cleanup
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

async def cleanup_expired_sessions():
    session_repo = app.kits["auth"].repository
    count = await session_repo.delete_expired_sessions()
    print(f"Cleaned up {count} expired sessions")

# Run daily at 3 AM
scheduler.add_job(cleanup_expired_sessions, "cron", hour=3)
scheduler.start()

# ❌ BAD - No cleanup
# Expired sessions accumulate in database indefinitely
```

### 4. Use Consistent Error Messages

Avoid revealing whether username exists:

```python
# ✅ GOOD - Generic error message
try:
    result = await auth_service.authenticate(username, password)
except AuthenticationError:
    return {"error": "Invalid username or password"}, 401
    # Doesn't reveal if user exists

# ❌ BAD - Reveals user existence
user = await user_service.get_by_email(email)
if not user:
    return {"error": "User not found"}, 404  # Information leak
if not verify_password(password):
    return {"error": "Wrong password"}, 401
```

### 5. Set Appropriate Session Timeouts

Balance security and user experience:

```python
# ✅ GOOD - Context-appropriate timeouts
# Banking app - short timeout
compose.auth(session_timeout_minutes=15)

# Internal tool - medium timeout
compose.auth(session_timeout_minutes=480)  # 8 hours

# Public forum - longer timeout
compose.auth(session_timeout_minutes=10080)  # 7 days

# ❌ BAD - One size fits all
compose.auth(session_timeout_minutes=30)  # Same for all apps
```

### 6. Handle Session Validation Errors Gracefully

Provide clear feedback for expired sessions:

```python
# ✅ GOOD - Clear error handling
@app.get("/api/data")
async def get_data(session_token: str = Cookie(None)):
    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Cookie"}
        )

    user = await auth_service.validate_session(session_token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid"
        )

    return {"data": "..."}

# ❌ BAD - Unclear errors
@app.get("/api/data")
async def get_data(session_token: str = Cookie(None)):
    user = await auth_service.validate_session(session_token)
    if not user:
        return {"error": "error"}  # What error? Why?
```

### 7. Use Dependency Injection for Protected Routes

Leverage FastAPI dependencies for cleaner code:

```python
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

# ✅ GOOD - Dependency injection
@app.get("/protected")
async def protected_route(current_user = deps.current_user):
    # current_user automatically validated
    return {"user": current_user.email}

# ❌ BAD - Manual validation in every route
@app.get("/protected")
async def protected_route(session_token: str = Cookie(None)):
    if not session_token:
        raise HTTPException(status_code=401)
    user = await auth_service.validate_session(session_token)
    if not user:
        raise HTTPException(status_code=401)
    # Repeated boilerplate
    return {"user": user.email}
```

## Security Considerations

### Password Security

The Auth Kit uses bcrypt for password hashing:

- **Never store plaintext passwords**
- **Never log passwords** (even in debug mode)
- **Use bcrypt's adaptive cost** (automatically future-proof)
- **Password hashing is CPU-intensive** (intentional anti-brute-force)

### Session Token Security

Session tokens are generated with cryptographically secure random bytes:

- **256-bit tokens** via `secrets.token_urlsafe(32)`
- **Opaque tokens** (not JWTs - no embedded data)
- **Single-use** (invalidated on logout)
- **Time-limited** (automatic expiration)

### Common Attack Mitigations

**Brute Force**:

- Implement rate limiting on login endpoints
- Use slow password hashing (bcrypt)
- Consider account lockout after N failures

**Session Fixation**:

- Generate new token on each login
- Don't accept tokens from query parameters

**Session Hijacking**:

- Use `httponly` cookies (prevent XSS)
- Use `secure` cookies (HTTPS only)
- Consider binding sessions to IP/User-Agent

**CSRF**:

- Use `samesite` cookie attribute
- Implement CSRF tokens for state-changing operations

### HTTPS Requirement

**Always use HTTPS in production**:

```python
# Production configuration
response.set_cookie(
    "session_token",
    token,
    secure=True,  # Requires HTTPS
    httponly=True,
    samesite="lax"
)
```

## FAQs

### Q: How do I implement "Remember Me" functionality?

A: Create two session types with different timeouts:

```python
# Standard session (30 minutes)
result = await auth_service.authenticate(username, password)

# If "remember me" checked, create longer session
if remember_me:
    # Override config temporarily or create custom session
    long_session = await auth_service.create_session(user.id)
    # Manually set longer expiration
```

Alternatively, use a separate "remember me" token table with longer expiration.

### Q: Can I use JWT tokens instead of database sessions?

A: Yes, use stateless mode and implement JWT separately:

```python
compose.auth(
    session_secret="...",
    enable_session_storage=False  # Stateless mode
)

# Implement JWT in your application layer
import jwt

def create_jwt(user_id: UUID) -> str:
    payload = {"user_id": str(user_id), "exp": datetime.now(UTC) + timedelta(hours=1)}
    return jwt.encode(payload, secret_key, algorithm="HS256")
```

### Q: How do I implement "logout from all devices"?

A: Query all sessions for a user and delete them:

```python
async def logout_all_devices(user_id: UUID):
    # Custom query in SessionRepository
    async with database.transaction() as session:
        await session.execute(
            delete(SessionModel).where(SessionModel.user_id == user_id)
        )
```

### Q: Should I validate sessions on every request?

A: Yes, for protected endpoints. Use dependency injection to minimize boilerplate:

```python
deps = Dependencies(app)

@app.get("/protected")
async def protected(user = deps.current_user):
    # Session validated automatically on each request
    pass
```

For high-traffic APIs, consider caching session validation with Redis.

### Q: How do I handle concurrent login sessions?

A: By default, multiple sessions are allowed (user can login from multiple devices). To enforce single session:

```python
async def authenticate_single_session(username: str, password: str):
    result = await auth_service.authenticate(username, password)

    # Delete other sessions for this user
    async with database.transaction() as session:
        await session.execute(
            delete(SessionModel).where(
                SessionModel.user_id == result.user.id,
                SessionModel.token != result.token
            )
        )

    return result
```

### Q: What's the performance impact of bcrypt?

A: Bcrypt is intentionally slow (anti-brute-force). Expect ~100-300ms per hash/verify on modern hardware. This is acceptable for login (infrequent) but too slow for per-request operations. Never hash passwords on every request - use session tokens instead.

### Q: How do I test authentication in unit tests?

A: Create test sessions directly or mock the auth service:

```python
# Create test user and session
user = await user_kit.service.create_user(CreateUserRequest(...))
session = await auth_kit.service.create_session(user.id)

# Use session token in test requests
response = client.get("/protected", cookies={"session_token": session.token})
```

### Q: Can I customize the session token format?

A: Yes, modify `create_session` in your application:

```python
# Custom token format (e.g., JWT)
import jwt

async def create_jwt_session(user_id: UUID):
    payload = {"user_id": str(user_id), "exp": ...}
    token = jwt.encode(payload, secret_key, algorithm="HS256")

    # Still store in database for validation/logout
    return await session_repo.create_session(user_id, token, expires_at)
```
