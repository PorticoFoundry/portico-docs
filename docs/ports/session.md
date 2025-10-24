# Session Port

## Overview

The Session Port defines the contract for session storage backends in Portico applications.

**Purpose**: Abstract session storage operations to enable secure, scalable user session management with pluggable storage backends.

**Domain**: Authentication, security, user session management

**Key Capabilities**:

- Secure session data storage and retrieval
- Session expiration and lifecycle management
- User-based session management and bulk operations
- Session metadata tracking (IP address, user agent, access times)
- Automatic expired session cleanup
- Multi-session support per user

**Port Type**: Storage

**When to Use**:

- Applications requiring stateful user authentication
- Multi-device session management
- Systems requiring session revocation capabilities
- Applications tracking session metadata for security auditing
- Services requiring granular session lifecycle control

## Domain Models

### SessionData

Represents session data with metadata for authentication and lifecycle management. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `Optional[str]` | No | `None` | Session token/identifier (validated as non-empty) |
| `user_id` | `UUID` | Yes | - | User who owns this session |
| `created_at` | `datetime` | Yes | - | When the session was created (UTC) |
| `expires_at` | `datetime` | Yes | - | When the session expires (UTC) |
| `last_accessed` | `datetime` | Yes | - | Last access timestamp (UTC) |
| `ip_address` | `Optional[str]` | No | `None` | IP address of session creator |
| `user_agent` | `Optional[str]` | No | `None` | User agent string from session creation |

**Properties**:

- `is_expired -> bool` - Returns True if current time exceeds expires_at
- `has_valid_token -> bool` - Returns True if session has non-empty token

**Methods**:

- `touch(extend_by: Optional[timedelta] = None) -> SessionData` - Returns new SessionData with updated last_accessed time and optionally extended expiration

**Example**:

```python
from datetime import UTC, datetime, timedelta
from uuid import UUID
from portico.ports.session import SessionData

# Create session data
session = SessionData(
    id="abc123token",
    user_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
    created_at=datetime.now(UTC),
    expires_at=datetime.now(UTC) + timedelta(hours=24),
    last_accessed=datetime.now(UTC),
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0..."
)

# Check expiration
if not session.is_expired:
    # Session is still valid
    user_id = session.user_id

# Update last accessed time
session = session.touch()

# Extend session expiration
session = session.touch(extend_by=timedelta(hours=1))
```

## Port Interfaces

### SessionStorage

The `SessionStorage` abstract base class defines the contract for all session storage backends.

**Location**: `portico.ports.session.SessionStorage`

#### Key Methods

##### store_session

```python
async def store_session(token: str, session_data: SessionData) -> None
```

Stores session data with the given token. Primary method for session creation and updates.

**Parameters**:

- `token`: Session token/identifier to use as the storage key
- `session_data`: SessionData object containing session information

**Example**:

```python
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from portico.ports.session import SessionData, SessionStorage

async def create_session(
    user_id: UUID,
    storage: SessionStorage,
    session_timeout: timedelta = timedelta(hours=24)
) -> str:
    """Create a new user session."""

    # Generate secure token
    token = secrets.token_urlsafe(32)

    # Create session data
    session_data = SessionData(
        id=token,
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + session_timeout,
        last_accessed=datetime.now(UTC),
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )

    # Store session
    await storage.store_session(token, session_data)

    return token
```

##### get_session

```python
async def get_session(token: str) -> Optional[SessionData]
```

Retrieves session data by token. Primary method for session validation and retrieval.

**Parameters**:

- `token`: Session token to retrieve

**Returns**: `SessionData` if found, `None` otherwise.

**Example**:

```python
async def validate_session(
    token: str,
    storage: SessionStorage
) -> Optional[UUID]:
    """Validate session and return user ID."""

    # Retrieve session
    session = await storage.get_session(token)

    if session is None:
        # Session not found
        return None

    if session.is_expired:
        # Session expired - delete it
        await storage.delete_session(token)
        return None

    # Update last accessed time
    session = session.touch()
    await storage.store_session(token, session)

    return session.user_id
```

#### Other Methods

##### delete_session

```python
async def delete_session(token: str) -> bool
```

Deletes session by token. Returns True if session existed and was deleted.

**Example**:

```python
# User logout
async def logout(token: str, storage: SessionStorage):
    deleted = await storage.delete_session(token)
    if deleted:
        logger.info("session_deleted", token=token)
```

##### delete_user_sessions

```python
async def delete_user_sessions(user_id: UUID) -> int
```

Deletes all sessions for a user. Returns number of sessions deleted.

**Example**:

```python
# User password changed - revoke all sessions
async def revoke_all_sessions(user_id: UUID, storage: SessionStorage):
    count = await storage.delete_user_sessions(user_id)
    logger.info("sessions_revoked", user_id=user_id, count=count)
```

##### cleanup_expired_sessions

```python
async def cleanup_expired_sessions() -> int
```

Removes expired sessions. Returns number of sessions cleaned up.

**Example**:

```python
# Periodic cleanup task
async def cleanup_task(storage: SessionStorage):
    while True:
        await asyncio.sleep(3600)  # Every hour
        count = await storage.cleanup_expired_sessions()
        logger.info("expired_sessions_cleaned", count=count)
```

##### get_user_sessions

```python
async def get_user_sessions(user_id: UUID) -> list[SessionData]
```

Gets all active sessions for a user.

**Example**:

```python
# List user's active sessions
async def list_sessions(user_id: UUID, storage: SessionStorage):
    sessions = await storage.get_user_sessions(user_id)

    return [
        {
            "token": s.id,
            "created_at": s.created_at,
            "last_accessed": s.last_accessed,
            "ip_address": s.ip_address,
            "user_agent": s.user_agent
        }
        for s in sessions
        if not s.is_expired
    ]
```

## Exceptions

The Session Port defines several specialized exceptions for session-related errors:

### SessionError

Base exception for all session-related errors.

**Error Code**: `SESSION_ERROR`

### SessionNotFoundError

Raised when attempting to access a session that doesn't exist.

**Error Code**: `SESSION_NOT_FOUND`
**HTTP Status**: 404

```python
from portico.exceptions import SessionNotFoundError

session = await storage.get_session(token)
if session is None:
    raise SessionNotFoundError(token)
```

### SessionExpiredError

Raised when attempting to use an expired session.

**Error Code**: `SESSION_EXPIRED`
**HTTP Status**: 401

```python
from portico.exceptions import SessionExpiredError

if session.is_expired:
    raise SessionExpiredError(token)
```

### InvalidSessionError

Raised when session token is malformed or corrupted.

**Error Code**: `INVALID_SESSION`
**HTTP Status**: 401

```python
from portico.exceptions import InvalidSessionError

if not session.has_valid_token:
    raise InvalidSessionError("Session token is empty or invalid")
```

## Common Patterns

### Session Creation with Cookie Storage

```python
from datetime import UTC, datetime, timedelta
from fastapi import Response
import secrets
from portico.ports.session import SessionData, SessionStorage

async def create_user_session(
    user_id: UUID,
    storage: SessionStorage,
    response: Response,
    session_timeout_minutes: int = 1440  # 24 hours
) -> str:
    """Create session and set cookie."""

    # Generate secure token
    token = secrets.token_urlsafe(32)

    # Create session data
    session_data = SessionData(
        id=token,
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=session_timeout_minutes),
        last_accessed=datetime.now(UTC),
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None
    )

    # Store in backend
    await storage.store_session(token, session_data)

    # Set HTTP-only cookie
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=True,  # HTTPS only in production
        samesite="lax",
        max_age=session_timeout_minutes * 60
    )

    return token
```

### Session Validation Middleware

```python
from fastapi import Request, HTTPException
from portico.ports.session import SessionStorage
from portico.exceptions import SessionExpiredError, SessionNotFoundError

async def validate_session_middleware(
    request: Request,
    storage: SessionStorage
) -> UUID:
    """Validate session from cookie and return user ID."""

    # Extract token from cookie
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="No session token")

    # Retrieve session
    session = await storage.get_session(token)
    if session is None:
        raise SessionNotFoundError(token)

    # Check expiration
    if session.is_expired:
        await storage.delete_session(token)
        raise SessionExpiredError(token)

    # Touch session (update last_accessed)
    session = session.touch()
    await storage.store_session(token, session)

    return session.user_id
```

### Sliding Session Expiration

```python
async def refresh_session(
    token: str,
    storage: SessionStorage,
    extension: timedelta = timedelta(hours=1)
) -> SessionData:
    """Extend session expiration on activity."""

    session = await storage.get_session(token)
    if session is None:
        raise SessionNotFoundError(token)

    if session.is_expired:
        raise SessionExpiredError(token)

    # Touch and extend expiration
    session = session.touch(extend_by=extension)
    await storage.store_session(token, session)

    return session
```

### Session Cleanup Background Task

```python
import asyncio
from portico.ports.session import SessionStorage
from portico.kits.logging import get_logger

logger = get_logger(__name__)

async def periodic_session_cleanup(
    storage: SessionStorage,
    interval_seconds: int = 3600  # Every hour
):
    """Background task to clean up expired sessions."""

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            count = await storage.cleanup_expired_sessions()

            if count > 0:
                logger.info(
                    "expired_sessions_cleaned",
                    count=count,
                    interval_seconds=interval_seconds
                )
        except Exception as e:
            logger.error(
                "session_cleanup_failed",
                error=str(e),
                error_type=type(e).__name__
            )
```

## Integration with Kits

The Session Port is designed to be used by authentication kits and session management services.

**Note**: Currently, the Session Port is not directly integrated with a specific kit. The **AuthKit** has its own session management implementation. Future versions may provide a dedicated SessionKit that uses this port.

**Potential Usage Pattern**:

```python
from portico import compose
from portico.ports.session import SessionStorage

# Future integration pattern (not yet implemented)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.session(
            backend="redis",  # or "database", "memory"
            redis_url="redis://localhost:6379/0",
            session_timeout_minutes=1440,
            cleanup_interval_seconds=3600
        ),
    ],
)

# Access session storage
session_storage: SessionStorage = app.kits["session"].storage
```

## Best Practices

1. **Use Secure Token Generation**: Always use cryptographically secure random tokens

   ```python
   # ✅ GOOD: Cryptographically secure
   import secrets
   token = secrets.token_urlsafe(32)

   # ❌ BAD: Predictable tokens
   import random
   token = str(random.randint(0, 999999))  # NEVER do this!
   ```

2. **Set Appropriate Expiration Times**: Different session types need different timeouts

   ```python
   # ✅ GOOD: Different timeouts for different contexts
   expires_at = datetime.now(UTC) + timedelta(hours=24)    # Regular sessions
   expires_at = datetime.now(UTC) + timedelta(minutes=15)  # Admin sessions
   expires_at = datetime.now(UTC) + timedelta(days=30)     # Remember-me tokens

   # ❌ BAD: No expiration
   expires_at = datetime.max  # Sessions never expire!
   ```

3. **Always Touch Sessions on Access**: Update last_accessed for sliding expiration

   ```python
   # ✅ GOOD: Update access time
   session = await storage.get_session(token)
   if session and not session.is_expired:
       session = session.touch()
       await storage.store_session(token, session)

   # ❌ BAD: Never update access time
   session = await storage.get_session(token)
   # Session will expire even if actively used
   ```

4. **Cleanup Expired Sessions Regularly**: Prevent storage bloat

   ```python
   # ✅ GOOD: Scheduled cleanup
   async def cleanup_task():
       while True:
           await asyncio.sleep(3600)  # Every hour
           await storage.cleanup_expired_sessions()

   asyncio.create_task(cleanup_task())

   # ❌ BAD: No cleanup
   # Expired sessions accumulate forever
   ```

5. **Store Minimal Session Data**: Keep session metadata lightweight

   ```python
   # ✅ GOOD: Store only user_id, fetch user data as needed
   session = SessionData(
       id=token,
       user_id=user_id,
       # ... minimal metadata
   )

   # Later, fetch full user data
   user = await user_repository.get_user(session.user_id)

   # ❌ BAD: Store entire user object in session
   # This duplicates data and can lead to stale information
   ```

6. **Revoke Sessions on Security Events**: Delete sessions when needed

   ```python
   # ✅ GOOD: Revoke on password change
   async def change_password(user_id: UUID, new_password: str):
       await user_service.update_password(user_id, new_password)
       # Revoke all existing sessions
       await storage.delete_user_sessions(user_id)

   # ❌ BAD: Keep old sessions active after password change
   # Allows potentially compromised sessions to remain valid
   ```

## FAQs

### How is SessionData different from the auth kit's Session model?

**SessionData** (from the Session Port) is designed as a generic session storage model with rich metadata (IP address, user agent, access tracking). It's intended for flexible session storage backends.

The **auth kit's Session model** is a simpler database-backed session model specific to the current AuthKit implementation.

Future versions may unify these or provide adapters between them.

### Should I use database or Redis for session storage?

**Database storage** when:
- You have a small user base (< 10,000 active sessions)
- You want session persistence across restarts
- You already have a database and want to minimize dependencies
- You need ACID guarantees for session operations

**Redis storage** when:
- You have a large user base (> 10,000 active sessions)
- Performance is critical (sub-millisecond session lookups)
- You can tolerate session loss on Redis restart (or use Redis persistence)
- You need distributed session sharing across app servers

### How do I implement "remember me" functionality?

Create two session types with different expiration times:

```python
# Regular session: 24 hours
regular_session = SessionData(
    id=session_token,
    user_id=user_id,
    expires_at=datetime.now(UTC) + timedelta(hours=24),
    # ...
)

# Remember-me session: 30 days
remember_token = secrets.token_urlsafe(32)
remember_session = SessionData(
    id=remember_token,
    user_id=user_id,
    expires_at=datetime.now(UTC) + timedelta(days=30),
    # ...
)
```

Store both tokens in separate cookies with appropriate expiration.

### Can one user have multiple active sessions?

Yes! This is common for users on multiple devices. Use `get_user_sessions()` to retrieve all active sessions:

```python
# List user's active sessions
sessions = await storage.get_user_sessions(user_id)

# Display in user settings
for session in sessions:
    print(f"Device: {session.user_agent}")
    print(f"Last active: {session.last_accessed}")
    print(f"IP: {session.ip_address}")
```

### How do I handle session fixation attacks?

Regenerate session tokens after privilege escalation (login, permission changes):

```python
async def login(username: str, password: str, storage: SessionStorage):
    # Validate credentials
    user = await authenticate(username, password)

    # Generate NEW token after successful login
    new_token = secrets.token_urlsafe(32)

    session = SessionData(
        id=new_token,
        user_id=user.id,
        # ...
    )

    await storage.store_session(new_token, session)

    # Never reuse pre-authentication tokens
    return new_token
```

### Should I encrypt session data?

The session token itself should be:
- Generated with cryptographically secure random (✅ `secrets.token_urlsafe()`)
- Transmitted only over HTTPS (✅ `secure=True` cookie flag)
- HTTP-only to prevent JavaScript access (✅ `httponly=True`)

Session *data* (SessionData fields) typically doesn't need encryption if:
- Tokens are stored securely
- Storage backend is secure (database, Redis with auth)
- Data is minimal (just user_id and metadata)

Encrypt session data if:
- Storing sensitive PII in session
- Regulatory requirements demand it
- Using untrusted storage backend
