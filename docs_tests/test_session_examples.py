"""Test examples for Session port documentation."""

from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

import pytest

from portico.ports.session import (
    SessionData,
    SessionStorage,
)


class MockSessionStorage(SessionStorage):
    """Mock session storage for testing."""

    def __init__(self):
        self.sessions: dict[str, SessionData] = {}

    async def store_session(self, token: str, session_data: SessionData) -> None:
        """Store session data with the given token."""
        # Store with token as the session ID
        updated_session = session_data.model_copy(update={"id": token})
        self.sessions[token] = updated_session

    async def get_session(self, token: str) -> Optional[SessionData]:
        """Retrieve session data by token."""
        return self.sessions.get(token)

    async def delete_session(self, token: str) -> bool:
        """Delete session by token. Returns True if session existed."""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False

    async def delete_user_sessions(self, user_id: UUID) -> int:
        """Delete all sessions for a user. Returns number of sessions deleted."""
        to_delete = [
            token
            for token, session in self.sessions.items()
            if session.user_id == user_id
        ]

        for token in to_delete:
            del self.sessions[token]

        return len(to_delete)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions. Returns number of sessions cleaned up."""
        now = datetime.now(UTC)
        expired = [
            token
            for token, session in self.sessions.items()
            if session.expires_at <= now
        ]

        for token in expired:
            del self.sessions[token]

        return len(expired)

    async def get_user_sessions(self, user_id: UUID) -> list[SessionData]:
        """Get all active sessions for a user."""
        now = datetime.now(UTC)
        return [
            session
            for session in self.sessions.values()
            if session.user_id == user_id and session.expires_at > now
        ]


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_session_creation():
    """Create and retrieve a basic session."""
    storage = MockSessionStorage()
    user_id = uuid4()
    token = "session_token_123"

    # Create session data
    session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        last_accessed=datetime.now(UTC),
    )

    # Store session
    await storage.store_session(token, session)

    # Retrieve session
    retrieved = await storage.get_session(token)
    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.has_valid_token is True


# --8<-- [end:basic-usage]


# --8<-- [start:session-metadata]
@pytest.mark.asyncio
async def test_session_with_metadata():
    """Create session with IP and user agent tracking."""
    storage = MockSessionStorage()
    user_id = uuid4()
    token = "session_token_456"

    # Create session with metadata
    session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        last_accessed=datetime.now(UTC),
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    )

    await storage.store_session(token, session)

    # Retrieve and verify metadata
    retrieved = await storage.get_session(token)
    assert retrieved is not None
    assert retrieved.ip_address == "192.168.1.100"
    assert "Mozilla" in retrieved.user_agent


# --8<-- [end:session-metadata]


# --8<-- [start:check-expiration]
@pytest.mark.asyncio
async def test_check_session_expiration():
    """Check if a session is expired."""
    storage = MockSessionStorage()
    user_id = uuid4()

    # Create expired session
    expired_session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC) - timedelta(hours=2),
        expires_at=datetime.now(UTC) - timedelta(hours=1),
        last_accessed=datetime.now(UTC) - timedelta(hours=1),
    )

    # Create valid session
    valid_session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        last_accessed=datetime.now(UTC),
    )

    assert expired_session.is_expired is True
    assert valid_session.is_expired is False


# --8<-- [end:check-expiration]


# --8<-- [start:touch-session]
@pytest.mark.asyncio
async def test_touch_session():
    """Update last accessed time and extend expiration."""
    storage = MockSessionStorage()
    user_id = uuid4()
    token = "session_token_789"

    # Create session
    session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        last_accessed=datetime.now(UTC),
    )

    await storage.store_session(token, session)

    # Touch session to extend expiration
    updated = session.touch(extend_by=timedelta(hours=1))

    # Store updated session
    await storage.store_session(token, updated)

    # Verify expiration was extended
    retrieved = await storage.get_session(token)
    assert retrieved is not None
    assert retrieved.expires_at > session.expires_at


# --8<-- [end:touch-session]


# --8<-- [start:delete-session]
@pytest.mark.asyncio
async def test_delete_session():
    """Delete a session (logout)."""
    storage = MockSessionStorage()
    user_id = uuid4()
    token = "session_token_abc"

    # Create session
    session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        last_accessed=datetime.now(UTC),
    )

    await storage.store_session(token, session)

    # Delete session
    deleted = await storage.delete_session(token)
    assert deleted is True

    # Verify session is gone
    retrieved = await storage.get_session(token)
    assert retrieved is None

    # Try deleting again
    deleted_again = await storage.delete_session(token)
    assert deleted_again is False


# --8<-- [end:delete-session]


# --8<-- [start:user-sessions]
@pytest.mark.asyncio
async def test_get_user_sessions():
    """Get all active sessions for a user."""
    storage = MockSessionStorage()
    user_id = uuid4()

    # Create multiple sessions for the same user
    for i in range(3):
        session = SessionData(
            user_id=user_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            last_accessed=datetime.now(UTC),
            ip_address=f"192.168.1.{100 + i}",
        )
        await storage.store_session(f"token_{i}", session)

    # Get all user sessions
    sessions = await storage.get_user_sessions(user_id)
    assert len(sessions) == 3


# --8<-- [end:user-sessions]


# --8<-- [start:delete-user-sessions]
@pytest.mark.asyncio
async def test_delete_all_user_sessions():
    """Delete all sessions for a user (logout everywhere)."""
    storage = MockSessionStorage()
    user_id = uuid4()

    # Create multiple sessions
    for i in range(5):
        session = SessionData(
            user_id=user_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            last_accessed=datetime.now(UTC),
        )
        await storage.store_session(f"token_{i}", session)

    # Delete all user sessions
    deleted_count = await storage.delete_user_sessions(user_id)
    assert deleted_count == 5

    # Verify all sessions are gone
    sessions = await storage.get_user_sessions(user_id)
    assert len(sessions) == 0


# --8<-- [end:delete-user-sessions]


# --8<-- [start:cleanup-expired]
@pytest.mark.asyncio
async def test_cleanup_expired_sessions():
    """Clean up expired sessions."""
    storage = MockSessionStorage()

    # Create expired sessions
    for i in range(3):
        expired_session = SessionData(
            user_id=uuid4(),
            created_at=datetime.now(UTC) - timedelta(hours=2),
            expires_at=datetime.now(UTC) - timedelta(hours=1),
            last_accessed=datetime.now(UTC) - timedelta(hours=1),
        )
        await storage.store_session(f"expired_{i}", expired_session)

    # Create active sessions
    for i in range(2):
        active_session = SessionData(
            user_id=uuid4(),
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            last_accessed=datetime.now(UTC),
        )
        await storage.store_session(f"active_{i}", active_session)

    # Cleanup expired sessions
    cleaned = await storage.cleanup_expired_sessions()
    assert cleaned == 3

    # Verify only active sessions remain
    assert len(storage.sessions) == 2


# --8<-- [end:cleanup-expired]


# --8<-- [start:concurrent-sessions]
@pytest.mark.asyncio
async def test_concurrent_session_limit():
    """Limit concurrent sessions per user."""
    storage = MockSessionStorage()
    user_id = uuid4()
    max_sessions = 3

    # Create sessions up to limit
    tokens = []
    for i in range(5):
        session = SessionData(
            user_id=user_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            last_accessed=datetime.now(UTC),
        )
        token = f"token_{i}"
        await storage.store_session(token, session)
        tokens.append(token)

        # Enforce limit by deleting oldest session
        sessions = await storage.get_user_sessions(user_id)
        if len(sessions) > max_sessions:
            # Delete oldest session (first token)
            oldest_token = tokens[i - max_sessions]
            await storage.delete_session(oldest_token)

    # Verify only max_sessions remain
    final_sessions = await storage.get_user_sessions(user_id)
    assert len(final_sessions) == max_sessions


# --8<-- [end:concurrent-sessions]


# --8<-- [start:session-validation]
@pytest.mark.asyncio
async def test_session_validation():
    """Validate session token format."""
    user_id = uuid4()

    # Valid session
    valid_session = SessionData(
        id="valid_token_123",
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        last_accessed=datetime.now(UTC),
    )
    assert valid_session.has_valid_token is True

    # Session without token
    no_token_session = SessionData(
        user_id=user_id,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        last_accessed=datetime.now(UTC),
    )
    assert no_token_session.has_valid_token is False

    # Invalid empty string token should raise error
    with pytest.raises(ValueError, match="Session ID cannot be empty string"):
        SessionData(
            id="",
            user_id=user_id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            last_accessed=datetime.now(UTC),
        )


# --8<-- [end:session-validation]
