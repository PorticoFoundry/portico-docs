"""Test examples for Audit port documentation."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from portico.ports.audit import (
    AuditAction,
    AuditAdapter,
    AuditEvent,
    AuditQuery,
    AuditSummary,
)


class MockAuditAdapter(AuditAdapter):
    """Mock audit adapter for testing."""

    def __init__(self):
        self.events = []

    async def log_event(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def search_events(self, query: AuditQuery) -> list[AuditEvent]:
        results = self.events.copy()

        if query.user_id:
            results = [e for e in results if e.user_id == query.user_id]
        if query.action:
            results = [e for e in results if e.action == query.action]
        if query.resource_type:
            results = [e for e in results if e.resource_type == query.resource_type]
        if query.success is not None:
            results = [e for e in results if e.success == query.success]

        return results[query.offset : query.offset + query.limit]

    async def get_user_activity(self, user_id, days=30) -> list[AuditEvent]:
        return [e for e in self.events if e.user_id == user_id]

    async def get_resource_history(
        self, resource_type: str, resource_id: str
    ) -> list[AuditEvent]:
        return [
            e
            for e in self.events
            if e.resource_type == resource_type and e.resource_id == resource_id
        ]

    async def generate_summary(self, start_date, end_date) -> AuditSummary:
        events = [e for e in self.events if start_date <= e.timestamp <= end_date]

        events_by_action = {}
        for event in events:
            action = event.action.value
            events_by_action[action] = events_by_action.get(action, 0) + 1

        success_count = sum(1 for e in events if e.success)
        success_rate = success_count / len(events) if events else 0.0

        return AuditSummary(
            total_events=len(events),
            events_by_action=events_by_action,
            events_by_user={},
            events_by_resource_type={},
            events_by_group={},
            success_rate=success_rate,
            time_range=(start_date, end_date),
        )

    async def cleanup_old_events(self, older_than_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        old_events = [e for e in self.events if e.timestamp < cutoff]
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        return len(old_events)


# --8<-- [start:basic-usage]
@pytest.mark.asyncio
async def test_basic_audit_logging():
    """Log basic audit events."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Log a user action
    event = AuditEvent(
        user_id=user_id,
        action=AuditAction.CREATE,
        resource_type="document",
        resource_id="doc-123",
        details={"title": "My Document"},
    )

    await audit.log_event(event)

    # Verify event was logged
    events = await audit.search_events(AuditQuery(user_id=user_id))
    assert len(events) == 1
    assert events[0].action == AuditAction.CREATE


# --8<-- [end:basic-usage]


# --8<-- [start:login-tracking]
@pytest.mark.asyncio
async def test_login_tracking():
    """Track user authentication events."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Successful login
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.LOGIN,
            resource_type="auth",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0...",
            session_id="sess-abc123",
            success=True,
        )
    )

    # Failed login attempt
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.LOGIN,
            resource_type="auth",
            ip_address="192.168.1.100",
            success=False,
            error_message="Invalid password",
        )
    )

    # Query login events
    logins = await audit.search_events(
        AuditQuery(user_id=user_id, action=AuditAction.LOGIN)
    )
    assert len(logins) == 2


# --8<-- [end:login-tracking]


# --8<-- [start:resource-changes]
@pytest.mark.asyncio
async def test_track_resource_changes():
    """Track changes to a specific resource."""
    audit = MockAuditAdapter()
    user_id = uuid4()
    resource_id = "doc-456"

    # Create
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="document",
            resource_id=resource_id,
            details={"title": "Draft"},
        )
    )

    # Update
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.UPDATE,
            resource_type="document",
            resource_id=resource_id,
            details={"title": "Final", "status": "published"},
        )
    )

    # Get complete history
    history = await audit.get_resource_history("document", resource_id)
    assert len(history) == 2
    assert history[0].action == AuditAction.CREATE
    assert history[1].action == AuditAction.UPDATE


# --8<-- [end:resource-changes]


# --8<-- [start:search-events]
@pytest.mark.asyncio
async def test_search_audit_events():
    """Search audit events with filters."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Log various events
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="document",
            resource_id="doc-1",
        )
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type="document",
            resource_id="doc-2",
        )
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="user",
            resource_id="user-3",
        )
    )

    # Search by resource type
    doc_events = await audit.search_events(AuditQuery(resource_type="document"))
    assert len(doc_events) == 2

    # Search by action
    create_events = await audit.search_events(AuditQuery(action=AuditAction.CREATE))
    assert len(create_events) == 2


# --8<-- [end:search-events]


# --8<-- [start:user-activity]
@pytest.mark.asyncio
async def test_get_user_activity():
    """Get recent activity for a user."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Log user actions
    await audit.log_event(
        AuditEvent(user_id=user_id, action=AuditAction.LOGIN, resource_type="auth")
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="document",
            resource_id="doc-1",
        )
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.UPDATE,
            resource_type="document",
            resource_id="doc-1",
        )
    )

    # Get activity for last 30 days
    activity = await audit.get_user_activity(user_id, days=30)
    assert len(activity) == 3


# --8<-- [end:user-activity]


# --8<-- [start:audit-summary]
@pytest.mark.asyncio
async def test_generate_audit_summary():
    """Generate audit summary statistics."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Log events
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="document",
            success=True,
        )
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.CREATE,
            resource_type="document",
            success=True,
        )
    )
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type="document",
            success=False,
            error_message="Permission denied",
        )
    )

    # Generate summary
    now = datetime.now(UTC)
    summary = await audit.generate_summary(
        start_date=now - timedelta(days=30), end_date=now
    )

    assert summary.total_events == 3
    assert summary.events_by_action["create"] == 2
    assert summary.events_by_action["delete"] == 1
    assert summary.success_rate == 2 / 3  # 2 out of 3 succeeded


# --8<-- [end:audit-summary]


# --8<-- [start:failed-operations]
@pytest.mark.asyncio
async def test_track_failed_operations():
    """Track failed operations for security monitoring."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Log failed attempts
    await audit.log_event(
        AuditEvent(
            user_id=user_id,
            action=AuditAction.DELETE,
            resource_type="document",
            resource_id="doc-1",
            success=False,
            error_message="Permission denied",
        )
    )

    # Query only failures
    failures = await audit.search_events(AuditQuery(user_id=user_id, success=False))
    assert len(failures) == 1
    assert failures[0].error_message == "Permission denied"


# --8<-- [end:failed-operations]


# --8<-- [start:cleanup-old-events]
@pytest.mark.asyncio
async def test_cleanup_old_events():
    """Clean up old audit events."""
    audit = MockAuditAdapter()

    # Create old event
    old_event = AuditEvent(
        user_id=uuid4(),
        action=AuditAction.CREATE,
        resource_type="document",
        timestamp=datetime.now(UTC) - timedelta(days=400),
    )
    old_event = old_event.model_copy(
        update={"timestamp": datetime.now(UTC) - timedelta(days=400)}
    )
    audit.events.append(old_event)

    # Create recent event
    await audit.log_event(
        AuditEvent(user_id=uuid4(), action=AuditAction.CREATE, resource_type="document")
    )

    # Cleanup events older than 365 days
    removed = await audit.cleanup_old_events(older_than_days=365)

    assert removed == 1
    assert len(audit.events) == 1


# --8<-- [end:cleanup-old-events]


# --8<-- [start:pagination]
@pytest.mark.asyncio
async def test_paginate_results():
    """Paginate through large result sets."""
    audit = MockAuditAdapter()
    user_id = uuid4()

    # Create many events
    for i in range(150):
        await audit.log_event(
            AuditEvent(
                user_id=user_id,
                action=AuditAction.CREATE,
                resource_type="document",
                resource_id=f"doc-{i}",
            )
        )

    # Get first page (100 results)
    page1 = await audit.search_events(AuditQuery(user_id=user_id, limit=100, offset=0))
    assert len(page1) == 100

    # Get second page (50 results)
    page2 = await audit.search_events(
        AuditQuery(user_id=user_id, limit=100, offset=100)
    )
    assert len(page2) == 50


# --8<-- [end:pagination]
