# Audit Kit

## Overview

**Purpose**: Provide comprehensive audit logging and compliance capabilities for tracking user actions, resource changes, and security events in your application.

**Key Features**:

- Log user actions with comprehensive metadata (IP, user agent, session)
- Search and query audit events with flexible criteria
- Track resource history and user activity
- Generate compliance reports and summaries
- Automatic retention policy enforcement
- Group-based audit scoping for multi-tenant applications
- Transactional consistency with application operations

**Dependencies**:

- **Injected services**: None
- **Port dependencies**: AuditAdapter (database storage interface)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose

# Basic configuration
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.audit(
            enable_auditing=True,
            retention_days=90,
        ),
    ]
)

# Access the audit service
audit_service = app.kits["audit"].service

# Log an audit event
await audit_service.log_event(
    user_id=user.id,
    action="create",
    resource_type="document",
    resource_id=str(document.id),
    details={"title": document.title},
)

# Search audit events
from portico.ports.audit import AuditQuery

events = await audit_service.search_events(
    AuditQuery(user_id=user.id, limit=50)
)
```

## Core Concepts

### Audit Events

Every audit event captures comprehensive metadata about an action:

```python
from portico.ports.audit import AuditAction

# Log a create action
await audit_service.log_event(
    user_id=user.id,
    action=AuditAction.CREATE,  # CREATE, READ, UPDATE, DELETE, LOGIN, etc.
    resource_type="user",
    resource_id=str(new_user.id),
    details={"email": new_user.email, "role": "member"},
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    session_id="session_abc123",
    success=True,
)

# Log a failed action
await audit_service.log_event(
    user_id=user.id,
    action=AuditAction.DELETE,
    resource_type="document",
    resource_id=str(doc_id),
    success=False,
    error_message="Permission denied",
)
```

Each event includes:

- **Who**: `user_id` and optional `group_id` for multi-tenant scoping
- **What**: `action` (enum) and `resource_type`/`resource_id`
- **When**: `timestamp` (auto-generated)
- **Where**: `ip_address`, `user_agent`, `session_id`
- **How**: `success` flag and optional `error_message`
- **Why**: `details` dict for custom metadata

### Standard Audit Actions

The `AuditAction` enum provides standard actions for consistency:

```python
from portico.ports.audit import AuditAction

# Standard CRUD operations
AuditAction.CREATE   # Resource created
AuditAction.READ     # Resource accessed
AuditAction.UPDATE   # Resource modified
AuditAction.DELETE   # Resource deleted

# Authentication events
AuditAction.LOGIN    # User logged in
AuditAction.LOGOUT   # User logged out

# Data transfer
AuditAction.EXPORT   # Data exported
AuditAction.IMPORT   # Data imported

# Workflow actions
AuditAction.APPROVE  # Request approved
AuditAction.REJECT   # Request rejected

# You can also use custom strings
await audit_service.log_event(
    action="archive",  # Converted to AuditAction if matches enum
    resource_type="document",
)
```

### Transactional Auditing

Audit events can participate in database transactions for consistency:

```python
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@app.post("/users")
async def create_user(
    user_data: CreateUserRequest,
    session: AsyncSession = deps.session,
):
    user_service = app.kits["user"].service
    audit_service = app.kits["audit"].service

    # Create user and log audit event in same transaction
    async with session.begin():
        user = await user_service.create_user(user_data)

        # Use db_session param for transactional consistency
        await audit_service.log_event(
            user_id=None,  # System action
            action=AuditAction.CREATE,
            resource_type="user",
            resource_id=str(user.id),
            details={"email": user.email},
            db_session=session,  # Same transaction
        )

    # Both committed together - no orphaned audit events
    return {"user_id": str(user.id)}
```

## Configuration

### Required Settings

None - all settings have sensible defaults.

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_auditing` | `bool` | `True` | Whether to log audit events (disable for testing) |
| `retention_days` | `int` | `90` | Days to retain audit events before cleanup |

**Example:**

```python
from portico import compose

app = compose.webapp(
    kits=[
        compose.audit(
            enable_auditing=True,
            retention_days=365,  # Keep for 1 year
        ),
    ]
)
```

## Usage Examples

### Example 1: Logging User Actions

```python
@app.post("/documents/{doc_id}/share")
async def share_document(
    doc_id: UUID,
    share_with: UUID,
    current_user: User = deps.current_user,
):
    audit_service = app.kits["audit"].service
    document_service = app.kits["document"].service

    # Perform action
    await document_service.share(doc_id, share_with)

    # Log audit event
    await audit_service.log_event(
        user_id=current_user.id,
        action="share",
        resource_type="document",
        resource_id=str(doc_id),
        details={"shared_with": str(share_with)},
    )

    return {"success": True}
```

### Example 2: Viewing User Activity

```python
@app.get("/admin/users/{user_id}/activity")
async def get_user_activity(
    user_id: UUID,
    days: int = 30,
    current_user: User = deps.current_user,
):
    audit_service = app.kits["audit"].service

    # Get recent activity
    events = await audit_service.get_user_activity(user_id, days=days)

    return {
        "user_id": str(user_id),
        "activity": [
            {
                "action": event.action.value,
                "resource_type": event.resource_type,
                "resource_id": event.resource_id,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success,
            }
            for event in events
        ]
    }
```

### Example 3: Compliance Reporting

```python
from datetime import datetime, timedelta

@app.get("/admin/reports/audit-summary")
async def audit_summary(current_user: User = deps.current_user):
    audit_service = app.kits["audit"].service

    # Generate summary for last 30 days
    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=30)

    summary = await audit_service.generate_summary(start_date, end_date)

    return {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "total_events": summary.total_events,
        "success_rate": summary.success_rate,
        "by_action": summary.events_by_action,
        "by_user": summary.events_by_user,
        "by_resource": summary.events_by_resource_type,
    }
```

## Domain Models

### AuditEvent

Represents a logged audit event with comprehensive metadata.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | Auto | Unique event identifier |
| `user_id` | `Optional[UUID]` | No | `None` | User who performed the action |
| `group_id` | `Optional[UUID]` | No | `None` | Group scope for multi-tenant filtering |
| `action` | `AuditAction` | Yes | - | Action performed (enum) |
| `resource_type` | `str` | Yes | - | Type of resource affected |
| `resource_id` | `Optional[str]` | No | `None` | Identifier of the resource |
| `details` | `Dict[str, Any]` | No | `{}` | Additional custom metadata |
| `ip_address` | `Optional[str]` | No | `None` | Client IP address |
| `user_agent` | `Optional[str]` | No | `None` | Client user agent string |
| `session_id` | `Optional[str]` | No | `None` | Session identifier |
| `timestamp` | `datetime` | Yes | Auto | When the event occurred (UTC) |
| `success` | `bool` | No | `True` | Whether the action succeeded |
| `error_message` | `Optional[str]` | No | `None` | Error message if failed |

### AuditQuery

Search criteria for querying audit events.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `user_id` | `Optional[UUID]` | `None` | Filter by single user |
| `user_ids` | `Optional[List[UUID]]` | `None` | Filter by multiple users |
| `group_id` | `Optional[UUID]` | `None` | Filter by group scope |
| `group_ids` | `Optional[List[UUID]]` | `None` | Filter by multiple groups |
| `action` | `Optional[AuditAction]` | `None` | Filter by action type |
| `actions` | `Optional[List[str]]` | `None` | Filter by multiple actions |
| `resource_type` | `Optional[str]` | `None` | Filter by resource type |
| `resource_types` | `Optional[List[str]]` | `None` | Filter by multiple resource types |
| `resource_id` | `Optional[str]` | `None` | Filter by specific resource |
| `start_date` | `Optional[datetime]` | `None` | Start of date range |
| `end_date` | `Optional[datetime]` | `None` | End of date range |
| `success` | `Optional[bool]` | `None` | Filter by success/failure |
| `limit` | `int` | `100` | Maximum results (1-1000) |
| `offset` | `int` | `0` | Pagination offset |

### AuditSummary

Summary statistics for audit events over a time period.

| Field | Type | Description |
|-------|------|-------------|
| `total_events` | `int` | Total number of events |
| `events_by_action` | `Dict[str, int]` | Count per action type |
| `events_by_user` | `Dict[str, int]` | Count per user |
| `events_by_resource_type` | `Dict[str, int]` | Count per resource type |
| `events_by_group` | `Dict[str, int]` | Count per group (multi-tenant) |
| `success_rate` | `float` | Percentage of successful events (0.0-1.0) |
| `time_range` | `Tuple[datetime, datetime]` | Start and end of summary period |

### AuditAction

Enumeration of standard audit actions.

| Value | Description |
|-------|-------------|
| `CREATE` | Resource created |
| `READ` | Resource accessed or viewed |
| `UPDATE` | Resource modified |
| `DELETE` | Resource deleted |
| `LOGIN` | User logged in |
| `LOGOUT` | User logged out |
| `EXPORT` | Data exported |
| `IMPORT` | Data imported |
| `APPROVE` | Request or action approved |
| `REJECT` | Request or action rejected |

## Events

This kit publishes the following events:

### `AuditEventLoggedEvent`

**Triggered when**: An audit event is successfully logged.

**Payload**:

```python
{
    "event_type": "audit.event_logged",
    "data": {
        "audit_event_id": "uuid-of-audit-event",
        "user_id": "uuid-of-user",
        "action": "create",
        "resource_type": "document",
        "resource_id": "resource-id",
        "timestamp": "2025-01-15T10:30:00Z"
    }
}
```

**Listeners**: Other kits can listen to audit events for real-time alerting, anomaly detection, or compliance workflows.

**Example listener:**

```python
from portico.events import EventBus
from portico.kits.audit.events import AuditEventLoggedEvent

async def alert_on_failed_login(event: AuditEventLoggedEvent):
    if event.action == "login" and not event.success:
        # Send security alert
        await send_security_notification(event.user_id)

# Register listener
events: EventBus = app.events
await events.subscribe(AuditEventLoggedEvent, alert_on_failed_login)
```

## Best Practices

### 1. Always Log State-Changing Operations

Log all operations that create, modify, or delete data:

```python
# ✅ GOOD - Log all state changes
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: UUID, current_user: User = deps.current_user):
    await document_service.delete(doc_id)

    await audit_service.log_event(
        user_id=current_user.id,
        action=AuditAction.DELETE,
        resource_type="document",
        resource_id=str(doc_id),
    )

    return {"success": True}

# ❌ BAD - Missing audit logging
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: UUID):
    await document_service.delete(doc_id)
    return {"success": True}  # No audit trail!
```

### 2. Include Request Context

Capture IP address, user agent, and session for forensics:

```python
from fastapi import Request

# ✅ GOOD - Include request metadata
@app.post("/api/transfer")
async def transfer_funds(
    request: Request,
    amount: float,
    current_user: User = deps.current_user,
):
    await audit_service.log_event(
        user_id=current_user.id,
        action="transfer",
        resource_type="account",
        details={"amount": amount},
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
    )

# ❌ BAD - Missing forensic context
@app.post("/api/transfer")
async def transfer_funds(amount: float, current_user: User = deps.current_user):
    await audit_service.log_event(
        user_id=current_user.id,
        action="transfer",
        resource_type="account",
    )  # Can't trace back to client
```

### 3. Use Transactional Auditing for Consistency

Keep audit events in the same transaction as the operation:

```python
# ✅ GOOD - Transactional consistency
async def create_user_with_audit(
    user_data: CreateUserRequest,
    session: AsyncSession,
):
    async with session.begin():
        user = await user_service.create_user(user_data)
        await audit_service.log_event(
            action=AuditAction.CREATE,
            resource_type="user",
            resource_id=str(user.id),
            db_session=session,  # Same transaction
        )
    # Both committed or both rolled back

# ❌ BAD - Audit event may succeed even if operation fails
async def create_user_with_audit(user_data: CreateUserRequest):
    user = await user_service.create_user(user_data)  # May fail
    await audit_service.log_event(...)  # Separate transaction
```

### 4. Log Both Success and Failure

Record failed attempts for security monitoring:

```python
# ✅ GOOD - Log failures with error details
@app.post("/admin/delete-all")
async def dangerous_operation(current_user: User = deps.current_user):
    try:
        await perform_deletion()
        await audit_service.log_event(
            user_id=current_user.id,
            action=AuditAction.DELETE,
            resource_type="system",
            success=True,
        )
    except PermissionError as e:
        await audit_service.log_event(
            user_id=current_user.id,
            action=AuditAction.DELETE,
            resource_type="system",
            success=False,
            error_message=str(e),
        )
        raise

# ❌ BAD - Only logging success
@app.post("/admin/delete-all")
async def dangerous_operation(current_user: User = deps.current_user):
    await perform_deletion()
    await audit_service.log_event(
        user_id=current_user.id,
        action=AuditAction.DELETE,
        resource_type="system",
    )  # Failures not tracked
```

### 5. Use Meaningful Details

Include context-specific information in the details field:

```python
# ✅ GOOD - Rich details for investigation
await audit_service.log_event(
    user_id=current_user.id,
    action=AuditAction.UPDATE,
    resource_type="user_profile",
    resource_id=str(user.id),
    details={
        "changed_fields": ["email", "phone"],
        "old_email": "old@example.com",
        "new_email": "new@example.com",
        "reason": "user_requested",
    },
)

# ❌ BAD - Insufficient context
await audit_service.log_event(
    user_id=current_user.id,
    action=AuditAction.UPDATE,
    resource_type="user_profile",
    details={},  # Can't determine what changed
)
```

### 6. Implement Regular Cleanup

Schedule cleanup jobs to enforce retention policy:

```python
# ✅ GOOD - Scheduled cleanup task
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

async def cleanup_old_audit_events():
    audit_service = app.kits["audit"].service
    count = await audit_service.cleanup_old_events()
    logger.info("cleaned_up_audit_events", count=count)

# Run daily at 2 AM
scheduler.add_job(cleanup_old_audit_events, "cron", hour=2)
scheduler.start()

# ❌ BAD - No cleanup, database grows unbounded
# Audit table fills disk space over time
```

### 7. Use Group Scoping for Multi-Tenancy

Include `group_id` for tenant isolation in multi-tenant applications:

```python
# ✅ GOOD - Multi-tenant audit scoping
@app.post("/orgs/{org_id}/projects")
async def create_project(
    org_id: UUID,
    project_data: CreateProjectRequest,
    current_user: User = deps.current_user,
):
    project = await project_service.create(org_id, project_data)

    await audit_service.log_event(
        user_id=current_user.id,
        group_id=org_id,  # Tenant scope
        action=AuditAction.CREATE,
        resource_type="project",
        resource_id=str(project.id),
    )

# Query org-specific events
events = await audit_service.search_events(
    AuditQuery(group_id=org_id)  # Only events for this org
)

# ❌ BAD - No tenant scoping
await audit_service.log_event(
    user_id=current_user.id,
    action=AuditAction.CREATE,
    resource_type="project",
)  # Can't filter by organization
```

## Security Considerations

### Data Sensitivity

Audit events may contain sensitive information. Consider:

- **Encrypt audit data at rest** if storing PII or sensitive details
- **Restrict access** to audit logs (admin-only endpoints)
- **Redact sensitive fields** in details (passwords, tokens, SSNs)
- **Use separate database** for audit events in high-security environments

### Immutability

Audit events should be **immutable** after creation:

- No update or delete operations on individual events
- Only bulk cleanup based on retention policy
- Consider write-only database roles for audit tables
- Use event sourcing patterns for tamper-proof logs

### Access Control

Protect audit endpoints with proper authorization:

```python
from portico.kits.fastapi import requires_permission

@app.get("/admin/audit/search")
@requires_permission(app, "audit.read")
async def search_audit(
    query: AuditQuery,
    current_user: User = deps.current_user,
):
    return await audit_service.search_events(query)
```

## FAQs

### Q: Should I log read operations?

A: For sensitive data (medical records, financial info), yes. For public data, it's optional. Use `AuditAction.READ` sparingly to avoid overwhelming your audit log.

### Q: How do I handle high-volume audit logging?

A: Use asynchronous logging (default), consider batch inserts for very high volumes, or use a dedicated audit database. For extreme cases, queue audit events to a message broker and process asynchronously.

### Q: Can I disable auditing for specific operations?

A: Yes, use `enable_auditing=False` in config, or conditionally skip `log_event()` calls:

```python
if operation.is_sensitive:
    await audit_service.log_event(...)
```

### Q: How do I query audit events for multiple users?

A: Use `AuditQuery.user_ids` to filter by multiple users:

```python
query = AuditQuery(
    user_ids=[user1_id, user2_id, user3_id],
    start_date=start,
    end_date=end,
)
events = await audit_service.search_events(query)
```

### Q: What happens if audit logging fails?

A: If using transactional auditing (`db_session` parameter), the entire transaction rolls back. For non-transactional logging, the exception propagates - catch it if you want operations to succeed despite audit failures.

### Q: How do I export audit logs for compliance?

A: Use `search_events()` with date ranges and export to CSV/JSON:

```python
query = AuditQuery(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    limit=1000,
)
events = await audit_service.search_events(query)

# Export to CSV for compliance officer
import csv
with open("audit_2024.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "user", "action", "resource"])
    for event in events:
        writer.writerow({...})
```

### Q: Can I track who viewed specific resources?

A: Yes, use resource history:

```python
history = await audit_service.get_resource_history(
    resource_type="medical_record",
    resource_id="patient-123",
)
# Shows all actions on this resource
```

### Q: What's the performance impact of audit logging?

A: Minimal for most applications. Each `log_event()` is a single async insert. For high-traffic apps (1000+ req/s), consider batching or queuing audit events.
