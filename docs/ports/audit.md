# Audit Port

## Overview

The Audit Port defines the contract for audit logging operations in Portico applications. It provides a comprehensive auditing system for tracking user actions, system events, and compliance requirements with support for flexible querying, reporting, and retention policies.

**Purpose**: Abstract audit logging operations to enable compliance, security tracking, and activity monitoring with pluggable storage backends.

**Domain**: Security, compliance, observability

**Key Capabilities**:

- Comprehensive event logging with rich metadata
- User activity tracking and reporting
- Resource history and change tracking
- Flexible querying with multiple filter criteria
- Compliance reporting and summary statistics
- Retention policy management and cleanup
- Group-scoped audit trails for multi-tenant applications
- Transactional audit logging for data consistency

**Port Type**: Adapter (infrastructure abstraction)

**When to Use**:

- Applications requiring compliance with audit regulations (SOC 2, HIPAA, GDPR)
- Systems needing security event tracking and forensics
- Multi-tenant applications requiring isolated audit trails
- Applications tracking user activity and resource changes
- Systems requiring detailed activity reporting and analytics
- Services needing historical audit data for investigations

## Architecture Role

The Audit Port sits at the boundary between your application's business logic (kits) and audit storage infrastructure. It enables kits to log audit events without depending on specific storage technologies.

```
┌─────────────────────────────────────────┐
│  Kits (Business Logic)                  │
│  - AuditKit wraps AuditAdapter          │
│  - Other kits log audit events          │
└─────────────────┬───────────────────────┘
                  │ depends on
                  ↓
┌─────────────────────────────────────────┐
│  Audit Port (Interface)                 │
│  - AuditAdapter (ABC)                   │
│  - AuditEvent, AuditQuery, AuditSummary │
│  - AuditAction (Enum)                   │
└─────────────────┬───────────────────────┘
                  ↑ implements
                  │
┌─────────────────────────────────────────┐
│  Adapters (Implementations)             │
│  - SqlAlchemyAuditAdapter (database)    │
│  - StructuredLoggingAudit (logs)        │
│  - MemoryAudit (testing)                │
│  - CompositeAudit (multi-destination)   │
└─────────────────────────────────────────┘
```

**Key Responsibilities**:

- Define audit event structure with comprehensive metadata
- Specify audit query capabilities for filtering and searching
- Provide audit summary statistics for reporting
- Abstract storage operations (log, search, query, cleanup)
- Support compliance requirements (retention, reporting)
- Enable multi-tenant audit isolation via group scoping

## Domain Models

### AuditEvent

Represents an audit event with comprehensive metadata for compliance and security tracking.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | Auto-generated | Unique event identifier |
| `user_id` | `Optional[UUID]` | No | `None` | User who performed the action |
| `group_id` | `Optional[UUID]` | No | `None` | Group scope for multi-tenant isolation |
| `action` | `AuditAction` | Yes | - | Action performed (CREATE, UPDATE, DELETE, etc.) |
| `resource_type` | `str` | Yes | - | Type of resource affected (e.g., "user", "document") |
| `resource_id` | `Optional[str]` | No | `None` | Identifier of the affected resource |
| `details` | `Dict[str, Any]` | No | `{}` | Additional context and metadata |
| `ip_address` | `Optional[str]` | No | `None` | Client IP address |
| `user_agent` | `Optional[str]` | No | `None` | Client user agent string |
| `timestamp` | `datetime` | No | Current UTC time | When the event occurred |
| `session_id` | `Optional[str]` | No | `None` | Session identifier |
| `success` | `bool` | No | `True` | Whether the action succeeded |
| `error_message` | `Optional[str]` | No | `None` | Error message if action failed |

**Example**:

```python
from portico.ports.audit import AuditEvent, AuditAction

event = AuditEvent(
    user_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
    group_id=UUID("987fcdeb-51a2-43f7-9876-543210fedcba"),
    action=AuditAction.UPDATE,
    resource_type="document",
    resource_id="doc-456",
    details={
        "field_changed": "title",
        "old_value": "Draft Document",
        "new_value": "Final Document"
    },
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    session_id="sess_abc123",
    success=True
)
```

**Immutability**: `AuditEvent` is frozen (immutable) - all fields are set at creation time to ensure audit integrity.

### AuditQuery

Query parameters for searching and filtering audit events.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | `Optional[UUID]` | No | `None` | Filter by single user |
| `user_ids` | `Optional[List[UUID]]` | No | `None` | Filter by multiple users |
| `group_id` | `Optional[UUID]` | No | `None` | Filter by single group |
| `group_ids` | `Optional[List[UUID]]` | No | `None` | Filter by multiple groups (e.g., user's accessible groups) |
| `action` | `Optional[AuditAction]` | No | `None` | Filter by single action |
| `actions` | `Optional[List[str]]` | No | `None` | Filter by multiple actions |
| `resource_type` | `Optional[str]` | No | `None` | Filter by single resource type |
| `resource_types` | `Optional[List[str]]` | No | `None` | Filter by multiple resource types |
| `resource_id` | `Optional[str]` | No | `None` | Filter by specific resource |
| `start_date` | `Optional[datetime]` | No | `None` | Events after this timestamp |
| `end_date` | `Optional[datetime]` | No | `None` | Events before this timestamp |
| `success` | `Optional[bool]` | No | `None` | Filter by success/failure status |
| `limit` | `int` | No | `100` | Maximum results (1-1000) |
| `offset` | `int` | No | `0` | Pagination offset |

**Example**:

```python
from portico.ports.audit import AuditQuery, AuditAction
from datetime import datetime, timedelta

# Query failed login attempts in last 7 days
query = AuditQuery(
    action=AuditAction.LOGIN,
    success=False,
    start_date=datetime.now() - timedelta(days=7),
    limit=50
)

# Query all user actions on specific resource
query = AuditQuery(
    user_id=user_id,
    resource_type="document",
    resource_id="doc-123",
    limit=100
)

# Query group activity across multiple resource types
query = AuditQuery(
    group_id=group_id,
    resource_types=["document", "template", "file"],
    actions=["create", "update", "delete"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

**Immutability**: `AuditQuery` is frozen to ensure query parameters don't change during execution.

### AuditSummary

Summary statistics for audit events over a time period.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `total_events` | `int` | Yes | - | Total number of events in period |
| `events_by_action` | `Dict[str, int]` | Yes | - | Count of events grouped by action |
| `events_by_user` | `Dict[str, int]` | Yes | - | Count of events grouped by user |
| `events_by_resource_type` | `Dict[str, int]` | Yes | - | Count of events grouped by resource type |
| `events_by_group` | `Dict[str, int]` | Yes | - | Count of events grouped by group |
| `success_rate` | `float` | Yes | - | Percentage of successful events (0.0-1.0) |
| `time_range` | `tuple[datetime, datetime]` | Yes | - | Start and end timestamps of summary period |

**Example**:

```python
summary = await audit_adapter.generate_summary(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

print(f"Total events: {summary.total_events}")
print(f"Success rate: {summary.success_rate:.2%}")
print(f"Events by action: {summary.events_by_action}")
# {'create': 150, 'update': 300, 'delete': 50, 'login': 1000}
```

**Immutability**: `AuditSummary` is frozen - snapshots represent point-in-time statistics.

## Enumerations

### AuditAction

Standard audit actions for consistent logging across the application.

| Value | Description |
|-------|-------------|
| `CREATE` | Resource creation |
| `READ` | Resource access/retrieval |
| `UPDATE` | Resource modification |
| `DELETE` | Resource deletion |
| `LOGIN` | User login |
| `LOGOUT` | User logout |
| `EXPORT` | Data export |
| `IMPORT` | Data import |
| `APPROVE` | Approval action |
| `REJECT` | Rejection action |

**Example**:

```python
from portico.ports.audit import AuditAction

# Use enum values
event = AuditEvent(
    action=AuditAction.CREATE,
    resource_type="user",
    # ...
)

# Convert from string
action = AuditAction("update")  # AuditAction.UPDATE
```

## Port Interfaces

### AuditAdapter

The `AuditAdapter` abstract base class defines the contract for all audit logging backends.

#### Logging Operations

##### log_event

```python
async def log_event(event: AuditEvent) -> None
```

Logs an audit event to storage.

**Parameters**:

- `event`: The audit event to log

**Example**:

```python
event = AuditEvent(
    user_id=user_id,
    action=AuditAction.UPDATE,
    resource_type="profile",
    resource_id=str(user_id),
    details={"field": "email", "old": "old@example.com", "new": "new@example.com"}
)

await audit_adapter.log_event(event)
```

**Note**: This method should never raise exceptions to prevent audit failures from breaking application flow. Implementations should log errors internally.

#### Query Operations

##### search_events

```python
async def search_events(query: AuditQuery) -> List[AuditEvent]
```

Searches audit events by criteria with pagination support.

**Parameters**:

- `query`: Search criteria including filters and pagination

**Returns**: List of audit events matching the query criteria.

**Example**:

```python
from portico.ports.audit import AuditQuery, AuditAction

# Find all failed login attempts
query = AuditQuery(
    action=AuditAction.LOGIN,
    success=False,
    limit=100
)
events = await audit_adapter.search_events(query)

# Find user activity in date range
query = AuditQuery(
    user_id=user_id,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    limit=1000
)
events = await audit_adapter.search_events(query)
```

##### get_user_activity

```python
async def get_user_activity(user_id: UUID, days: int = 30) -> List[AuditEvent]
```

Retrieves recent activity for a specific user.

**Parameters**:

- `user_id`: User identifier
- `days`: Number of days to look back (default: 30)

**Returns**: List of audit events for the user within the specified time period.

**Example**:

```python
# Get user's last 7 days of activity
events = await audit_adapter.get_user_activity(
    user_id=user_id,
    days=7
)

for event in events:
    print(f"{event.timestamp}: {event.action.value} {event.resource_type}")
```

##### get_resource_history

```python
async def get_resource_history(resource_type: str, resource_id: str) -> List[AuditEvent]
```

Retrieves complete audit history for a specific resource.

**Parameters**:

- `resource_type`: Type of resource (e.g., "document", "user")
- `resource_id`: Resource identifier

**Returns**: List of all audit events for the specified resource.

**Example**:

```python
# Get all changes to a document
history = await audit_adapter.get_resource_history(
    resource_type="document",
    resource_id="doc-123"
)

# Display change timeline
for event in sorted(history, key=lambda e: e.timestamp):
    print(f"{event.timestamp}: {event.action.value} by user {event.user_id}")
    if event.details:
        print(f"  Details: {event.details}")
```

#### Reporting Operations

##### generate_summary

```python
async def generate_summary(start_date: datetime, end_date: datetime) -> AuditSummary
```

Generates audit summary statistics for a date range.

**Parameters**:

- `start_date`: Start of date range
- `end_date`: End of date range

**Returns**: `AuditSummary` containing statistics for the specified date range.

**Example**:

```python
from datetime import datetime, timedelta

# Monthly audit report
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)

summary = await audit_adapter.generate_summary(start, end)

print(f"Total Events: {summary.total_events}")
print(f"Success Rate: {summary.success_rate:.2%}")
print("\nEvents by Action:")
for action, count in summary.events_by_action.items():
    print(f"  {action}: {count}")

print("\nTop Users:")
for user_id, count in sorted(
    summary.events_by_user.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:
    print(f"  User {user_id}: {count} events")
```

#### Maintenance Operations

##### cleanup_old_events

```python
async def cleanup_old_events(older_than_days: int) -> int
```

Removes audit events older than the specified number of days.

**Parameters**:

- `older_than_days`: Remove events older than this many days

**Returns**: Number of audit events removed.

**Example**:

```python
# Remove events older than 90 days (compliance retention)
removed = await audit_adapter.cleanup_old_events(older_than_days=90)
print(f"Cleaned up {removed} old audit events")

# Scheduled cleanup task
async def cleanup_task():
    while True:
        await asyncio.sleep(86400)  # Daily
        removed = await audit_adapter.cleanup_old_events(90)
        logger.info("audit_cleanup", removed=removed)
```

## Usage Patterns

### Basic Event Logging

```python
from portico.ports.audit import AuditEvent, AuditAction, AuditAdapter

async def update_user_profile(
    user_id: UUID,
    updates: dict,
    audit: AuditAdapter
):
    # Update profile
    old_email = user.email
    user.email = updates["email"]
    await db.commit()

    # Log audit event
    await audit.log_event(AuditEvent(
        user_id=user_id,
        action=AuditAction.UPDATE,
        resource_type="user_profile",
        resource_id=str(user_id),
        details={
            "field": "email",
            "old_value": old_email,
            "new_value": updates["email"]
        },
        success=True
    ))
```

### Security Event Tracking

```python
async def track_login_attempt(
    username: str,
    success: bool,
    ip_address: str,
    user_agent: str,
    audit: AuditAdapter,
    error_message: Optional[str] = None
):
    event = AuditEvent(
        user_id=user.id if success else None,
        action=AuditAction.LOGIN,
        resource_type="authentication",
        resource_id=username,
        ip_address=ip_address,
        user_agent=user_agent,
        success=success,
        error_message=error_message,
        details={"username": username}
    )

    await audit.log_event(event)

    # Check for suspicious activity
    if not success:
        recent_failures = await audit.search_events(AuditQuery(
            action=AuditAction.LOGIN,
            success=False,
            resource_id=username,
            start_date=datetime.now() - timedelta(hours=1)
        ))

        if len(recent_failures) >= 5:
            logger.warning(
                "multiple_failed_logins",
                username=username,
                attempts=len(recent_failures),
                ip_address=ip_address
            )
```

### Multi-Tenant Audit Isolation

```python
async def log_group_action(
    user_id: UUID,
    group_id: UUID,
    action: AuditAction,
    resource_type: str,
    resource_id: str,
    audit: AuditAdapter
):
    """Log action with group scope for multi-tenant isolation."""
    event = AuditEvent(
        user_id=user_id,
        group_id=group_id,  # Tenant/group scope
        action=action,
        resource_type=resource_type,
        resource_id=resource_id
    )

    await audit.log_event(event)

# Query group-specific audit trail
async def get_group_audit_trail(group_id: UUID, audit: AuditAdapter):
    """Get all audit events for a specific group/tenant."""
    query = AuditQuery(
        group_id=group_id,
        limit=1000
    )
    return await audit.search_events(query)
```

### Resource Change Tracking

```python
async def track_document_changes(
    document_id: str,
    old_version: dict,
    new_version: dict,
    user_id: UUID,
    audit: AuditAdapter
):
    """Track what changed in a document."""
    changes = {}
    for key in new_version:
        if old_version.get(key) != new_version[key]:
            changes[key] = {
                "old": old_version.get(key),
                "new": new_version[key]
            }

    event = AuditEvent(
        user_id=user_id,
        action=AuditAction.UPDATE,
        resource_type="document",
        resource_id=document_id,
        details={"changes": changes}
    )

    await audit.log_event(event)

# Later: Get document history
async def get_document_timeline(document_id: str, audit: AuditAdapter):
    history = await audit.get_resource_history("document", document_id)

    timeline = []
    for event in sorted(history, key=lambda e: e.timestamp):
        timeline.append({
            "timestamp": event.timestamp,
            "user": event.user_id,
            "action": event.action.value,
            "changes": event.details.get("changes", {})
        })

    return timeline
```

### Compliance Reporting

```python
from datetime import datetime, timedelta

async def generate_compliance_report(
    audit: AuditAdapter,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """Generate compliance report for auditors."""

    # Get summary statistics
    summary = await audit.generate_summary(start_date, end_date)

    # Get all failed actions for review
    failures = await audit.search_events(AuditQuery(
        success=False,
        start_date=start_date,
        end_date=end_date,
        limit=1000
    ))

    # Get privileged actions (admin operations)
    privileged_actions = await audit.search_events(AuditQuery(
        actions=["delete", "export", "approve"],
        start_date=start_date,
        end_date=end_date,
        limit=1000
    ))

    return {
        "period": {
            "start": start_date,
            "end": end_date
        },
        "summary": {
            "total_events": summary.total_events,
            "success_rate": summary.success_rate,
            "events_by_action": summary.events_by_action,
            "unique_users": len(summary.events_by_user)
        },
        "failures": [
            {
                "timestamp": f.timestamp,
                "user": f.user_id,
                "action": f.action.value,
                "resource": f"{f.resource_type}:{f.resource_id}",
                "error": f.error_message
            }
            for f in failures
        ],
        "privileged_operations": [
            {
                "timestamp": p.timestamp,
                "user": p.user_id,
                "action": p.action.value,
                "resource": f"{p.resource_type}:{p.resource_id}"
            }
            for p in privileged_actions
        ]
    }
```

### Activity Monitoring

```python
async def monitor_user_activity(
    user_id: UUID,
    audit: AuditAdapter
) -> dict:
    """Get user activity summary."""

    # Last 30 days of activity
    events = await audit.get_user_activity(user_id, days=30)

    # Categorize actions
    action_counts = {}
    resource_counts = {}
    daily_activity = {}

    for event in events:
        # Count by action
        action = event.action.value
        action_counts[action] = action_counts.get(action, 0) + 1

        # Count by resource type
        resource_counts[event.resource_type] = \
            resource_counts.get(event.resource_type, 0) + 1

        # Count by day
        day = event.timestamp.date()
        daily_activity[day] = daily_activity.get(day, 0) + 1

    return {
        "user_id": str(user_id),
        "period_days": 30,
        "total_events": len(events),
        "actions": action_counts,
        "resources": resource_counts,
        "daily_activity": daily_activity,
        "last_activity": max(e.timestamp for e in events) if events else None
    }
```

### Failed Operation Analysis

```python
async def analyze_failures(
    audit: AuditAdapter,
    hours: int = 24
) -> dict:
    """Analyze recent failures for security monitoring."""

    start = datetime.now() - timedelta(hours=hours)

    failures = await audit.search_events(AuditQuery(
        success=False,
        start_date=start,
        limit=1000
    ))

    # Group by type
    by_action = {}
    by_user = {}
    by_resource = {}

    for failure in failures:
        action = failure.action.value
        by_action[action] = by_action.get(action, 0) + 1

        if failure.user_id:
            user = str(failure.user_id)
            by_user[user] = by_user.get(user, 0) + 1

        resource = failure.resource_type
        by_resource[resource] = by_resource.get(resource, 0) + 1

    # Identify potential issues
    alerts = []
    for user, count in by_user.items():
        if count >= 10:
            alerts.append({
                "type": "high_failure_rate",
                "user": user,
                "failures": count,
                "period_hours": hours
            })

    return {
        "total_failures": len(failures),
        "by_action": by_action,
        "by_user": by_user,
        "by_resource": by_resource,
        "alerts": alerts
    }
```

## Integration with Kits

The Audit Port is used by the **AuditKit** to provide high-level audit logging services to your application.

### Accessing AuditKit

```python
from portico import compose
from portico.kits.fastapi import Dependencies

# Configure audit in webapp
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.audit(
            enable_auditing=True,
            retention_days=90  # 90-day retention for compliance
        ),
    ],
)

deps = Dependencies(app)

# Access audit service
audit_service = app.kits["audit"].service
```

### Using AuditService

The `AuditService` provides a higher-level API built on top of `AuditAdapter`:

```python
from portico.ports.audit import AuditAction

# Log event via service
await audit_service.log_event(
    user_id=user_id,
    action=AuditAction.CREATE,
    resource_type="document",
    resource_id=doc_id,
    details={"title": "New Document"},
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent")
)

# Search events
from portico.ports.audit import AuditQuery

events = await audit_service.search_events(
    AuditQuery(user_id=user_id, limit=100)
)

# Get user activity
activity = await audit_service.get_user_activity(user_id, days=30)

# Generate summary
summary = await audit_service.generate_summary(start_date, end_date)
```

### FastAPI Integration

```python
from fastapi import Request
from portico.kits.fastapi import Dependencies
from portico.ports.audit import AuditAction

deps = Dependencies(app)

@app.post("/documents")
async def create_document(
    data: CreateDocumentRequest,
    request: Request,
    current_user: User = deps.current_user
):
    # Create document
    document = await document_service.create(data)

    # Log audit event
    audit_service = app.kits["audit"].service
    await audit_service.log_event(
        user_id=current_user.id,
        action=AuditAction.CREATE,
        resource_type="document",
        resource_id=str(document.id),
        details={"title": document.title, "type": document.type},
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )

    return document

@app.get("/audit/user/{user_id}")
async def get_user_audit_trail(
    user_id: UUID,
    days: int = 30,
    current_user: User = deps.current_user
):
    """Get audit trail for a user."""
    # Check permissions
    if current_user.id != user_id and not current_user.is_admin:
        raise AuthorizationError("Cannot view other user's audit trail")

    audit_service = app.kits["audit"].service
    events = await audit_service.get_user_activity(user_id, days)

    return {
        "user_id": str(user_id),
        "period_days": days,
        "total_events": len(events),
        "events": [
            {
                "timestamp": e.timestamp,
                "action": e.action.value,
                "resource": f"{e.resource_type}:{e.resource_id}",
                "success": e.success
            }
            for e in events
        ]
    }
```

## Best Practices

### 1. Always Log Security-Sensitive Operations

```python
# ✅ GOOD: Log authentication events
await audit.log_event(AuditEvent(
    user_id=user.id,
    action=AuditAction.LOGIN,
    resource_type="authentication",
    ip_address=ip_address,
    success=True
))

# ✅ GOOD: Log data exports
await audit.log_event(AuditEvent(
    user_id=user_id,
    action=AuditAction.EXPORT,
    resource_type="customer_data",
    details={"record_count": len(records)}
))
```

### 2. Include Contextual Information

```python
# ✅ GOOD: Rich context in details
await audit.log_event(AuditEvent(
    user_id=user_id,
    action=AuditAction.UPDATE,
    resource_type="user_profile",
    resource_id=str(user_id),
    details={
        "field_changed": "email",
        "old_value": old_email,
        "new_value": new_email,
        "verification_required": True
    },
    ip_address=ip_address,
    user_agent=user_agent
))

# ❌ BAD: Minimal context
await audit.log_event(AuditEvent(
    action=AuditAction.UPDATE,
    resource_type="user"
))
```

### 3. Use Group Scoping for Multi-Tenancy

```python
# ✅ GOOD: Group-scoped events
await audit.log_event(AuditEvent(
    user_id=user_id,
    group_id=group_id,  # Tenant isolation
    action=AuditAction.CREATE,
    resource_type="document",
    resource_id=doc_id
))

# Query only accessible events
events = await audit.search_events(AuditQuery(
    group_ids=user_accessible_groups  # User can only see their groups
))
```

### 4. Log Both Success and Failure

```python
# ✅ GOOD: Log failures with error details
try:
    await sensitive_operation()
    await audit.log_event(AuditEvent(
        action=AuditAction.UPDATE,
        resource_type="config",
        success=True
    ))
except Exception as e:
    await audit.log_event(AuditEvent(
        action=AuditAction.UPDATE,
        resource_type="config",
        success=False,
        error_message=str(e)
    ))
    raise
```

### 5. Implement Retention Policies

```python
# ✅ GOOD: Regular cleanup based on compliance requirements
async def scheduled_audit_cleanup():
    """Run daily to enforce retention policy."""
    retention_days = 90  # SOC 2 / HIPAA compliance

    removed = await audit.cleanup_old_events(retention_days)

    logger.info(
        "audit_retention_cleanup",
        removed=removed,
        retention_days=retention_days
    )

# Schedule cleanup
asyncio.create_task(periodic_cleanup(interval=86400))
```

### 6. Monitor Audit System Health

```python
# ✅ GOOD: Monitor audit system
async def check_audit_health(audit: AuditAdapter):
    """Verify audit system is working."""

    # Check recent events are being logged
    recent = await audit.search_events(AuditQuery(
        start_date=datetime.now() - timedelta(hours=1),
        limit=10
    ))

    if not recent:
        logger.error("audit_system_not_logging", alert=True)

    # Check storage growth
    summary = await audit.generate_summary(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )

    events_per_day = summary.total_events / 30
    if events_per_day > 100000:
        logger.warning(
            "high_audit_volume",
            events_per_day=events_per_day
        )
```

### 7. Don't Log Sensitive Data

```python
# ✅ GOOD: Hash or redact sensitive data
await audit.log_event(AuditEvent(
    action=AuditAction.UPDATE,
    resource_type="user",
    details={
        "field": "password",
        "changed": True  # Don't log the actual password!
    }
))

# ❌ BAD: Logging sensitive information
await audit.log_event(AuditEvent(
    details={
        "password": "user_password_123",  # Never do this!
        "ssn": "123-45-6789"  # Never do this!
    }
))
```

### 8. Use Transactional Logging When Available

```python
# ✅ GOOD: Audit event in same transaction as data change
async def update_with_audit(
    user_id: UUID,
    updates: dict,
    db_session: AsyncSession,
    audit_service: AuditService
):
    # Update data
    user.email = updates["email"]
    db_session.add(user)

    # Log audit event in same transaction
    await audit_service.log_event(
        user_id=user_id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(user_id),
        details=updates,
        db_session=db_session  # Same transaction
    )

    # Both committed together
    await db_session.commit()
```

### 9. Provide Audit Trail Access

```python
# ✅ GOOD: Users can view their own audit trail
@app.get("/my/audit-trail")
async def my_audit_trail(
    current_user: User = deps.current_user,
    days: int = 30
):
    events = await audit_service.get_user_activity(
        current_user.id,
        days=days
    )

    return {
        "events": events,
        "total": len(events),
        "period_days": days
    }
```

### 10. Document Audit Event Schema

```python
# ✅ GOOD: Document what gets audited
"""
Audit Event Schema for Document Operations:

CREATE document:
  resource_type: "document"
  resource_id: document UUID
  details: {
    "title": str,
    "type": str,
    "template_id": str (optional)
  }

UPDATE document:
  details: {
    "changes": {
      "field_name": {"old": value, "new": value}
    }
  }

DELETE document:
  details: {
    "title": str,
    "permanent": bool
  }
"""
```

## FAQs

### Should I log every database operation?

No. Focus on **security-sensitive** and **compliance-relevant** operations:

- ✅ Log: Authentication, authorization, data exports, admin operations, configuration changes
- ❌ Don't log: Read operations, UI interactions, routine data access

```python
# ✅ Log this
await audit.log_event(AuditEvent(action=AuditAction.EXPORT, ...))

# ❌ Don't log this
user_profile = await db.get_user_profile(user_id)  # Routine read
```

### How do I handle audit failures?

Audit logging should **never break your application**. Implementations should catch and log errors internally:

```python
# Adapter implementation handles errors
async def log_event(self, event: AuditEvent) -> None:
    try:
        await self._store_event(event)
    except Exception as e:
        # Log error but don't raise
        logger.error("audit_log_failed", error=str(e))
        # Application continues normally
```

### What's the difference between audit logging and application logging?

- **Audit Logging** (AuditPort): Compliance, security, "who did what when"
  - Structured events with metadata
  - Long retention (90+ days)
  - Queryable and reportable
  - Immutable records

- **Application Logging** (LoggingKit): Debugging, monitoring, "what happened"
  - Diagnostic messages
  - Shorter retention (7-30 days)
  - Text-based search
  - Can be modified/rotated

```python
# Audit log (structured, compliance)
await audit.log_event(AuditEvent(
    user_id=user_id,
    action=AuditAction.DELETE,
    resource_type="user",
    resource_id=str(deleted_user_id)
))

# Application log (diagnostic)
logger.info(
    "user_deleted",
    user_id=str(deleted_user_id),
    deleted_by=str(user_id)
)
```

### How do I query audit events efficiently?

Use appropriate indexes and limit result sets:

```python
# ✅ GOOD: Specific query with limit
events = await audit.search_events(AuditQuery(
    user_id=user_id,
    start_date=datetime.now() - timedelta(days=7),
    limit=100
))

# ❌ BAD: Unbounded query
events = await audit.search_events(AuditQuery(
    limit=10000  # Too many results
))
```

Adapters should index: `user_id`, `group_id`, `resource_type`, `resource_id`, `timestamp`, `action`.

### Can I audit events from background jobs?

Yes! Set `user_id=None` for system-initiated actions:

```python
# Background job audit event
await audit.log_event(AuditEvent(
    user_id=None,  # System action
    action=AuditAction.DELETE,
    resource_type="expired_session",
    resource_id=session_id,
    details={"reason": "expired", "age_days": 30}
))
```

### How do I implement audit log export for compliance?

```python
async def export_audit_logs(
    start_date: datetime,
    end_date: datetime,
    audit: AuditAdapter
) -> str:
    """Export audit logs to CSV for compliance."""

    events = await audit.search_events(AuditQuery(
        start_date=start_date,
        end_date=end_date,
        limit=10000
    ))

    # Convert to CSV
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Timestamp", "User ID", "Action", "Resource Type",
        "Resource ID", "Success", "IP Address", "Details"
    ])

    # Data
    for event in events:
        writer.writerow([
            event.timestamp.isoformat(),
            str(event.user_id) if event.user_id else "SYSTEM",
            event.action.value,
            event.resource_type,
            event.resource_id or "",
            "SUCCESS" if event.success else "FAILURE",
            event.ip_address or "",
            json.dumps(event.details)
        ])

    return output.getvalue()
```

### Should I use CompositeAudit for multiple destinations?

Yes, for critical applications. Log to both database (queryable) and structured logs (long-term storage):

```python
from portico import compose

app = compose.webapp(
    kits=[
        compose.audit(
            # Composite adapter logs to multiple destinations
            backends=["database", "structured_logging"]
        )
    ]
)
```

### How do I handle group-based access to audit logs?

Filter by user's accessible groups:

```python
async def get_accessible_audit_events(
    user: User,
    user_groups: List[UUID],
    audit: AuditAdapter
) -> List[AuditEvent]:
    """Get audit events user is allowed to see."""

    if user.is_admin:
        # Admins see everything
        query = AuditQuery(limit=1000)
    else:
        # Users see only their groups
        query = AuditQuery(
            group_ids=user_groups,
            limit=1000
        )

    return await audit.search_events(query)
```

### What retention period should I use?

Depends on compliance requirements:

- **SOC 2**: 90 days minimum
- **HIPAA**: 6 years for healthcare data
- **GDPR**: As long as necessary for the purpose
- **PCI DSS**: 1 year minimum (3 months online, 9 months archived)

```python
# Configure based on compliance needs
app = compose.webapp(
    kits=[
        compose.audit(retention_days=90)  # SOC 2
        # compose.audit(retention_days=2190)  # HIPAA (6 years)
    ]
)
```

### How do I test audit logging?

Use `MemoryAudit` adapter in tests:

```python
import pytest
from portico.adapters.audit import MemoryAudit
from portico.ports.audit import AuditEvent, AuditAction

@pytest.fixture
def audit_adapter():
    return MemoryAudit()

async def test_audit_logging(audit_adapter):
    # Log event
    event = AuditEvent(
        user_id=user_id,
        action=AuditAction.CREATE,
        resource_type="document"
    )
    await audit_adapter.log_event(event)

    # Verify logged
    events = await audit_adapter.search_events(AuditQuery(
        user_id=user_id
    ))
    assert len(events) == 1
    assert events[0].action == AuditAction.CREATE
```

### Can I customize audit actions beyond the standard set?

The `AuditAction` enum provides standard actions, but you can pass custom strings:

```python
# Standard action
event = AuditEvent(action=AuditAction.CREATE, ...)

# Custom action (stored as string)
event = AuditEvent(
    action="custom_workflow_action",
    resource_type="workflow",
    details={"step": "approval"}
)
```

However, using standard `AuditAction` values is recommended for consistency.
