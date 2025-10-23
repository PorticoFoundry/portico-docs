# Audit Port

## Overview

The Audit Port defines the contract for audit logging operations in Portico applications.

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

**Port Type**: Adapter

**When to Use**:

- Applications requiring compliance with audit regulations (SOC 2, HIPAA, GDPR)
- Systems needing security event tracking and forensics
- Multi-tenant applications requiring isolated audit trails
- Applications tracking user activity and resource changes
- Systems requiring detailed activity reporting and analytics

## Domain Models

### AuditEvent

Represents an audit event with comprehensive metadata for compliance and security tracking. Immutable record.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | Auto-generated | Unique event identifier |
| `user_id` | `Optional[UUID]` | No | `None` | User who performed the action |
| `group_id` | `Optional[UUID]` | No | `None` | Group scope for multi-tenant isolation |
| `action` | `AuditAction` | Yes | - | Action performed (CREATE, UPDATE, DELETE, etc.) |
| `resource_type` | `str` | Yes | - | Type of resource affected |
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
    user_id=user_id,
    group_id=group_id,
    action=AuditAction.UPDATE,
    resource_type="document",
    resource_id="doc-456",
    details={
        "field_changed": "title",
        "old_value": "Draft",
        "new_value": "Final"
    },
    ip_address="192.168.1.100",
    success=True
)
```

### AuditQuery

Query parameters for searching and filtering audit events.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | `Optional[UUID]` | No | `None` | Filter by single user |
| `user_ids` | `Optional[List[UUID]]` | No | `None` | Filter by multiple users |
| `group_id` | `Optional[UUID]` | No | `None` | Filter by single group |
| `group_ids` | `Optional[List[UUID]]` | No | `None` | Filter by multiple groups |
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

# Query group activity
query = AuditQuery(
    group_id=group_id,
    resource_types=["document", "template"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

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
```

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

event = AuditEvent(
    action=AuditAction.CREATE,
    resource_type="user",
    # ...
)
```

## Port Interfaces

### AuditAdapter

The `AuditAdapter` abstract base class defines the contract for all audit logging backends.

**Location**: `portico.ports.audit.AuditAdapter`

#### Key Methods

##### log_event

```python
async def log_event(event: AuditEvent) -> None
```

Logs an audit event to storage. Primary method for recording audit trails.

**Parameters**:

- `event`: The audit event to log

**Example**:

```python
event = AuditEvent(
    user_id=user_id,
    action=AuditAction.UPDATE,
    resource_type="profile",
    resource_id=str(user_id),
    details={
        "field": "email",
        "old": "old@example.com",
        "new": "new@example.com"
    }
)

await audit_adapter.log_event(event)
```

**Note**: This method should never raise exceptions to prevent audit failures from breaking application flow.

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

#### Other Methods

##### get_user_activity

```python
async def get_user_activity(user_id: UUID, days: int = 30) -> List[AuditEvent]
```

Retrieves recent activity for a specific user within the specified time period.

##### get_resource_history

```python
async def get_resource_history(resource_type: str, resource_id: str) -> List[AuditEvent]
```

Retrieves complete audit history for a specific resource.

##### generate_summary

```python
async def generate_summary(start_date: datetime, end_date: datetime) -> AuditSummary
```

Generates audit summary statistics for a date range.

##### cleanup_old_events

```python
async def cleanup_old_events(older_than_days: int) -> int
```

Removes audit events older than the specified number of days. Returns count removed.

## Common Patterns

### Security Event Tracking

```python
from portico.ports.audit import AuditEvent, AuditAction, AuditAdapter

async def track_login_attempt(
    username: str,
    success: bool,
    ip_address: str,
    user_agent: str,
    audit: AuditAdapter,
    user_id: Optional[UUID] = None,
    error_message: Optional[str] = None
):
    event = AuditEvent(
        user_id=user_id if success else None,
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
                attempts=len(recent_failures)
            )
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

    # Get privileged actions
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

## Integration with Kits

The Audit Port is used by the **AuditKit** to provide high-level audit logging services.

```python
from portico import compose
from portico.ports.audit import AuditAction

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

# Access audit service
audit_service = app.kits["audit"].service

# Log event
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

# Generate summary
summary = await audit_service.generate_summary(start_date, end_date)
```

The Audit Kit provides:

- Database-backed audit storage with SqlAlchemyAuditAdapter
- Transactional audit logging for data consistency
- Event publishing for audit event notifications
- Automatic retention policy enforcement

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Log Security-Sensitive Operations**: Always log authentication, authorization, data exports, and admin operations

   ```python
   # ✅ GOOD
   await audit.log_event(AuditEvent(
       action=AuditAction.LOGIN,
       resource_type="authentication",
       ip_address=ip_address,
       success=True
   ))

   # ✅ GOOD
   await audit.log_event(AuditEvent(
       action=AuditAction.EXPORT,
       resource_type="customer_data",
       details={"record_count": len(records)}
   ))
   ```

2. **Include Contextual Information**: Add rich metadata in details field for forensics

   ```python
   # ✅ GOOD - Rich context
   await audit.log_event(AuditEvent(
       action=AuditAction.UPDATE,
       resource_type="user_profile",
       details={
           "field_changed": "email",
           "old_value": old_email,
           "new_value": new_email,
           "verification_required": True
       },
       ip_address=ip_address,
       user_agent=user_agent
   ))

   # ❌ BAD - Minimal context
   await audit.log_event(AuditEvent(
       action=AuditAction.UPDATE,
       resource_type="user"
   ))
   ```

3. **Use Group Scoping for Multi-Tenancy**: Include group_id for tenant isolation

   ```python
   # ✅ GOOD
   await audit.log_event(AuditEvent(
       user_id=user_id,
       group_id=group_id,  # Tenant isolation
       action=AuditAction.CREATE,
       resource_type="document"
   ))
   ```

4. **Implement Retention Policies**: Regular cleanup based on compliance requirements

   ```python
   # ✅ GOOD - Scheduled cleanup
   async def scheduled_audit_cleanup():
       retention_days = 90  # SOC 2 compliance
       removed = await audit.cleanup_old_events(retention_days)
       logger.info("audit_cleanup", removed=removed)

   # Schedule daily
   asyncio.create_task(periodic_cleanup(interval=86400))
   ```

5. **Never Log Sensitive Data**: Hash or redact passwords, SSNs, credit cards

   ```python
   # ✅ GOOD
   await audit.log_event(AuditEvent(
       action=AuditAction.UPDATE,
       resource_type="user",
       details={"field": "password", "changed": True}  # Don't log actual password
   ))

   # ❌ BAD
   await audit.log_event(AuditEvent(
       details={"password": "user_password_123"}  # Never do this!
   ))
   ```

## FAQs

### Should I log every database operation?

No. Focus on **security-sensitive** and **compliance-relevant** operations:

- ✅ Log: Authentication, authorization, data exports, admin operations, configuration changes
- ❌ Don't log: Routine read operations, UI interactions, regular data access

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
        logger.error("audit_log_failed", error=str(e))
        # Application continues normally
```

### What's the difference between audit logging and application logging?

- **Audit Logging** (AuditPort): Compliance, security, "who did what when"
  - Structured events with metadata
  - Long retention (90+ days)
  - Queryable and reportable
  - Immutable records

- **Application Logging**: Debugging, monitoring, "what happened"
  - Diagnostic messages
  - Shorter retention (7-30 days)
  - Text-based search
  - Can be modified/rotated

### What retention period should I use?

Depends on compliance requirements:

- **SOC 2**: 90 days minimum
- **HIPAA**: 6 years for healthcare data
- **GDPR**: As long as necessary for the purpose
- **PCI DSS**: 1 year minimum

```python
app = compose.webapp(
    kits=[
        compose.audit(retention_days=90)  # SOC 2
    ]
)
```

### How do I test audit logging?

Use the `MemoryAudit` adapter in tests:

```python
import pytest
from portico.adapters.audit import MemoryAudit
from portico.ports.audit import AuditEvent, AuditAction

@pytest.fixture
async def audit_adapter():
    return MemoryAudit()

async def test_audit_logging(audit_adapter):
    event = AuditEvent(
        user_id=user_id,
        action=AuditAction.CREATE,
        resource_type="document"
    )
    await audit_adapter.log_event(event)

    # Verify logged
    events = await audit_adapter.search_events(
        AuditQuery(user_id=user_id)
    )
    assert len(events) == 1
    assert events[0].action == AuditAction.CREATE
```
