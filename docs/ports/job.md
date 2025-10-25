# Job Domain Models

## Overview

The Job domain models define the core data structures for background job processing in Portico. These immutable domain models are shared across all job-related ports (JobCreator, JobHandler, JobQueue, JobTrigger) and provide a consistent representation of jobs throughout their lifecycle.

**Purpose**: Provide type-safe, immutable domain models for representing background jobs, execution results, job statuses, and queue statistics.

**Domain**: Background job processing, asynchronous task execution, and job lifecycle management

**Key Capabilities**:

- Immutable job representation with full lifecycle tracking
- Type-safe job status enumeration
- Execution result modeling with success/failure tracking
- Queue statistics for monitoring and observability
- Support for job prioritization, scheduling, and retry logic

**Model Type**: Domain Models (not an interface)

**When to Use**:

- When creating, storing, or processing background jobs
- When tracking job execution status and results
- When implementing job queues, handlers, or triggers
- When monitoring job processing metrics

## Domain Models

### Job

Represents a background job with complete lifecycle information.

**Location**: `portico.ports.job.Job`

**Immutability**: `frozen=True` - Job instances are immutable

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | - | Unique identifier for the job |
| `queue_name` | `str` | Yes | - | Name of the queue this job belongs to (e.g., "default", "high_priority") |
| `job_type` | `str` | Yes | - | Type of job for handler routing (e.g., "email.send", "report.generate") |
| `payload` | `dict[str, Any]` | Yes | - | Job-specific data passed to the handler (must be JSON-serializable) |
| `priority` | `int` | No | `0` | Job priority (higher values = more urgent, processed first) |
| `max_retries` | `int` | No | `3` | Maximum number of retry attempts before moving to dead letter |
| `retry_count` | `int` | No | `0` | Current number of retry attempts |
| `scheduled_at` | `datetime \| None` | No | `None` | When the job should be executed (for delayed/scheduled jobs) |
| `started_at` | `datetime \| None` | No | `None` | When job execution started |
| `completed_at` | `datetime \| None` | No | `None` | When job completed successfully |
| `failed_at` | `datetime \| None` | No | `None` | When job failed permanently (exhausted retries or non-retriable error) |
| `status` | `JobStatus` | No | `JobStatus.PENDING` | Current execution status |
| `error_message` | `str \| None` | No | `None` | Error message from last failure |
| `created_by` | `UUID \| None` | No | `None` | User ID who created the job (for audit trail) |
| `created_at` | `datetime \| None` | No | `None` | When the job was created |
| `metadata` | `dict[str, Any]` | No | `{}` | Additional metadata for extensibility |

**Example**:

```python
from uuid import uuid4
from datetime import UTC, datetime, timedelta
from portico.ports.job import Job, JobStatus

# Basic job
job = Job(
    id=uuid4(),
    queue_name="emails",
    job_type="email.send",
    payload={
        "recipient": "user@example.com",
        "subject": "Welcome",
        "body": "Welcome to our service!"
    }
)

# High-priority job with retries
urgent_job = Job(
    id=uuid4(),
    queue_name="critical",
    job_type="payment.process",
    payload={"payment_id": "pay_123", "amount": 1000},
    priority=10,  # High priority
    max_retries=5  # More retries for critical jobs
)

# Scheduled job
scheduled_job = Job(
    id=uuid4(),
    queue_name="reminders",
    job_type="reminder.send",
    payload={"user_id": "user_123", "message": "Meeting in 1 hour"},
    scheduled_at=datetime.now(UTC) + timedelta(hours=1)  # Execute in 1 hour
)

# Job with user tracking and metadata
tracked_job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={"report_type": "monthly", "month": "2025-01"},
    created_by=uuid4(),  # Track who created the job
    metadata={"department": "sales", "cost_center": "CC-123"}
)
```

### JobStatus

Enumeration of job execution statuses representing the job lifecycle.

**Location**: `portico.ports.job.JobStatus`

**Type**: `str, Enum` - String-based enumeration

| Value | Description |
|-------|-------------|
| `PENDING` | Job is queued and waiting to be processed |
| `SCHEDULED` | Job is scheduled for future execution (has `scheduled_at` set) |
| `RUNNING` | Job is currently being processed by a worker |
| `COMPLETED` | Job finished successfully |
| `FAILED` | Job failed and will not be retried (non-retriable error or retry count < max_retries when requeue=False) |
| `RETRYING` | Job failed but will be retried (retry_count < max_retries) |
| `DEAD_LETTER` | Job exhausted all retries and moved to dead letter queue for manual review |

**Lifecycle Flow**:

```
PENDING → RUNNING → COMPLETED  ✓ Success
    ↓         ↓
SCHEDULED     RETRYING → RUNNING → COMPLETED  ✓ Retry succeeded
    ↓         ↓
    ↓         RETRYING → RUNNING → DEAD_LETTER  ✗ All retries exhausted
    ↓         ↓
    ↓         FAILED  ✗ Non-retriable error
    ↓
    → (wait for scheduled_at) → PENDING
```

**Example**:

```python
from portico.ports.job import JobStatus

# Check job status
if job.status == JobStatus.PENDING:
    print("Job is waiting to be processed")
elif job.status == JobStatus.RUNNING:
    print("Job is currently executing")
elif job.status == JobStatus.COMPLETED:
    print("Job completed successfully")
elif job.status == JobStatus.RETRYING:
    print(f"Job failed, retrying ({job.retry_count}/{job.max_retries})")
elif job.status == JobStatus.DEAD_LETTER:
    print(f"Job failed permanently: {job.error_message}")

# Status transitions
assert job.status == JobStatus.PENDING  # Initially pending

# After dequeue
job = Job(**{**job.__dict__, "status": JobStatus.RUNNING, "started_at": datetime.now(UTC)})

# After successful completion
job = Job(**{**job.__dict__, "status": JobStatus.COMPLETED, "completed_at": datetime.now(UTC)})
```

### JobResult

Represents the result of job execution returned by job handlers.

**Location**: `portico.ports.job.JobResult`

**Immutability**: `frozen=True` - JobResult instances are immutable

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `success` | `bool` | Yes | - | Whether the job executed successfully |
| `error` | `Exception \| None` | No | `None` | Exception that caused failure (if `success=False`) |
| `result_data` | `dict[str, Any]` | No | `{}` | Result data from successful execution or additional failure context |

**Example**:

```python
from portico.ports.job import JobResult

# Successful execution
result = JobResult(
    success=True,
    result_data={
        "email_sent": True,
        "message_id": "msg_abc123",
        "sent_at": "2025-01-15T10:30:00Z"
    }
)

# Failed execution with error
result = JobResult(
    success=False,
    error=ConnectionError("SMTP server unavailable"),
    result_data={"retry_recommended": True}
)

# Partial success
result = JobResult(
    success=True,
    result_data={
        "records_processed": 95,
        "records_skipped": 5,
        "warnings": ["Invalid data in row 23", "Missing field in row 67"]
    }
)

# Usage in handler
async def handle(self, job: Job) -> JobResult:
    try:
        await self.send_email(job.payload["recipient"])
        return JobResult(success=True)
    except Exception as e:
        return JobResult(success=False, error=e)
```

### QueueStats

Statistics for monitoring queue health and performance.

**Location**: `portico.ports.job.QueueStats`

**Immutability**: `frozen=True` - QueueStats instances are immutable

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `queue_name` | `str` | Yes | - | Name of the queue |
| `pending_count` | `int` | Yes | - | Number of jobs waiting to be processed (PENDING status) |
| `running_count` | `int` | Yes | - | Number of jobs currently being processed (RUNNING status) |
| `completed_count` | `int` | Yes | - | Number of successfully completed jobs |
| `failed_count` | `int` | Yes | - | Number of failed jobs (FAILED status, not retrying) |
| `dead_letter_count` | `int` | Yes | - | Number of jobs in dead letter queue (exhausted retries) |

**Example**:

```python
from portico.ports.job import QueueStats

stats = QueueStats(
    queue_name="emails",
    pending_count=42,
    running_count=5,
    completed_count=1250,
    failed_count=3,
    dead_letter_count=1
)

# Monitor queue health
if stats.pending_count > 100:
    print(f"Warning: Queue {stats.queue_name} has backlog of {stats.pending_count} jobs")

if stats.dead_letter_count > 0:
    print(f"Alert: {stats.dead_letter_count} jobs in dead letter queue need review")

# Calculate failure rate
total_processed = stats.completed_count + stats.failed_count + stats.dead_letter_count
if total_processed > 0:
    failure_rate = (stats.failed_count + stats.dead_letter_count) / total_processed
    print(f"Failure rate: {failure_rate:.2%}")
```

## Common Patterns

### 1. Creating Jobs with Different Priorities

```python
from uuid import uuid4
from portico.ports.job import Job

# Critical payment processing (high priority)
payment_job = Job(
    id=uuid4(),
    queue_name="payments",
    job_type="payment.process",
    payload={"payment_id": "pay_123"},
    priority=10  # Highest priority
)

# Standard email notification (medium priority)
email_job = Job(
    id=uuid4(),
    queue_name="notifications",
    job_type="email.send",
    payload={"recipient": "user@example.com"},
    priority=5  # Medium priority
)

# Background cleanup (low priority)
cleanup_job = Job(
    id=uuid4(),
    queue_name="maintenance",
    job_type="cleanup.temp_files",
    payload={"max_age_days": 30},
    priority=1  # Low priority
)

# Jobs will be processed in priority order: payment (10) → email (5) → cleanup (1)
```

### 2. Scheduled Jobs for Delayed Execution

```python
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from portico.ports.job import Job, JobStatus

# Send reminder 24 hours before event
event_time = datetime(2025, 10, 25, 14, 0, tzinfo=UTC)
reminder_time = event_time - timedelta(hours=24)

reminder_job = Job(
    id=uuid4(),
    queue_name="reminders",
    job_type="reminder.send",
    payload={
        "user_id": "user_123",
        "event": "Team Meeting",
        "event_time": event_time.isoformat()
    },
    scheduled_at=reminder_time  # Execute 24 hours before event
)

# Job starts as SCHEDULED status
assert reminder_job.status == JobStatus.SCHEDULED

# Queue adapter won't dequeue until reminder_time is reached
```

### 3. Job Retry Configuration

```python
from uuid import uuid4
from portico.ports.job import Job

# Critical job with more retries
critical_job = Job(
    id=uuid4(),
    queue_name="critical",
    job_type="payment.process",
    payload={"payment_id": "pay_123"},
    max_retries=5  # Allow 5 retry attempts
)

# Quick-fail job with no retries (idempotency required)
idempotent_job = Job(
    id=uuid4(),
    queue_name="webhooks",
    job_type="webhook.send",
    payload={"url": "https://api.example.com/webhook"},
    max_retries=0  # Don't retry, fail immediately
)

# Standard job with default retries
standard_job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="report.generate",
    payload={"report_type": "daily"},
    max_retries=3  # Default: 3 retries
)
```

### 4. Tracking Job Lifecycle with Timestamps

```python
from datetime import UTC, datetime
from uuid import uuid4
from portico.ports.job import Job, JobStatus

# Create job
job = Job(
    id=uuid4(),
    queue_name="processing",
    job_type="data.import",
    payload={"file_path": "/uploads/data.csv"},
    created_at=datetime.now(UTC)
)

print(f"Job created at: {job.created_at}")

# Job dequeued and started
job = Job(**{**job.__dict__,
    "status": JobStatus.RUNNING,
    "started_at": datetime.now(UTC)
})

print(f"Job started at: {job.started_at}")
print(f"Wait time: {(job.started_at - job.created_at).seconds}s")

# Job completed
job = Job(**{**job.__dict__,
    "status": JobStatus.COMPLETED,
    "completed_at": datetime.now(UTC)
})

print(f"Job completed at: {job.completed_at}")
print(f"Execution time: {(job.completed_at - job.started_at).seconds}s")
print(f"Total time: {(job.completed_at - job.created_at).seconds}s")
```

### 5. Job Metadata for Extensibility

```python
from uuid import uuid4
from portico.ports.job import Job

# Job with business context metadata
job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={"report_type": "sales", "month": "2025-01"},
    metadata={
        "department": "sales",
        "cost_center": "CC-SALES-01",
        "requested_by": "manager@example.com",
        "urgency": "high",
        "tags": ["monthly", "executive"]
    }
)

# Metadata can be used for:
# - Filtering/searching jobs
# - Routing to specific workers
# - Cost allocation
# - Audit trails
# - Custom business logic
```

### 6. Handling Job Results

```python
from portico.ports.job import Job, JobResult

async def process_job(job: Job) -> JobResult:
    """Process a job and return detailed result."""
    try:
        if job.job_type == "email.send":
            message_id = await send_email(
                to=job.payload["recipient"],
                subject=job.payload["subject"],
                body=job.payload["body"]
            )

            return JobResult(
                success=True,
                result_data={
                    "message_id": message_id,
                    "sent_at": datetime.now(UTC).isoformat(),
                    "recipient": job.payload["recipient"]
                }
            )

        elif job.job_type == "data.import":
            records = await import_data(job.payload["file_path"])

            return JobResult(
                success=True,
                result_data={
                    "records_imported": len(records),
                    "file_path": job.payload["file_path"],
                    "duration_ms": 1234
                }
            )

    except ValidationError as e:
        # Don't retry validation errors
        return JobResult(
            success=False,
            error=e,
            result_data={"error_type": "validation", "retry_recommended": False}
        )

    except ConnectionError as e:
        # Retry connection errors
        return JobResult(
            success=False,
            error=e,
            result_data={"error_type": "connection", "retry_recommended": True}
        )
```

### 7. Monitoring Queue Health

```python
from portico.ports.job import QueueStats

async def monitor_queue_health(stats: QueueStats) -> None:
    """Monitor queue and send alerts based on metrics."""

    # Alert on large backlog
    if stats.pending_count > 1000:
        await send_alert(
            level="warning",
            message=f"Queue {stats.queue_name} has {stats.pending_count} pending jobs",
            recommendation="Consider scaling up workers"
        )

    # Alert on dead letter jobs
    if stats.dead_letter_count > 0:
        await send_alert(
            level="error",
            message=f"{stats.dead_letter_count} jobs in dead letter queue",
            recommendation="Review and fix failed jobs manually"
        )

    # Calculate and alert on failure rate
    total_completed = stats.completed_count + stats.failed_count + stats.dead_letter_count
    if total_completed > 100:
        failure_rate = (stats.failed_count + stats.dead_letter_count) / total_completed

        if failure_rate > 0.05:  # 5% failure rate
            await send_alert(
                level="warning",
                message=f"Queue {stats.queue_name} has {failure_rate:.1%} failure rate",
                recommendation="Investigate error patterns"
            )

    # Alert on stalled processing
    if stats.running_count == 0 and stats.pending_count > 0:
        await send_alert(
            level="critical",
            message=f"Queue {stats.queue_name} has pending jobs but no active workers",
            recommendation="Check worker health"
        )
```

## Best Practices

### 1. Keep Payloads JSON-Serializable

Job payloads must be serializable for storage in queues.

```python
from datetime import datetime
from uuid import UUID, uuid4
import json

# ✅ GOOD - JSON-serializable payload
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="email.send",
    payload={
        "recipient": "user@example.com",
        "subject": "Hello",
        "user_id": str(uuid4()),  # Convert UUID to string
        "timestamp": datetime.now().isoformat()  # Convert datetime to string
    }
)

# Verify serializable
assert json.dumps(job.payload)  # No error

# ❌ BAD - Non-serializable payload
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="email.send",
    payload={
        "recipient": "user@example.com",
        "user_id": uuid4(),  # ❌ UUID not JSON-serializable
        "timestamp": datetime.now()  # ❌ datetime not JSON-serializable
    }
)

# Will fail when queue adapter tries to serialize
json.dumps(job.payload)  # ❌ TypeError
```

### 2. Use Descriptive Job Types

Use hierarchical, descriptive job type names for clarity and routing.

```python
# ✅ GOOD - Descriptive, hierarchical names
email_job = Job(
    id=uuid4(),
    queue_name="notifications",
    job_type="email.send.transactional",  # Clear hierarchy
    payload={"recipient": "user@example.com"}
)

report_job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate.monthly.sales",  # Very specific
    payload={"month": "2025-01"}
)

# ❌ BAD - Vague names
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="process",  # ❌ Too vague
    payload={"data": "something"}
)

job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="job1",  # ❌ Meaningless
    payload={}
)
```

### 3. Set Appropriate Max Retries

Configure retries based on job characteristics.

```python
# ✅ GOOD - Retries match job characteristics

# Critical jobs: more retries
payment_job = Job(
    id=uuid4(),
    queue_name="payments",
    job_type="payment.charge",
    payload={"amount": 1000},
    max_retries=5  # Important, allow more retries
)

# Idempotent jobs: fewer retries
webhook_job = Job(
    id=uuid4(),
    queue_name="webhooks",
    job_type="webhook.send",
    payload={"url": "https://api.example.com"},
    max_retries=1  # Quick fail, external system may be down
)

# Non-idempotent jobs: no retries
unique_job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="user.create",
    payload={"email": "user@example.com"},
    max_retries=0  # Don't retry to avoid duplicates
)

# ❌ BAD - Same retries for everything
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="critical.payment",
    max_retries=3  # ❌ Too few for critical job
)
```

### 4. Use Priority for Urgency, Not Importance

Priority should reflect how quickly a job needs to run, not its business importance.

```python
# ✅ GOOD - Priority based on urgency

# Urgent: Real-time user action (needs immediate response)
verification_job = Job(
    id=uuid4(),
    queue_name="realtime",
    job_type="sms.verification",
    payload={"phone": "+1234567890"},
    priority=10  # High priority - user is waiting
)

# Normal: Scheduled notification (not time-sensitive)
newsletter_job = Job(
    id=uuid4(),
    queue_name="emails",
    job_type="email.newsletter",
    payload={"list": "subscribers"},
    priority=3  # Low priority - can wait
)

# ❌ BAD - Priority based on business importance
payment_job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="payment.record",  # Important but not urgent
    payload={"amount": 1000000},  # Large amount
    priority=10  # ❌ High priority just because amount is large
)
```

### 5. Leverage Metadata for Context

Use metadata for extensibility without cluttering payload.

```python
# ✅ GOOD - Metadata for cross-cutting concerns
job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={
        # Only data needed by handler
        "report_type": "sales",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31"
    },
    metadata={
        # Cross-cutting concerns
        "tenant_id": "tenant_123",
        "cost_center": "CC-SALES",
        "requested_by": "manager@example.com",
        "correlation_id": "req_abc123",
        "tags": ["monthly", "executive"]
    }
)

# Metadata can be used for:
# - Audit trails (who requested, when)
# - Cost allocation (tenant, cost center)
# - Distributed tracing (correlation IDs)
# - Filtering/searching
# - Business analytics

# ❌ BAD - Everything in payload
job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={
        "report_type": "sales",
        "start_date": "2025-01-01",
        "tenant_id": "tenant_123",  # ❌ Cross-cutting concern
        "requested_by": "manager@example.com",  # ❌ Not needed by handler
        "correlation_id": "req_abc123"  # ❌ Audit data
    }
)
```

### 6. Use Scheduled Jobs for Delays

Use `scheduled_at` for delayed execution instead of manual delays.

```python
from datetime import UTC, datetime, timedelta

# ✅ GOOD - Use scheduled_at
future_job = Job(
    id=uuid4(),
    queue_name="reminders",
    job_type="reminder.send",
    payload={"user_id": "user_123"},
    scheduled_at=datetime.now(UTC) + timedelta(hours=24)
)

# Job won't be processed until scheduled_at time
# Queue adapter handles the delay efficiently

# ❌ BAD - Handler delays execution
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="delayed.task",
    payload={"delay_seconds": 86400}  # ❌ Handler must sleep
)

# Handler implementation (bad):
async def handle(self, job: Job):
    await asyncio.sleep(job.payload["delay_seconds"])  # ❌ Blocks worker
    await self.do_work()
```

### 7. Track Job Ownership for Audit Trails

Use `created_by` to track who created jobs.

```python
from uuid import UUID, uuid4

# ✅ GOOD - Track job creator
def create_report_job(user_id: UUID, report_type: str):
    return Job(
        id=uuid4(),
        queue_name="reports",
        job_type="report.generate",
        payload={"report_type": report_type},
        created_by=user_id,  # Track who requested the report
        metadata={
            "requested_at": datetime.now(UTC).isoformat(),
            "ip_address": "192.168.1.100"
        }
    )

# Enables audit queries:
# - Which user created the most jobs?
# - Who requested this report?
# - What jobs did user X create?

# ❌ BAD - No creator tracking
job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={"report_type": "sales"},
    # ❌ No created_by - can't trace back to user
)
```

## FAQs

### Why are Job models immutable (frozen=True)?

Immutability provides several benefits:

1. **Thread safety** - Jobs can be safely shared across workers without locking
2. **Consistency** - Job state can't be accidentally modified during processing
3. **Audit trail** - Each state change creates a new Job instance, preserving history
4. **Functional programming** - Enables clean, predictable transformations

```python
from portico.ports.job import Job, JobStatus

# Job is immutable - can't modify fields
job = Job(id=uuid4(), queue_name="default", job_type="test", payload={})

# ❌ This would raise an error
# job.status = JobStatus.RUNNING  # FrozenInstanceError

# ✅ Create new instance with updated fields
updated_job = Job(**{**job.__dict__, "status": JobStatus.RUNNING})
```

### How do I update a Job's status?

Since Jobs are immutable, create a new Job instance with updated fields using dictionary unpacking:

```python
from datetime import UTC, datetime

# Original job
job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="email.send",
    payload={"recipient": "user@example.com"},
    status=JobStatus.PENDING
)

# Update status to RUNNING
running_job = Job(
    **{**job.__dict__,
        "status": JobStatus.RUNNING,
        "started_at": datetime.now(UTC)
    }
)

# Update status to COMPLETED
completed_job = Job(
    **{**running_job.__dict__,
        "status": JobStatus.COMPLETED,
        "completed_at": datetime.now(UTC)
    }
)
```

### What's the difference between FAILED and DEAD_LETTER status?

**FAILED**: Job failed and will NOT be retried
- Occurs when `requeue=False` on rejection
- Or when handler explicitly returns failure without retrying
- Typically for validation errors or non-retriable failures

**DEAD_LETTER**: Job exhausted all retry attempts
- Occurs when `retry_count >= max_retries`
- Job attempted multiple times but kept failing
- Moved to dead letter queue for manual review

```python
# FAILED - validation error, don't retry
await job_queue.reject(
    job.id,
    requeue=False,  # Don't retry
    error_message="Invalid email address"
)
# Job status becomes FAILED

# DEAD_LETTER - exhausted retries
job_with_retries = Job(
    id=uuid4(),
    queue_name="default",
    job_type="test",
    payload={},
    max_retries=3,
    retry_count=3  # Already retried 3 times
)

await job_queue.reject(
    job_with_retries.id,
    requeue=True,  # Try to requeue
    error_message="Still failing"
)
# Job status becomes DEAD_LETTER (exhausted retries)
```

### Should I put large data in job payloads?

No. Job payloads should contain **references** to data, not the data itself.

```python
# ✅ GOOD - Reference to data
job = Job(
    id=uuid4(),
    queue_name="processing",
    job_type="file.process",
    payload={
        "file_path": "s3://bucket/uploads/data.csv",  # Reference
        "file_id": "file_123"
    }
)

# Handler fetches data
async def handle(self, job: Job):
    file_path = job.payload["file_path"]
    data = await self.storage.download(file_path)
    # Process data...

# ❌ BAD - Large data in payload
job = Job(
    id=uuid4(),
    queue_name="processing",
    job_type="data.process",
    payload={
        "data": "...10MB of data...",  # ❌ Too large
        "records": [{"id": 1, "name": "..."}, ...]  # ❌ 10,000 records
    }
)

# Problems:
# - Slow to serialize/deserialize
# - High memory usage
# - Database/queue storage bloat
# - Network overhead
```

### How do I implement job dependencies?

Job models don't have built-in dependency tracking. Implement in your handler:

```python
# Option 1: Chain jobs in handler
class ReportJobHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        # Process report
        report = await self.generate_report(job.payload)

        # Create dependent job
        await self.job_creator.create_job(
            job_type="email.send",
            payload={
                "recipient": job.payload["user_email"],
                "subject": "Report Ready",
                "report_url": report.url
            }
        )

        return JobResult(success=True, result_data={"report_id": report.id})

# Option 2: Use metadata to track dependencies
parent_job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={"report_type": "sales"}
)

child_job = Job(
    id=uuid4(),
    queue_name="emails",
    job_type="email.send",
    payload={"subject": "Report Ready"},
    metadata={"parent_job_id": str(parent_job.id)}  # Track parent
)
```

### How do I query jobs by status?

Job models are data structures. Querying is handled by queue adapters:

```python
# Queue adapter provides query methods
class JobQueueAdapter(ABC):
    async def get_job(self, job_id: UUID) -> Job | None:
        """Get job by ID."""

    async def list_jobs_by_status(self, status: JobStatus) -> list[Job]:
        """List jobs by status."""

    async def list_jobs_by_type(self, job_type: str) -> list[Job]:
        """List jobs by type."""

# Usage
dead_letter_jobs = await job_queue.list_jobs_by_status(JobStatus.DEAD_LETTER)
for job in dead_letter_jobs:
    print(f"Failed job: {job.id}, error: {job.error_message}")
```

### What should I put in JobResult.result_data?

Include data useful for:
- Monitoring and metrics
- Audit trails
- Downstream processing
- Debugging

```python
# ✅ GOOD - Useful result data
return JobResult(
    success=True,
    result_data={
        # Metrics
        "duration_ms": 1234,
        "records_processed": 1000,

        # IDs for audit trail
        "report_id": "rpt_123",
        "file_path": "s3://bucket/report.pdf",

        # Business data
        "total_sales": 50000.00,
        "top_product": "Widget A",

        # Warnings (success but with issues)
        "warnings": ["Record 45 had invalid date"]
    }
)

# ❌ BAD - Useless result data
return JobResult(
    success=True,
    result_data={"status": "ok"}  # ❌ Not useful
)
```

### How do I handle job cancellation?

Job models don't have a CANCELLED status. Use queue adapter's cancel method:

```python
# Cancel pending job
cancelled = await job_queue.cancel_job(job_id)

if cancelled:
    # Job was cancelled, status set to FAILED
    job = await job_queue.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert job.error_message == "Job cancelled"
else:
    # Job already running, can't cancel
    print("Job already started, cannot cancel")

# For long-running jobs, check cancellation flag
class LongRunningHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        for i in range(1000):
            # Check if job was cancelled
            current_job = await self.job_queue.get_job(job.id)
            if current_job.status == JobStatus.FAILED:
                return JobResult(success=False, error=Exception("Cancelled"))

            # Process item
            await self.process_item(i)
```

## Related Ports

All job-related ports use these domain models:

- **[Job Creator Port](job_creator.md)** - Creates Job instances
- **[Job Handler Port](job_handler.md)** - Processes Jobs and returns JobResult
- **[Job Queue Port](job_queue.md)** - Stores and retrieves Jobs, provides QueueStats
- **[Job Trigger Port](job_trigger.md)** - Creates Jobs from external events

## Architecture Notes

The Job domain models follow hexagonal architecture principles:

- **Domain models are the core** - All ports depend on these shared models
- **Immutability ensures consistency** - Jobs can't be accidentally modified
- **Type safety** - Pydantic-style dataclasses with full type annotations
- **Separation of concerns** - Models define structure, ports define behavior

This pattern enables:
- Consistent job representation across all components
- Type-safe job processing
- Clear separation between data and behavior
- Easy testing with predictable data structures
