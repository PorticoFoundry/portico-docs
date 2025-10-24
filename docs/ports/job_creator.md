# Job Creator Port

## Overview

The Job Creator Port defines the contract for creating and enqueueing background jobs. It serves as a dependency inversion interface allowing job triggers (adapters) to create jobs without depending on the concrete JobService implementation.

**Purpose**: Enable job creation while maintaining hexagonal architecture - adapters depend on this port, not on the service implementation.

**Domain**: Background job processing, task queuing, asynchronous operations

**Key Capabilities**:

- **Job Creation**: Create jobs with type, payload, and priority
- **Queue Selection**: Route jobs to specific named queues
- **Priority Management**: Set job priority for execution ordering
- **Scheduled Jobs**: Create jobs scheduled for future execution
- **Extensible**: Accepts implementation-specific kwargs (max_retries, created_by, metadata)
- **Hexagonal Architecture**: Adapters depend on ports, not kits/services

**Port Type**: Provider (Factory pattern)

**When to Use**:

- Job triggers (adapters) that create background jobs
- External systems scheduling tasks (webhooks, cron, API endpoints)
- Event handlers that need to enqueue work
- Systems requiring dependency inversion between adapters and services
- Any code that creates jobs but shouldn't depend on JobService directly

**When NOT to Use**:

- Direct job execution (use job handlers/workers)
- Job status queries (use JobService or job repository)
- Job cancellation or management operations
- If you're implementing the job service itself (implement the interface instead)

## Domain Models

### Job

Domain model representing a background job (defined in `portico.ports.job`).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | - | Unique job identifier |
| `queue_name` | `str` | Yes | - | Queue name for job routing |
| `job_type` | `str` | Yes | - | Job type identifier (e.g., "email.send", "document.process") |
| `payload` | `dict[str, Any]` | Yes | - | Job payload data (must be JSON-serializable) |
| `priority` | `int` | No | `0` | Job priority (higher = more urgent) |
| `max_retries` | `int` | No | `3` | Maximum retry attempts on failure |
| `retry_count` | `int` | No | `0` | Current retry attempt count |
| `scheduled_at` | `Optional[datetime]` | No | `None` | Scheduled execution time (None = immediate) |
| `started_at` | `Optional[datetime]` | No | `None` | Job start timestamp |
| `completed_at` | `Optional[datetime]` | No | `None` | Job completion timestamp |
| `failed_at` | `Optional[datetime]` | No | `None` | Job failure timestamp |
| `status` | `JobStatus` | No | `PENDING` | Current job status |
| `error_message` | `Optional[str]` | No | `None` | Error message if failed |
| `created_by` | `Optional[UUID]` | No | `None` | User ID that created the job |
| `created_at` | `Optional[datetime]` | No | `None` | Job creation timestamp |
| `metadata` | `dict[str, Any]` | No | `{}` | Additional job metadata |

**Example**:

```python
from portico.ports.job import Job, JobStatus
from datetime import datetime
from uuid import uuid4

job = Job(
    id=uuid4(),
    queue_name="emails",
    job_type="email.send",
    payload={
        "to": "user@example.com",
        "subject": "Welcome!",
        "template": "welcome_email",
    },
    priority=5,
    max_retries=3,
    status=JobStatus.PENDING,
    created_by=uuid4(),
    created_at=datetime.now(),
    metadata={"campaign": "onboarding"},
)
```

## Enumerations

### JobStatus

Job execution status (defined in `portico.ports.job`).

| Value | Description |
|-------|-------------|
| `PENDING` | Job created and waiting to be processed |
| `SCHEDULED` | Job scheduled for future execution |
| `RUNNING` | Job currently executing |
| `COMPLETED` | Job completed successfully |
| `FAILED` | Job failed (may retry) |
| `RETRYING` | Job failed and is being retried |
| `DEAD_LETTER` | Job failed after all retries exhausted |

**Example**:

```python
from portico.ports.job import JobStatus

# Check job status
if job.status == JobStatus.PENDING:
    print("Job waiting to execute")
elif job.status == JobStatus.COMPLETED:
    print("Job finished successfully")
elif job.status == JobStatus.FAILED:
    print(f"Job failed: {job.error_message}")
```

## Port Interface

### JobCreator

The `JobCreator` abstract base class defines the contract for creating jobs. Job triggers (adapters) use this interface to create jobs without depending on the concrete JobService implementation, maintaining hexagonal architecture.

**Location**: `portico.ports.job_creator.JobCreator`

**Architectural Pattern**:

```
┌─────────────────────────────────┐
│  Job Triggers (Adapters)        │
│  - ScheduleTrigger               │
│  - WebhookTrigger                │
│  - EventTrigger                  │
└────────────┬────────────────────┘
             │ depends on
             ↓
┌─────────────────────────────────┐
│  JobCreator (Port/Interface)    │  ← Dependency inversion
│  - create_job()                  │
└────────────┬────────────────────┘
             ↑ implements
             │
┌─────────────────────────────────┐
│  JobService (Kit/Implementation) │
│  - create_job()                  │
│  - get_job()                     │
│  - process_jobs()                │
└─────────────────────────────────┘
```

**Key Method**

##### create_job

```python
async def create_job(
    job_type: str,
    payload: Dict[str, Any],
    queue_name: str = "default",
    priority: int = 0,
    **kwargs: Any,
) -> Job
```

Create a job and enqueue it for processing.

**Parameters**:

- `job_type`: Job type identifier (must have registered handler, e.g., "email.send", "document.process")
- `payload`: Job payload data (must be JSON-serializable dict)
- `queue_name`: Queue name to submit job to (default: "default")
- `priority`: Job priority for execution ordering (higher = more urgent, default: 0)
- `**kwargs`: Implementation-specific options:
  - `scheduled_at` (datetime): Schedule job for future execution
  - `max_retries` (int): Override default retry limit
  - `created_by` (UUID): User ID that created the job
  - `metadata` (dict): Additional job metadata

**Returns**: Created Job object with generated ID and PENDING/SCHEDULED status

**Raises**:
- `ValueError`: If job_type is not registered
- `ValidationError`: If payload is not JSON-serializable

**Example**:

```python
from portico.ports.job_creator import JobCreator
from datetime import datetime, timedelta
from uuid import uuid4

# Basic job creation
job = await job_creator.create_job(
    job_type="email.send",
    payload={
        "to": "user@example.com",
        "subject": "Order Confirmation",
        "template": "order_confirmation",
        "order_id": "12345",
    },
)

# High-priority job with custom queue
job = await job_creator.create_job(
    job_type="payment.process",
    payload={"transaction_id": "txn_789", "amount": 99.99},
    queue_name="payments",
    priority=10,
)

# Scheduled job with metadata
job = await job_creator.create_job(
    job_type="report.generate",
    payload={"report_type": "monthly", "month": "2025-03"},
    scheduled_at=datetime.now() + timedelta(days=1),
    created_by=uuid4(),
    metadata={"department": "finance"},
)

# Job with custom retry limit
job = await job_creator.create_job(
    job_type="api.webhook",
    payload={"url": "https://api.example.com/webhook", "event": "user.created"},
    max_retries=5,
)
```

## Common Patterns

### Job Trigger Using JobCreator

```python
from portico.ports.job_creator import JobCreator
from portico.utils.logging import get_logger

logger = get_logger(__name__)


class ScheduleTrigger:
    """Job trigger that creates scheduled jobs.

    This adapter depends on JobCreator port, not JobService directly.
    This maintains hexagonal architecture.
    """

    def __init__(self, job_creator: JobCreator, schedules: list):
        self.job_creator = job_creator
        self.schedules = schedules

    async def trigger_daily_report(self):
        """Trigger daily report generation job."""
        job = await self.job_creator.create_job(
            job_type="report.daily",
            payload={
                "date": datetime.now().isoformat(),
                "report_type": "sales",
            },
            queue_name="reports",
            priority=5,
        )

        logger.info(
            "daily_report_scheduled",
            job_id=str(job.id),
            scheduled_for=job.scheduled_at,
        )

        return job

    async def trigger_user_cleanup(self):
        """Trigger user cleanup job."""
        job = await self.job_creator.create_job(
            job_type="user.cleanup",
            payload={"inactive_days": 90},
            queue_name="maintenance",
            metadata={"trigger": "schedule", "frequency": "weekly"},
        )

        logger.info("cleanup_job_created", job_id=str(job.id))
        return job
```

### Webhook Handler Creating Jobs

```python
from fastapi import APIRouter, HTTPException
from portico.ports.job_creator import JobCreator
from pydantic import BaseModel


class WebhookPayload(BaseModel):
    event: str
    data: dict


def create_webhook_router(job_creator: JobCreator) -> APIRouter:
    """Create webhook router that enqueues jobs."""
    router = APIRouter()

    @router.post("/webhooks/external")
    async def handle_external_webhook(payload: WebhookPayload):
        """Handle incoming webhook by creating a job."""
        try:
            # Create job to process webhook asynchronously
            job = await job_creator.create_job(
                job_type=f"webhook.{payload.event}",
                payload=payload.data,
                queue_name="webhooks",
                priority=7,
                metadata={
                    "source": "external_api",
                    "event": payload.event,
                },
            )

            return {
                "status": "accepted",
                "job_id": str(job.id),
                "message": "Webhook queued for processing",
            }

        except ValueError as e:
            # Job type not registered
            raise HTTPException(status_code=400, detail=str(e))

    return router
```

### Event Handler Creating Background Jobs

```python
from portico.ports.job_creator import JobCreator
from portico.events import EventBus, Event


class UserEventHandler:
    """Event handler that creates background jobs for user events."""

    def __init__(self, job_creator: JobCreator, event_bus: EventBus):
        self.job_creator = job_creator
        self.event_bus = event_bus

        # Subscribe to events
        event_bus.subscribe("user.created", self.on_user_created)
        event_bus.subscribe("user.deleted", self.on_user_deleted)

    async def on_user_created(self, event: Event):
        """Handle user creation by enqueuing welcome email."""
        user_id = event.data["user_id"]
        user_email = event.data["email"]

        # Create welcome email job
        await self.job_creator.create_job(
            job_type="email.send",
            payload={
                "to": user_email,
                "template": "welcome_email",
                "user_id": str(user_id),
            },
            queue_name="emails",
            priority=5,
            created_by=user_id,
        )

        # Create onboarding tasks job
        await self.job_creator.create_job(
            job_type="onboarding.create_tasks",
            payload={"user_id": str(user_id)},
            queue_name="default",
            metadata={"trigger": "user_created"},
        )

    async def on_user_deleted(self, event: Event):
        """Handle user deletion by enqueuing cleanup."""
        user_id = event.data["user_id"]

        await self.job_creator.create_job(
            job_type="user.cleanup_data",
            payload={"user_id": str(user_id)},
            queue_name="maintenance",
            priority=3,
            metadata={"trigger": "user_deleted"},
        )
```

### Mock JobCreator for Testing

```python
from portico.ports.job_creator import JobCreator
from portico.ports.job import Job, JobStatus
from uuid import uuid4


class MockJobCreator(JobCreator):
    """Mock JobCreator for testing adapters without real job service."""

    def __init__(self):
        self.created_jobs = []

    async def create_job(
        self,
        job_type: str,
        payload: dict,
        queue_name: str = "default",
        priority: int = 0,
        **kwargs,
    ) -> Job:
        """Mock job creation that tracks all created jobs."""
        job_id = uuid4()
        job = Job(
            id=job_id,
            queue_name=queue_name,
            job_type=job_type,
            payload=payload,
            priority=priority,
            status=JobStatus.PENDING,
            **kwargs,
        )
        self.created_jobs.append(job)
        return job

    def get_created_jobs(self, job_type: str = None) -> list[Job]:
        """Get all created jobs, optionally filtered by type."""
        if job_type:
            return [j for j in self.created_jobs if j.job_type == job_type]
        return self.created_jobs

    def reset(self):
        """Reset tracked jobs."""
        self.created_jobs.clear()


# Usage in tests
@pytest.mark.asyncio
async def test_webhook_handler_creates_job():
    """Test webhook handler creates correct job."""
    mock_creator = MockJobCreator()
    handler = WebhookHandler(job_creator=mock_creator)

    await handler.process_webhook({"event": "user.created", "data": {...}})

    # Verify job was created
    assert len(mock_creator.created_jobs) == 1
    job = mock_creator.created_jobs[0]
    assert job.job_type == "webhook.user.created"
    assert job.queue_name == "webhooks"
```

## Integration with Kits

The JobCreator Port is implemented by the **Job Service** (JobService kit).

```python
from portico.kits.job import JobService
from portico.adapters.job_queue import MemoryJobQueueAdapter

# JobService implements JobCreator interface
job_queue = MemoryJobQueueAdapter()
job_service = JobService(job_queue=job_queue)

# JobService is a JobCreator
assert isinstance(job_service, JobCreator)

# Use as JobCreator
job = await job_service.create_job(
    job_type="email.send",
    payload={"to": "user@example.com"},
)

# Job triggers receive JobCreator, not JobService
# This maintains dependency inversion
trigger = ScheduleTrigger(job_creator=job_service, schedules=[...])
```

**Hexagonal Architecture Benefits**:

```python
# ✅ GOOD - Adapter depends on port
class MyTrigger:
    def __init__(self, job_creator: JobCreator):  # Port interface
        self.job_creator = job_creator

    async def trigger(self):
        await self.job_creator.create_job(...)

# ❌ BAD - Adapter depends on concrete implementation
class MyTrigger:
    def __init__(self, job_service: JobService):  # Concrete kit
        self.job_service = job_service

    async def trigger(self):
        await self.job_service.create_job(...)
```

## Best Practices

1. **Use JobCreator Interface in Adapters**: Depend on the port, not the service implementation

   ```python
   # ✅ GOOD - Depends on port interface
   class WebhookAdapter:
       def __init__(self, job_creator: JobCreator):
           self.job_creator = job_creator

       async def handle_webhook(self, payload):
           await self.job_creator.create_job("webhook.process", payload)

   # ❌ BAD - Depends on concrete service
   from portico.kits.job import JobService

   class WebhookAdapter:
       def __init__(self, job_service: JobService):
           self.job_service = job_service
   ```

2. **Use Descriptive Job Types**: Use namespaced job type identifiers

   ```python
   # ✅ GOOD - Clear, namespaced job types
   await job_creator.create_job(
       job_type="email.send",
       payload={...},
   )

   await job_creator.create_job(
       job_type="document.process",
       payload={...},
   )

   # ❌ BAD - Generic, unclear job types
   await job_creator.create_job(
       job_type="send",
       payload={...},
   )
   ```

3. **Ensure Payload is JSON-Serializable**: Only include serializable data

   ```python
   # ✅ GOOD - Serializable payload
   await job_creator.create_job(
       job_type="user.notify",
       payload={
           "user_id": str(user.id),  # Convert UUID to string
           "timestamp": datetime.now().isoformat(),  # ISO format
           "metadata": {"key": "value"},
       },
   )

   # ❌ BAD - Non-serializable payload
   await job_creator.create_job(
       job_type="user.notify",
       payload={
           "user_id": user.id,  # UUID object
           "timestamp": datetime.now(),  # datetime object
           "user": user,  # Complex object
       },
   )
   ```

4. **Use Appropriate Priorities**: Reserve high priorities for critical jobs

   ```python
   # ✅ GOOD - Priorities match importance
   # Critical payment processing
   await job_creator.create_job(
       job_type="payment.process",
       payload={...},
       priority=10,
   )

   # Normal notification
   await job_creator.create_job(
       job_type="email.send",
       payload={...},
       priority=5,
   )

   # Low priority cleanup
   await job_creator.create_job(
       job_type="data.cleanup",
       payload={...},
       priority=1,
   )

   # ❌ BAD - Everything high priority
   await job_creator.create_job(
       job_type="email.send",
       payload={...},
       priority=10,  # Not actually critical
   )
   ```

5. **Route Jobs to Appropriate Queues**: Use dedicated queues for different job types

   ```python
   # ✅ GOOD - Jobs routed to appropriate queues
   await job_creator.create_job(
       job_type="email.send",
       payload={...},
       queue_name="emails",
   )

   await job_creator.create_job(
       job_type="report.generate",
       payload={...},
       queue_name="reports",
   )

   # ❌ BAD - Everything in default queue
   await job_creator.create_job(
       job_type="email.send",
       payload={...},
       # Uses default queue - may block other jobs
   )
   ```

6. **Add Metadata for Tracking**: Include metadata for debugging and analytics

   ```python
   # ✅ GOOD - Rich metadata
   await job_creator.create_job(
       job_type="webhook.process",
       payload={...},
       created_by=current_user.id,
       metadata={
           "source": "external_api",
           "webhook_id": "wh_123",
           "retry_strategy": "exponential",
       },
   )

   # ❌ BAD - No context
   await job_creator.create_job(
       job_type="webhook.process",
       payload={...},
   )
   ```

7. **Use MockJobCreator for Testing**: Test adapters without real job infrastructure

   ```python
   # ✅ GOOD - Mock for testing
   @pytest.mark.asyncio
   async def test_trigger_creates_job():
       mock_creator = MockJobCreator()
       trigger = MyTrigger(job_creator=mock_creator)

       await trigger.execute()

       # Verify job creation
       assert len(mock_creator.created_jobs) == 1
       assert mock_creator.created_jobs[0].job_type == "expected.type"

   # ❌ BAD - Requires real job service
   async def test_trigger_creates_job():
       job_service = JobService(...)  # Complex setup
       trigger = MyTrigger(job_creator=job_service)
   ```

## FAQs

### What's the difference between JobCreator and JobService?

`JobCreator` is a port (interface) with a single method `create_job()`, while `JobService` is the full service implementation (kit) that includes job creation, querying, processing, and management.

- **JobCreator**: Port interface for creating jobs (used by adapters)
- **JobService**: Complete job service implementation (implements JobCreator + more)

Adapters should depend on `JobCreator` to maintain hexagonal architecture - they don't need the full service functionality.

### Why use JobCreator instead of calling JobService directly?

This maintains hexagonal architecture and dependency inversion:

- **Adapters** (outer layer) depend on **Ports** (interfaces), not **Kits** (services)
- Makes adapters testable with mock implementations
- Reduces coupling between layers
- Follows SOLID principles (Dependency Inversion Principle)

```python
# ✅ Hexagonal Architecture
class MyAdapter:
    def __init__(self, job_creator: JobCreator):  # Depends on port
        pass

# ❌ Violates Hexagonal Architecture
class MyAdapter:
    def __init__(self, job_service: JobService):  # Depends on kit
        pass
```

### How do I schedule a job for future execution?

Use the `scheduled_at` kwarg with a future datetime:

```python
from datetime import datetime, timedelta

# Schedule for 1 hour from now
job = await job_creator.create_job(
    job_type="reminder.send",
    payload={"user_id": "123", "message": "Don't forget!"},
    scheduled_at=datetime.now() + timedelta(hours=1),
)

# Job status will be SCHEDULED (not PENDING)
assert job.status == JobStatus.SCHEDULED
```

### What kwargs does create_job() accept?

The base interface requires `job_type`, `payload`, `queue_name`, and `priority`. Implementations may accept additional kwargs:

- `scheduled_at` (datetime): Schedule for future execution
- `max_retries` (int): Override retry limit
- `created_by` (UUID): User who created the job
- `metadata` (dict): Additional tracking information

Check your JobService implementation documentation for supported kwargs.

### How do I ensure my payload is JSON-serializable?

Convert all non-serializable types before creating the job:

```python
from datetime import datetime
from uuid import UUID

# Convert types to serializable formats
job = await job_creator.create_job(
    job_type="user.process",
    payload={
        "user_id": str(user_id),  # UUID → str
        "timestamp": datetime.now().isoformat(),  # datetime → ISO string
        "amount": float(amount),  # Decimal → float
        "data": dict(data),  # Custom object → dict
    },
)
```

### What happens if I create a job with an unregistered job_type?

The implementation will raise `ValueError` if the job type doesn't have a registered handler:

```python
try:
    job = await job_creator.create_job(
        job_type="nonexistent.job",
        payload={...},
    )
except ValueError as e:
    print(f"Job type not registered: {e}")
    # Handle error - maybe log and skip, or notify admin
```

Register job handlers before creating jobs of that type.

### How do I test code that uses JobCreator?

Use `MockJobCreator` to test without real job infrastructure:

```python
from portico.ports.job_creator import JobCreator
from portico.ports.job import Job, JobStatus

class MockJobCreator(JobCreator):
    def __init__(self):
        self.created_jobs = []

    async def create_job(self, job_type, payload, **kwargs):
        job = Job(id=uuid4(), job_type=job_type, payload=payload, ...)
        self.created_jobs.append(job)
        return job

# Use in tests
@pytest.mark.asyncio
async def test_my_adapter():
    mock_creator = MockJobCreator()
    adapter = MyAdapter(job_creator=mock_creator)

    await adapter.do_something()

    # Verify job creation
    assert len(mock_creator.created_jobs) == 1
    assert mock_creator.created_jobs[0].job_type == "expected.type"
```

### Can I create jobs synchronously?

No, `create_job()` is an async method and must be awaited. Job creation involves I/O operations (database, queue):

```python
# ✅ CORRECT - Async usage
job = await job_creator.create_job(...)

# ❌ WRONG - Synchronous attempt
job = job_creator.create_job(...)  # Returns coroutine, doesn't execute
```

If you need to create jobs from synchronous code, you'll need to run it in an async context:

```python
import asyncio

def sync_function():
    # Create event loop to run async code
    job = asyncio.run(job_creator.create_job(...))
```
