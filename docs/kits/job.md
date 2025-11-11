# Job Kit

## Overview

The Job Kit provides background job processing capabilities for asynchronous task execution. It enables applications to offload long-running operations, scheduled tasks, and event-driven workflows to background workers with robust retry mechanisms and failure handling.

**Purpose**: Decouple time-consuming operations from request-response cycles and provide reliable async task execution.

**Domain**: Background job processing, task queues, worker management, async execution

**Capabilities**:

- Create and enqueue jobs with custom payloads
- Manage worker processes that execute jobs
- Automatic retry with exponential backoff on failures
- Scheduled job execution with delays
- Job status tracking and result storage
- Pluggable queue backends through JobQueueAdapter
- Handler-based job processing with custom business logic
- Integration with audit logging for job lifecycle events

**Architecture Type**: Service/Utility layer (not a traditional Kit with compose factory)

**When to Use**:

- Long-running operations (report generation, data processing, ML inference)
- Scheduled tasks (cleanup jobs, batch updates, periodic syncs)
- Event-driven workflows (webhook processing, notification delivery)
- Rate-limited operations (API calls with quotas)
- Fire-and-forget tasks (email sending, analytics tracking)

## Quick Start

### Creating and Executing Jobs

```python
from portico.kits.job import JobService, WorkerManager
from portico.ports.job_handler import JobHandler
from portico.ports.job import Job, JobResult
from portico.adapters.job_queue import InMemoryJobQueue

# Define a job handler
class SendEmailHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        email = job.payload.get("email")
        subject = job.payload.get("subject")

        # Send email logic
        await send_email(email, subject)

        return JobResult(
            job_id=job.id,
            status="completed",
            result={"sent_at": datetime.utcnow().isoformat()}
        )

    async def on_failure(self, job: Job, error: Exception) -> None:
        # Handle failure (logging, alerts, etc.)
        logger.error(f"Failed to send email: {error}")

# Create job service
queue_adapter = InMemoryJobQueue()
job_service = JobService(
    queue_adapter=queue_adapter,
    audit_service=audit_service
)

# Register handlers
handlers = {
    "send_email": SendEmailHandler()
}

# Start workers
worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,
    num_workers=4
)
await worker_manager.start()

# Create and enqueue a job
job = await job_service.create_job(
    job_type="send_email",
    payload={"email": "user@example.com", "subject": "Welcome!"},
    created_by="user_123"
)
print(f"Job created: {job.id}")

# Stop workers when done
await worker_manager.stop()
```

## Core Concepts

### JobService

The `JobService` class implements the `JobCreator` interface and provides methods for creating, retrieving, and managing jobs. It integrates with the audit service to track job lifecycle events.

**Key Methods**:

- `create_job()` - Create and enqueue a new job
- `get_job()` - Retrieve job by ID
- `cancel_job()` - Cancel a pending job
- `get_job_stats()` - Get queue statistics

**Integration**: JobService automatically publishes audit events for job creation, completion, and cancellation.

### WorkerManager

The `WorkerManager` orchestrates worker processes that consume jobs from the queue and execute them using registered handlers. It manages worker lifecycle, handles retries, and ensures graceful shutdown.

**Key Features**:

- Configurable worker count
- Automatic retry with exponential backoff
- Graceful shutdown with in-flight job completion
- Error handling and logging
- Worker health monitoring

### Job Handlers

Job handlers contain the business logic for processing specific job types. Each handler implements the `JobHandler` interface with two methods:

- `handle()` - Process the job and return a result
- `on_failure()` - Handle job failures (optional cleanup, logging, alerts)

### Job Queue Adapter

The `JobQueueAdapter` port interface abstracts queue operations, allowing different backend implementations (in-memory, Redis, RabbitMQ, SQS, etc.).

**Key Operations**:

- `enqueue()` - Add job to queue
- `dequeue()` - Fetch next job for processing
- `acknowledge()` - Mark job as successfully processed
- `reject()` - Return job to queue for retry
- `get_stats()` - Get queue metrics

### Retry Logic

Jobs that fail are automatically retried with exponential backoff:

1. Handler raises exception
2. Worker calls `reject()` to return job to queue
3. Job's `retry_count` increments
4. Job is re-enqueued with delay: `base_delay * (2 ^ retry_count)`
5. Process repeats until max retries reached or job succeeds

### Job Lifecycle

```
Created → Queued → Processing → Completed
                       ↓
                    Failed → Retry (if retries remaining)
                       ↓
                  Max Retries → Dead Letter Queue
```

## Configuration

The Job Kit does not have a standard `JobKitConfig` class. Configuration is managed through:

### JobService Configuration

```python
job_service = JobService(
    queue_adapter=queue_adapter,  # Required: queue implementation
    audit_service=audit_service   # Optional: for audit logging
)
```

### WorkerManager Configuration

```python
worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,           # Dict[str, JobHandler]
    num_workers=4,               # Number of concurrent workers
    poll_interval=1.0,           # Seconds between queue polls
    max_retries=3,               # Max retry attempts per job
    base_retry_delay=2.0         # Base delay for exponential backoff
)
```

### Queue Adapter Configuration

Configuration depends on the specific adapter implementation:

```python
# In-memory (dev/testing)
queue_adapter = InMemoryJobQueue()

# Redis (production)
queue_adapter = RedisJobQueue(redis_url="redis://localhost:6379/0")

# Custom adapter
queue_adapter = MyCustomQueueAdapter(config=my_config)
```

## Usage Examples

### 1. Creating Scheduled Jobs

```python
from datetime import datetime, timedelta

# Schedule job for future execution
scheduled_time = datetime.utcnow() + timedelta(hours=24)

job = await job_service.create_job(
    job_type="cleanup_old_files",
    payload={"max_age_days": 90},
    created_by="system",
    scheduled_at=scheduled_time
)

print(f"Job scheduled for {scheduled_time}")
```

### 2. Implementing a Data Processing Handler

```python
from portico.ports.job_handler import JobHandler
from portico.ports.job import Job, JobResult

class ProcessCsvHandler(JobHandler):
    def __init__(self, database, file_storage):
        self.database = database
        self.file_storage = file_storage

    async def handle(self, job: Job) -> JobResult:
        file_id = job.payload["file_id"]

        # Download file
        file_data = await self.file_storage.get_file(file_id)

        # Process CSV
        rows_processed = 0
        async with self.database.transaction():
            for row in parse_csv(file_data):
                await self.database.insert_record(row)
                rows_processed += 1

        return JobResult(
            job_id=job.id,
            status="completed",
            result={"rows_processed": rows_processed}
        )

    async def on_failure(self, job: Job, error: Exception) -> None:
        # Clean up partial data
        file_id = job.payload["file_id"]
        await self.database.rollback_import(file_id)

        # Alert admin
        await self.notify_admin(f"CSV import failed: {error}")
```

### 3. Worker Management in Application Lifecycle

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start workers
    worker_manager = app.state.worker_manager
    await worker_manager.start()
    print("Workers started")

    yield

    # Shutdown: Stop workers gracefully
    await worker_manager.stop()
    print("Workers stopped")

app = FastAPI(lifespan=lifespan)

# Store worker manager in app state
app.state.worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,
    num_workers=4
)
```

### 4. Job Status Tracking

```python
# Create job
job = await job_service.create_job(
    job_type="generate_report",
    payload={"report_id": "annual_2024"},
    created_by="user_456"
)

# Poll for completion
import asyncio

while True:
    current_job = await job_service.get_job(job.id)

    if current_job.status == "completed":
        print(f"Report ready: {current_job.result}")
        break
    elif current_job.status == "failed":
        print(f"Report generation failed: {current_job.error}")
        break

    await asyncio.sleep(2)  # Poll every 2 seconds
```

### 5. Priority Queue with Multiple Job Types

```python
# Register handlers for different job types
handlers = {
    "send_email": SendEmailHandler(),
    "process_csv": ProcessCsvHandler(database, file_storage),
    "generate_report": ReportGeneratorHandler(database),
    "cleanup": CleanupHandler(file_storage),
}

worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,
    num_workers=8  # Scale workers for throughput
)

await worker_manager.start()

# Enqueue jobs of different types
await job_service.create_job("send_email", {"email": "user@example.com"}, "system")
await job_service.create_job("process_csv", {"file_id": "csv_123"}, "user_789")
await job_service.create_job("generate_report", {"month": "2024-01"}, "admin_001")
```

## Domain Models

### Job

Represents a background job to be executed.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique job identifier (UUID) |
| `job_type` | `str` | Type of job (matches handler key) |
| `payload` | `dict[str, Any]` | Job-specific data |
| `status` | `JobStatus` | Current status (queued, processing, completed, failed) |
| `created_at` | `datetime` | When job was created |
| `created_by` | `str` | User/system that created the job |
| `scheduled_at` | `Optional[datetime]` | When to execute (for scheduled jobs) |
| `started_at` | `Optional[datetime]` | When processing started |
| `completed_at` | `Optional[datetime]` | When processing completed |
| `result` | `Optional[dict[str, Any]]` | Job result data |
| `error` | `Optional[str]` | Error message if failed |
| `retry_count` | `int` | Number of retry attempts |
| `max_retries` | `int` | Maximum allowed retries |

### JobResult

Represents the outcome of job execution.

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `str` | Job identifier |
| `status` | `str` | Result status (completed, failed) |
| `result` | `Optional[dict[str, Any]]` | Result data on success |
| `error` | `Optional[str]` | Error message on failure |

### QueueStats

Queue metrics for monitoring.

| Field | Type | Description |
|-------|------|-------------|
| `queued` | `int` | Number of jobs waiting |
| `processing` | `int` | Number of jobs being processed |
| `completed` | `int` | Number of completed jobs |
| `failed` | `int` | Number of failed jobs |

### JobStatus Enum

| Value | Description |
|-------|-------------|
| `QUEUED` | Job is waiting in queue |
| `PROCESSING` | Job is being processed by a worker |
| `COMPLETED` | Job completed successfully |
| `FAILED` | Job failed after max retries |
| `CANCELLED` | Job was cancelled |

## Best Practices

### 1. Keep Handlers Stateless

Design job handlers to be stateless and idempotent when possible.

```python
# GOOD - Stateless handler with injected dependencies
class ProcessOrderHandler(JobHandler):
    def __init__(self, database, payment_gateway):
        self.database = database
        self.payment_gateway = payment_gateway

    async def handle(self, job: Job) -> JobResult:
        order_id = job.payload["order_id"]

        # Idempotent: Check if already processed
        order = await self.database.get_order(order_id)
        if order.status == "processed":
            return JobResult(job_id=job.id, status="completed",
                           result={"already_processed": True})

        # Process order
        await self.payment_gateway.charge(order)
        await self.database.update_order_status(order_id, "processed")

        return JobResult(job_id=job.id, status="completed")

# BAD - Stateful handler with instance variables
class ProcessOrderHandler(JobHandler):
    def __init__(self):
        self.processed_orders = []  # BAD: Shared mutable state

    async def handle(self, job: Job) -> JobResult:
        order_id = job.payload["order_id"]
        self.processed_orders.append(order_id)  # BAD: Not thread-safe
        # ...
```

**Why**: Stateless handlers are easier to test, scale, and reason about. Multiple workers can process jobs concurrently without race conditions.

### 2. Use Explicit Error Handling

Always handle expected errors explicitly and use `on_failure()` for cleanup.

```python
# GOOD - Explicit error handling
class ImportDataHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        try:
            data = await self.fetch_data(job.payload["url"])
            await self.validate_data(data)
            await self.import_data(data)

            return JobResult(job_id=job.id, status="completed",
                           result={"records_imported": len(data)})

        except ValidationError as e:
            # Don't retry validation errors
            return JobResult(job_id=job.id, status="failed",
                           error=f"Invalid data: {e}")

        except NetworkError as e:
            # Re-raise for retry
            raise

    async def on_failure(self, job: Job, error: Exception) -> None:
        # Clean up partial imports
        await self.rollback_import(job.id)

# BAD - Generic error handling
class ImportDataHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        try:
            # ... processing ...
            pass
        except Exception as e:  # BAD: Catches everything
            return JobResult(job_id=job.id, status="failed", error=str(e))
```

**Why**: Explicit error handling allows you to distinguish between retryable errors (network issues) and permanent failures (validation errors).

### 3. Set Appropriate Retry Limits

Configure retry limits based on job characteristics.

```python
# GOOD - Different retry strategies for different job types
handlers = {
    # Network operations: More retries
    "fetch_external_api": FetchApiHandler(max_retries=5),

    # Database operations: Fewer retries
    "batch_insert": BatchInsertHandler(max_retries=2),

    # User-triggered operations: No retries
    "user_export": ExportHandler(max_retries=0),
}

# Configure in WorkerManager
worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,
    max_retries=3,  # Default for handlers without explicit limit
    base_retry_delay=2.0
)

# BAD - Same retry strategy for all jobs
worker_manager = WorkerManager(
    queue_adapter=queue_adapter,
    handlers=handlers,
    max_retries=10,  # BAD: Too many retries for permanent failures
    base_retry_delay=1.0
)
```

**Why**: Different job types have different failure modes. Network errors benefit from retries, but validation errors don't.

### 4. Use Structured Payloads

Use structured, versioned payloads with clear schemas.

```python
# GOOD - Structured payload with version
from pydantic import BaseModel

class SendEmailPayload(BaseModel):
    version: int = 1
    email: str
    subject: str
    template: str
    variables: dict[str, Any]

class SendEmailHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        # Validate and parse payload
        try:
            payload = SendEmailPayload(**job.payload)
        except ValidationError as e:
            return JobResult(job_id=job.id, status="failed",
                           error=f"Invalid payload: {e}")

        # Use typed payload
        await send_email(
            to=payload.email,
            subject=payload.subject,
            template=payload.template,
            variables=payload.variables
        )

        return JobResult(job_id=job.id, status="completed")

# BAD - Unstructured payload
class SendEmailHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        # BAD: Direct dict access without validation
        email = job.payload["email"]  # May raise KeyError
        subject = job.payload.get("subject", "No Subject")  # Inconsistent
        # ...
```

**Why**: Structured payloads catch errors early, provide clear contracts, and enable payload versioning for backwards compatibility.

### 5. Monitor Job Metrics

Implement monitoring and alerting for job health.

```python
# GOOD - Expose metrics for monitoring
class MetricsCollector:
    def __init__(self):
        self.jobs_created = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.processing_times = []

    async def record_job_completed(self, job: Job, duration_ms: float):
        self.jobs_completed += 1
        self.processing_times.append(duration_ms)

    async def get_metrics(self) -> dict:
        return {
            "jobs_created": self.jobs_created,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0,
            "success_rate": self.jobs_completed / (self.jobs_completed + self.jobs_failed)
                if (self.jobs_completed + self.jobs_failed) > 0 else 1.0
        }

# Integrate with job service
class MonitoredJobService(JobService):
    def __init__(self, queue_adapter, audit_service, metrics_collector):
        super().__init__(queue_adapter, audit_service)
        self.metrics = metrics_collector

    async def create_job(self, *args, **kwargs):
        job = await super().create_job(*args, **kwargs)
        self.metrics.jobs_created += 1
        return job
```

**Why**: Production job systems need observability to detect issues like stuck queues, high failure rates, or performance degradation.

### 6. Implement Graceful Shutdown

Always stop workers gracefully to avoid losing in-flight jobs.

```python
# GOOD - Graceful shutdown with signal handling
import signal
import asyncio

class Application:
    def __init__(self, worker_manager):
        self.worker_manager = worker_manager
        self.shutdown_event = asyncio.Event()

    async def start(self):
        # Start workers
        await self.worker_manager.start()

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown_event.set)

        # Wait for shutdown signal
        await self.shutdown_event.wait()

        # Graceful shutdown
        print("Shutting down workers...")
        await self.worker_manager.stop()  # Waits for in-flight jobs
        print("Workers stopped")

# BAD - Abrupt shutdown
async def main():
    worker_manager = WorkerManager(...)
    await worker_manager.start()

    # BAD: No graceful shutdown
    # Workers may be killed mid-job

if __name__ == "__main__":
    asyncio.run(main())
```

**Why**: Graceful shutdown ensures in-flight jobs complete or are properly returned to the queue, preventing data loss.

### 7. Use Dead Letter Queues

Implement dead letter queues for jobs that exceed max retries.

```python
# GOOD - Dead letter queue for failed jobs
class DeadLetterHandler:
    def __init__(self, database, alerting_service):
        self.database = database
        self.alerting_service = alerting_service

    async def handle_dead_letter(self, job: Job):
        # Store failed job for investigation
        await self.database.store_dead_letter_job(job)

        # Alert engineers
        await self.alerting_service.send_alert(
            severity="high",
            message=f"Job {job.id} failed after {job.max_retries} retries",
            context={"job_type": job.job_type, "payload": job.payload}
        )

class CustomWorkerManager(WorkerManager):
    def __init__(self, *args, dead_letter_handler, **kwargs):
        super().__init__(*args, **kwargs)
        self.dead_letter_handler = dead_letter_handler

    async def process_job(self, job: Job):
        try:
            result = await super().process_job(job)
            return result
        except MaxRetriesExceeded:
            # Move to dead letter queue
            await self.dead_letter_handler.handle_dead_letter(job)
            raise
```

**Why**: Dead letter queues prevent permanently failed jobs from blocking the queue and provide a mechanism for investigating failures.

## Security Considerations

### 1. Payload Validation

Always validate job payloads to prevent injection attacks and malformed data.

```python
from pydantic import BaseModel, validator

class JobPayloadBase(BaseModel):
    class Config:
        max_anystr_length = 10000  # Limit string sizes
        validate_assignment = True

class ProcessFilePayload(JobPayloadBase):
    file_id: str
    user_id: str

    @validator('file_id')
    def validate_file_id(cls, v):
        if not v.startswith('file_'):
            raise ValueError('Invalid file_id format')
        return v
```

### 2. Access Control

Verify that job creators have permission to create jobs and access job results.

```python
class SecureJobService(JobService):
    def __init__(self, queue_adapter, audit_service, permissions_service):
        super().__init__(queue_adapter, audit_service)
        self.permissions = permissions_service

    async def create_job(self, job_type: str, payload: dict, created_by: str, **kwargs):
        # Check permission
        if not await self.permissions.can_create_job(created_by, job_type):
            raise PermissionDenied(f"User {created_by} cannot create {job_type} jobs")

        return await super().create_job(job_type, payload, created_by, **kwargs)

    async def get_job(self, job_id: str, requested_by: str) -> Job:
        job = await super().get_job(job_id)

        # Only job creator or admin can view
        if job.created_by != requested_by and not await self.permissions.is_admin(requested_by):
            raise PermissionDenied("Cannot view this job")

        return job
```

### 3. Secrets Management

Never store secrets in job payloads. Use secret references instead.

```python
# GOOD - Secret references
job = await job_service.create_job(
    job_type="send_email",
    payload={
        "email": "user@example.com",
        "smtp_config_id": "smtp_config_prod"  # Reference to secret store
    },
    created_by="system"
)

class SendEmailHandler(JobHandler):
    def __init__(self, secret_store):
        self.secret_store = secret_store

    async def handle(self, job: Job) -> JobResult:
        config_id = job.payload["smtp_config_id"]
        smtp_config = await self.secret_store.get_secret(config_id)

        # Use secret
        await send_email(smtp_config=smtp_config, ...)

# BAD - Secrets in payload
job = await job_service.create_job(
    job_type="send_email",
    payload={
        "email": "user@example.com",
        "smtp_password": "hunter2"  # BAD: Secret exposed in logs/database
    },
    created_by="system"
)
```

### 4. Resource Limits

Implement timeouts and resource limits to prevent resource exhaustion.

```python
class ResourceLimitedHandler(JobHandler):
    def __init__(self, max_execution_seconds: int = 300):
        self.max_execution_seconds = max_execution_seconds

    async def handle(self, job: Job) -> JobResult:
        try:
            # Set timeout
            async with asyncio.timeout(self.max_execution_seconds):
                result = await self.process_job(job)
                return result

        except asyncio.TimeoutError:
            return JobResult(
                job_id=job.id,
                status="failed",
                error=f"Job exceeded {self.max_execution_seconds}s timeout"
            )
```

## FAQs

### 1. How do I choose between JobService and direct queue access?

**Use JobService** for standard job creation with audit logging and consistent job ID generation. JobService implements the `JobCreator` interface and integrates with the audit system.

**Use direct queue access** only for low-level queue operations or custom job enqueueing logic.

```python
# Standard approach (recommended)
job = await job_service.create_job("process_data", {"file_id": "123"}, "user_456")

# Low-level approach (advanced use cases)
await queue_adapter.enqueue(job)  # Manual job object creation required
```

### 2. How do I handle jobs that depend on other jobs?

Implement job chaining in your handler's `handle()` method:

```python
class ProcessAndNotifyHandler(JobHandler):
    def __init__(self, job_service):
        self.job_service = job_service

    async def handle(self, job: Job) -> JobResult:
        # Process data
        result = await process_data(job.payload["data_id"])

        # Chain next job
        await self.job_service.create_job(
            job_type="send_notification",
            payload={"result_id": result.id, "user_id": job.payload["user_id"]},
            created_by=job.created_by
        )

        return JobResult(job_id=job.id, status="completed", result={"result_id": result.id})
```

For complex workflows, consider using a workflow orchestration library like Temporal or Airflow.

### 3. How do I test job handlers?

Test handlers independently of the queue infrastructure:

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_send_email_handler():
    # Arrange
    handler = SendEmailHandler()
    handler.email_service = AsyncMock()

    job = Job(
        id="job_123",
        job_type="send_email",
        payload={"email": "test@example.com", "subject": "Test"},
        status="queued",
        created_at=datetime.utcnow(),
        created_by="test_user",
        retry_count=0,
        max_retries=3
    )

    # Act
    result = await handler.handle(job)

    # Assert
    assert result.status == "completed"
    handler.email_service.send.assert_called_once_with(
        to="test@example.com",
        subject="Test"
    )
```

### 4. How do I scale workers horizontally?

Deploy multiple instances of your application with WorkerManager. Each instance will consume from the same queue:

```python
# Instance 1
worker_manager_1 = WorkerManager(
    queue_adapter=redis_queue,  # Shared queue
    handlers=handlers,
    num_workers=4
)

# Instance 2 (separate process/container)
worker_manager_2 = WorkerManager(
    queue_adapter=redis_queue,  # Same queue
    handlers=handlers,
    num_workers=4
)
```

Ensure your queue adapter supports concurrent consumption (Redis, RabbitMQ, SQS do).

### 5. How do I implement priority queues?

Use multiple queues with different worker managers:

```python
# High-priority queue
high_priority_queue = RedisJobQueue(redis_url="...", queue_name="jobs:high")
high_priority_workers = WorkerManager(
    queue_adapter=high_priority_queue,
    handlers=handlers,
    num_workers=8  # More workers for high-priority
)

# Low-priority queue
low_priority_queue = RedisJobQueue(redis_url="...", queue_name="jobs:low")
low_priority_workers = WorkerManager(
    queue_adapter=low_priority_queue,
    handlers=handlers,
    num_workers=2  # Fewer workers for low-priority
)

# Enqueue to appropriate queue based on priority
if priority == "high":
    await job_service_high.create_job(...)
else:
    await job_service_low.create_job(...)
```

### 6. How do I handle jobs that need to run at specific times?

Use the `scheduled_at` parameter when creating jobs:

```python
from datetime import datetime, timedelta

# Schedule job for tomorrow at 9 AM
scheduled_time = datetime.utcnow().replace(hour=9, minute=0, second=0) + timedelta(days=1)

job = await job_service.create_job(
    job_type="daily_report",
    payload={"date": "2024-01-15"},
    created_by="system",
    scheduled_at=scheduled_time
)
```

Workers will process jobs when `job.scheduled_at <= datetime.utcnow()`.

### 7. How do I handle long-running jobs that exceed worker timeout?

Break long-running jobs into smaller chunks with continuation:

```python
class ProcessLargeDatasetHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        dataset_id = job.payload["dataset_id"]
        offset = job.payload.get("offset", 0)
        chunk_size = 1000

        # Process one chunk
        records = await fetch_records(dataset_id, offset, chunk_size)
        await process_records(records)

        # If more records, enqueue continuation
        if len(records) == chunk_size:
            await self.job_service.create_job(
                job_type="process_large_dataset",
                payload={"dataset_id": dataset_id, "offset": offset + chunk_size},
                created_by=job.created_by
            )

        return JobResult(
            job_id=job.id,
            status="completed",
            result={"records_processed": len(records), "offset": offset}
        )
```

### 8. How do I debug failed jobs?

Enable comprehensive logging and store job results:

```python
import logging

logger = logging.getLogger(__name__)

class DebugJobHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        logger.info(f"Starting job {job.id} of type {job.job_type}")
        logger.debug(f"Job payload: {job.payload}")

        try:
            result = await self.process(job)
            logger.info(f"Job {job.id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            raise

    async def on_failure(self, job: Job, error: Exception) -> None:
        # Store failure details
        await self.database.store_job_failure(
            job_id=job.id,
            error=str(error),
            traceback=traceback.format_exc(),
            retry_count=job.retry_count
        )
```

Access failed job details through your database or dead letter queue for investigation.

### 9. How do I prevent duplicate job execution?

Implement idempotency in your handler:

```python
class IdempotentHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        # Check if already processed
        existing_result = await self.database.get_job_result(job.id)
        if existing_result:
            logger.info(f"Job {job.id} already processed, skipping")
            return existing_result

        # Process job
        result = await self.process(job)

        # Store result atomically
        await self.database.store_job_result(job.id, result)

        return result
```

Alternatively, use unique job IDs based on payload content:

```python
import hashlib
import json

def create_idempotent_job_id(job_type: str, payload: dict) -> str:
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{job_type}_{payload_hash}"
```

### 10. How do I monitor job queue health?

Expose queue metrics through an API endpoint:

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/admin/job-stats")
async def get_job_stats(job_service: JobService):
    stats = await job_service.get_job_stats()

    return {
        "queued": stats.queued,
        "processing": stats.processing,
        "completed": stats.completed,
        "failed": stats.failed,
        "queue_depth": stats.queued + stats.processing,
        "failure_rate": stats.failed / (stats.completed + stats.failed)
            if (stats.completed + stats.failed) > 0 else 0
    }
```

Set up alerts for abnormal metrics (high queue depth, high failure rate).

## Related Components

- **Audit Port** - Job lifecycle events are logged through the audit system
- **Database Port** - Job handlers often interact with databases
- **Notification Port** - Common use case for background job processing
- **File Storage Port** - Jobs often process uploaded files

## Architecture Notes

The Job Kit represents a **service/utility layer** rather than a traditional Kit with a compose factory. It provides two main components:

1. **JobService** - Implements the `JobCreator` interface for creating and managing jobs
2. **WorkerManager** - Orchestrates worker processes that execute jobs

This design allows applications to integrate background job processing without requiring a full Kit composition. The `JobQueueAdapter` port enables pluggable queue backends (in-memory for testing, Redis/RabbitMQ/SQS for production).

**Key Architectural Decisions**:

- **Handler-based processing**: Job business logic is encapsulated in `JobHandler` implementations, promoting separation of concerns
- **Port-based queue abstraction**: Queue operations are defined by the `JobQueueAdapter` interface, enabling different queue technologies
- **Automatic retry with backoff**: Built-in retry logic reduces boilerplate in handlers
- **Audit integration**: Job lifecycle events are automatically published for compliance and monitoring
- **Graceful shutdown**: Workers complete in-flight jobs before stopping, preventing data loss

The Job Kit follows hexagonal architecture principles by depending on ports (interfaces) rather than concrete implementations, allowing for flexible deployment configurations.
