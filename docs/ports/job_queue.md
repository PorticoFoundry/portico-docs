# Job Queue Port

## Overview

The Job Queue Port defines the contract for queue-based job processing systems. This is the adapter interface that different queue implementations (memory, database, Redis, etc.) must implement.

**Purpose**: Enable pluggable queue backends for background job processing while maintaining a consistent interface for job enqueueing, dequeueing, acknowledgment, and failure handling.

**Domain**: Background job processing, asynchronous task queues, and worker orchestration

**Key Capabilities**:

- Enqueue jobs with priority ordering
- Dequeue jobs with blocking and timeout support
- Acknowledge successful job completion
- Reject jobs with retry logic and error tracking
- Schedule jobs for future execution
- Cancel pending or scheduled jobs
- Track queue statistics (pending, running, completed, failed, dead letter)
- Support multiple independent queues

**Port Type**: Adapter

**When to Use**:

- When implementing queue backends (in-memory, database-backed, Redis, RabbitMQ, etc.)
- When building job processing systems with worker pools
- When you need reliable job execution with retry logic
- When integrating with existing queue infrastructure

## Domain Models

The Job Queue Port uses domain models from `portico.ports.job`:

- **Job** - Represents a background job with type, payload, status, retry configuration, timestamps
- **JobStatus** - Enumeration of job statuses (PENDING, SCHEDULED, RUNNING, COMPLETED, FAILED, RETRYING, DEAD_LETTER)
- **QueueStats** - Statistics about a queue (pending_count, running_count, completed_count, failed_count, dead_letter_count)

### QueueStats

Statistics for a job queue, used for monitoring and observability.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `queue_name` | `str` | Yes | - | Name of the queue |
| `pending_count` | `int` | Yes | - | Number of jobs waiting to be processed |
| `running_count` | `int` | Yes | - | Number of jobs currently being processed |
| `completed_count` | `int` | Yes | - | Number of successfully completed jobs |
| `failed_count` | `int` | Yes | - | Number of failed jobs (not dead letter) |
| `dead_letter_count` | `int` | Yes | - | Number of jobs that exhausted retries |

**Example**:

```python
from portico.ports.job import QueueStats

stats = QueueStats(
    queue_name="email_queue",
    pending_count=42,
    running_count=5,
    completed_count=1250,
    failed_count=3,
    dead_letter_count=1
)
```

## Port Interface

### JobQueueAdapter

The `JobQueueAdapter` abstract base class defines the contract for queue implementations.

**Location**: `portico.ports.job_queue.JobQueueAdapter`

#### Methods

##### enqueue

```python
@abstractmethod
async def enqueue(self, job: Job) -> Job
```

Add job to queue.

**Parameters**:

- `job` (`Job`): Job to enqueue with type, payload, priority, scheduling information

**Returns**: `Job` - The enqueued job (may have updated fields like `created_at`, `status`)

**Purpose**: Add a job to the queue for processing. Sets initial status (PENDING or SCHEDULED) and timestamp.

**Example**:

```python
from uuid import uuid4
from portico.ports.job import Job

job = Job(
    id=uuid4(),
    queue_name="email_queue",
    job_type="email.send",
    payload={"recipient": "user@example.com", "subject": "Welcome"},
    priority=5,
    max_retries=3
)

enqueued_job = await job_queue.enqueue(job)
assert enqueued_job.status == JobStatus.PENDING
assert enqueued_job.created_at is not None
```

##### dequeue

```python
@abstractmethod
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None
```

Get next job from queue (blocking with timeout).

**Parameters**:

- `queue_name` (`str`): Name of queue to dequeue from
- `timeout` (`int`): Maximum seconds to wait for a job (default: 30)

**Returns**: `Job | None` - Next available job, or None if timeout reached

**Purpose**: Workers call this method to get the next job to process. Blocks until a job is available or timeout is reached. Updates job status to RUNNING and sets `started_at` timestamp.

**Important Notes**:

- Jobs are dequeued in priority order (higher priority first)
- Scheduled jobs are only returned when `scheduled_at` time has passed
- Returns None after timeout if no jobs available
- Atomically marks job as RUNNING to prevent duplicate processing

**Example**:

```python
# Worker loop
while True:
    job = await job_queue.dequeue("email_queue", timeout=30)

    if job is None:
        continue  # Timeout, try again

    # Process job
    result = await process_job(job)

    if result.success:
        await job_queue.acknowledge(job.id)
    else:
        await job_queue.reject(job.id, requeue=True)
```

##### acknowledge

```python
@abstractmethod
async def acknowledge(self, job_id: UUID) -> None
```

Mark job as successfully completed.

**Parameters**:

- `job_id` (`UUID`): ID of job to acknowledge

**Returns**: None

**Purpose**: Called by workers when a job completes successfully. Updates status to COMPLETED and sets `completed_at` timestamp.

**Example**:

```python
# Process job
result = await handler.handle(job)

if result.success:
    await job_queue.acknowledge(job.id)
```

##### reject

```python
@abstractmethod
async def reject(
    self,
    job_id: UUID,
    requeue: bool = True,
    error_message: str | None = None
) -> None
```

Mark job as failed, optionally requeue for retry.

**Parameters**:

- `job_id` (`UUID`): ID of job to reject
- `requeue` (`bool`): Whether to requeue job for retry (default: True)
- `error_message` (`str | None`): Error message to store with the job

**Returns**: None

**Purpose**: Called by workers when a job fails. Implements retry logic based on `max_retries` and `retry_count`.

**Behavior**:

- If `requeue=True` and `retry_count < max_retries`: Increments `retry_count`, sets status to RETRYING, requeues job
- If `requeue=True` and `retry_count >= max_retries`: Sets status to DEAD_LETTER, sets `failed_at`
- If `requeue=False`: Sets status to FAILED, sets `failed_at`
- Stores `error_message` in job for debugging

**Example**:

```python
try:
    result = await handler.handle(job)

    if result.success:
        await job_queue.acknowledge(job.id)
    else:
        # Retry transient errors
        await job_queue.reject(
            job.id,
            requeue=True,
            error_message=str(result.error)
        )
except ValidationError as e:
    # Don't retry validation errors
    await job_queue.reject(
        job.id,
        requeue=False,
        error_message=f"Validation failed: {e}"
    )
```

##### get_job

```python
@abstractmethod
async def get_job(self, job_id: UUID) -> Job | None
```

Get job by ID.

**Parameters**:

- `job_id` (`UUID`): ID of job to retrieve

**Returns**: `Job | None` - Job if found, None otherwise

**Purpose**: Retrieve job information for monitoring, debugging, or status checks.

**Example**:

```python
job = await job_queue.get_job(job_id)

if job:
    print(f"Status: {job.status}")
    print(f"Retry count: {job.retry_count}/{job.max_retries}")
    if job.error_message:
        print(f"Error: {job.error_message}")
```

##### cancel_job

```python
@abstractmethod
async def cancel_job(self, job_id: UUID) -> bool
```

Cancel a pending or scheduled job.

**Parameters**:

- `job_id` (`UUID`): ID of job to cancel

**Returns**: `bool` - True if job was cancelled, False if not found or already running

**Purpose**: Cancel jobs that haven't started processing yet. Cannot cancel jobs that are already running.

**Example**:

```python
# User cancels report generation
cancelled = await job_queue.cancel_job(report_job_id)

if cancelled:
    print("Report generation cancelled")
else:
    print("Job already running, cannot cancel")
```

##### get_queue_stats

```python
@abstractmethod
async def get_queue_stats(self, queue_name: str) -> QueueStats
```

Get statistics for a queue.

**Parameters**:

- `queue_name` (`str`): Name of queue

**Returns**: `QueueStats` - Statistics including counts for pending, running, completed, failed, and dead letter jobs

**Purpose**: Monitor queue health, track processing metrics, and identify bottlenecks.

**Example**:

```python
stats = await job_queue.get_queue_stats("email_queue")

print(f"Queue: {stats.queue_name}")
print(f"Pending: {stats.pending_count}")
print(f"Running: {stats.running_count}")
print(f"Completed: {stats.completed_count}")
print(f"Failed: {stats.failed_count}")
print(f"Dead letter: {stats.dead_letter_count}")

# Alert if queue backing up
if stats.pending_count > 1000:
    await alert_service.send_alert("Queue backlog detected")
```

##### initialize

```python
@abstractmethod
async def initialize(self) -> None
```

Initialize the queue adapter (create tables, connections, etc.).

**Returns**: None

**Purpose**: Perform one-time setup like creating database tables, establishing connections, or initializing resources.

**Example**:

```python
# Application startup
job_queue = DatabaseJobQueueAdapter(database=database)
await job_queue.initialize()  # Creates job tables
```

##### close

```python
@abstractmethod
async def close(self) -> None
```

Close the queue adapter and cleanup resources.

**Returns**: None

**Purpose**: Cleanup connections, close database sessions, release resources on shutdown.

**Example**:

```python
# Application shutdown
await job_queue.close()
```

## Common Patterns

### 1. Worker Processing Loop

```python
from portico.ports.job_queue import JobQueueAdapter
from portico.utils.logging import get_logger

logger = get_logger(__name__)

async def worker_loop(
    job_queue: JobQueueAdapter,
    handler: JobHandler,
    queue_name: str = "default"
) -> None:
    """Basic worker processing loop."""
    logger.info("worker_started", queue_name=queue_name)

    while True:
        try:
            # Get next job (blocks for up to 30 seconds)
            job = await job_queue.dequeue(queue_name, timeout=30)

            if job is None:
                continue  # Timeout, try again

            logger.info(
                "processing_job",
                job_id=str(job.id),
                job_type=job.job_type
            )

            # Process job
            result = await handler.handle(job)

            if result.success:
                logger.info("job_completed", job_id=str(job.id))
                await job_queue.acknowledge(job.id)
            else:
                logger.warning(
                    "job_failed",
                    job_id=str(job.id),
                    error=str(result.error)
                )
                await job_queue.reject(
                    job.id,
                    requeue=True,
                    error_message=str(result.error)
                )

        except Exception as e:
            logger.error("worker_error", error=str(e), exc_info=True)
            await asyncio.sleep(1)  # Back off on error
```

### 2. Priority Queue Usage

```python
from uuid import uuid4
from portico.ports.job import Job

# High priority job (urgent)
urgent_job = Job(
    id=uuid4(),
    queue_name="notifications",
    job_type="sms.send",
    payload={"phone": "+1234567890", "message": "Alert!"},
    priority=10,  # High priority
    max_retries=5
)

# Normal priority job
normal_job = Job(
    id=uuid4(),
    queue_name="notifications",
    job_type="email.send",
    payload={"recipient": "user@example.com"},
    priority=5,  # Medium priority
    max_retries=3
)

# Low priority job (background task)
background_job = Job(
    id=uuid4(),
    queue_name="notifications",
    job_type="report.generate",
    payload={"report_id": "123"},
    priority=1,  # Low priority
    max_retries=3
)

# Enqueue all jobs
await job_queue.enqueue(normal_job)
await job_queue.enqueue(background_job)
await job_queue.enqueue(urgent_job)

# Worker will dequeue in priority order: urgent (10) -> normal (5) -> background (1)
job1 = await job_queue.dequeue("notifications")  # Gets urgent_job
job2 = await job_queue.dequeue("notifications")  # Gets normal_job
job3 = await job_queue.dequeue("notifications")  # Gets background_job
```

### 3. Scheduled Jobs

```python
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from portico.ports.job import Job

# Schedule job for 1 hour from now
scheduled_time = datetime.now(UTC) + timedelta(hours=1)

scheduled_job = Job(
    id=uuid4(),
    queue_name="reminders",
    job_type="reminder.send",
    payload={"user_id": "123", "message": "Meeting in 1 hour"},
    scheduled_at=scheduled_time
)

enqueued_job = await job_queue.enqueue(scheduled_job)
assert enqueued_job.status == JobStatus.SCHEDULED

# Worker won't dequeue this job until scheduled_at time passes
job = await job_queue.dequeue("reminders", timeout=5)
assert job is None  # Too early, returns None

# After scheduled time passes, job becomes available
await asyncio.sleep(3600)  # Wait 1 hour
job = await job_queue.dequeue("reminders", timeout=5)
assert job is not None  # Now available
```

### 4. Retry Logic with Error Tracking

```python
from uuid import uuid4
from portico.ports.job import Job

job = Job(
    id=uuid4(),
    queue_name="data_import",
    job_type="import.csv",
    payload={"url": "https://example.com/data.csv"},
    max_retries=3
)

await job_queue.enqueue(job)

# First attempt - fails
job1 = await job_queue.dequeue("data_import")
await job_queue.reject(
    job1.id,
    requeue=True,
    error_message="Connection timeout"
)

# Check job state
retrying_job = await job_queue.get_job(job.id)
assert retrying_job.status == JobStatus.RETRYING
assert retrying_job.retry_count == 1
assert retrying_job.error_message == "Connection timeout"

# Second attempt - fails
job2 = await job_queue.dequeue("data_import")
await job_queue.reject(
    job2.id,
    requeue=True,
    error_message="Invalid CSV format"
)

# Third attempt - fails
job3 = await job_queue.dequeue("data_import")
await job_queue.reject(
    job3.id,
    requeue=True,
    error_message="Parse error"
)

# Fourth attempt - exhausted retries
job4 = await job_queue.dequeue("data_import")
await job_queue.reject(
    job4.id,
    requeue=True,
    error_message="Still failing"
)

# Job moves to dead letter
dead_job = await job_queue.get_job(job.id)
assert dead_job.status == JobStatus.DEAD_LETTER
assert dead_job.retry_count == 3  # Max retries reached
```

### 5. Multiple Queue Management

```python
# Use different queues for different priorities or job types

# High priority queue for critical tasks
critical_job = Job(
    id=uuid4(),
    queue_name="critical",
    job_type="payment.process",
    payload={"amount": 1000},
    priority=10
)
await job_queue.enqueue(critical_job)

# Normal queue for regular tasks
normal_job = Job(
    id=uuid4(),
    queue_name="default",
    job_type="email.send",
    payload={"recipient": "user@example.com"},
    priority=5
)
await job_queue.enqueue(normal_job)

# Background queue for low priority tasks
background_job = Job(
    id=uuid4(),
    queue_name="background",
    job_type="cleanup.old_files",
    payload={"days": 30},
    priority=1
)
await job_queue.enqueue(background_job)

# Workers can process queues in order
async def multi_queue_worker():
    while True:
        # Check critical queue first
        job = await job_queue.dequeue("critical", timeout=1)
        if job:
            await process_job(job)
            continue

        # Then default queue
        job = await job_queue.dequeue("default", timeout=1)
        if job:
            await process_job(job)
            continue

        # Finally background queue
        job = await job_queue.dequeue("background", timeout=1)
        if job:
            await process_job(job)
```

### 6. Job Cancellation

```python
from uuid import uuid4
from portico.ports.job import Job

# Create and enqueue job
job = Job(
    id=uuid4(),
    queue_name="reports",
    job_type="report.generate",
    payload={"report_type": "monthly", "user_id": "123"}
)

await job_queue.enqueue(job)

# User changes mind, cancel the job
cancelled = await job_queue.cancel_job(job.id)

if cancelled:
    print("Report generation cancelled successfully")
else:
    print("Cannot cancel - job already started processing")

# Check job state
cancelled_job = await job_queue.get_job(job.id)
if cancelled_job.status == JobStatus.FAILED:
    assert cancelled_job.error_message == "Job cancelled"
```

### 7. Queue Monitoring and Alerting

```python
from portico.ports.job_queue import JobQueueAdapter

async def monitor_queues(job_queue: JobQueueAdapter):
    """Monitor queue health and send alerts."""
    queues_to_monitor = ["critical", "default", "background"]

    for queue_name in queues_to_monitor:
        stats = await job_queue.get_queue_stats(queue_name)

        # Alert on queue backlog
        if stats.pending_count > 100:
            await send_alert(
                level="warning",
                message=f"Queue {queue_name} has {stats.pending_count} pending jobs"
            )

        # Alert on high failure rate
        total_jobs = stats.completed_count + stats.failed_count
        if total_jobs > 0:
            failure_rate = stats.failed_count / total_jobs
            if failure_rate > 0.1:  # 10% failure rate
                await send_alert(
                    level="error",
                    message=f"Queue {queue_name} has {failure_rate:.1%} failure rate"
                )

        # Alert on dead letter jobs
        if stats.dead_letter_count > 0:
            await send_alert(
                level="critical",
                message=f"Queue {queue_name} has {stats.dead_letter_count} dead letter jobs"
            )

        # Log metrics
        logger.info(
            "queue_stats",
            queue=queue_name,
            pending=stats.pending_count,
            running=stats.running_count,
            completed=stats.completed_count,
            failed=stats.failed_count,
            dead_letter=stats.dead_letter_count
        )
```

## Best Practices

### 1. ✅ Implement Atomic Dequeue Operations

Ensure dequeue operations are atomic to prevent duplicate processing.

```python
# ✅ GOOD - Atomic dequeue with database row locking
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    async with self.database.session() as session:
        # Use SELECT FOR UPDATE to lock the row
        stmt = (
            select(JobModel)
            .where(
                JobModel.queue_name == queue_name,
                JobModel.status == JobStatus.PENDING,
            )
            .order_by(JobModel.priority.desc(), JobModel.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)  # Skip locked rows
        )

        result = await session.execute(stmt)
        job_model = result.scalar_one_or_none()

        if job_model:
            job_model.status = JobStatus.RUNNING
            job_model.started_at = datetime.now(UTC)
            await session.commit()
            return job_model.to_domain()

        return None

# ❌ BAD - Race condition, multiple workers could dequeue same job
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    # Get job
    job = await self.get_next_pending_job(queue_name)

    if job:
        # Race condition here! Another worker might get same job
        await self.mark_as_running(job.id)
        return job
```

### 2. ✅ Respect Priority Ordering

Always dequeue jobs in priority order (higher priority first).

```python
# ✅ GOOD - Dequeue by priority, then FIFO
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    async with self._lock:
        queue = self._queues.get(queue_name, [])

        # Sort by priority (descending), then created_at (ascending)
        ready_jobs = [
            j for j in queue
            if j.status == JobStatus.PENDING
            and (not j.scheduled_at or j.scheduled_at <= datetime.now(UTC))
        ]

        ready_jobs.sort(key=lambda j: (-j.priority, j.created_at))

        if ready_jobs:
            job = ready_jobs[0]
            # Mark as running...
            return job

# ❌ BAD - FIFO only, ignores priority
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    queue = self._queues.get(queue_name, [])

    if queue:
        return queue.pop(0)  # ❌ Ignores priority
```

### 3. ✅ Handle Scheduled Jobs Correctly

Don't dequeue jobs before their scheduled time.

```python
# ✅ GOOD - Check scheduled_at before dequeueing
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    for job in self._queues.get(queue_name, []):
        # Skip jobs scheduled for future
        if job.scheduled_at and job.scheduled_at > datetime.now(UTC):
            continue

        # This job is ready
        return self._mark_running(job)

# ❌ BAD - Returns scheduled jobs too early
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    queue = self._queues.get(queue_name, [])

    if queue:
        return queue.pop(0)  # ❌ Might return scheduled job
```

### 4. ✅ Implement Proper Retry Logic

Move jobs to dead letter queue after max retries.

```python
# ✅ GOOD - Proper retry and dead letter handling
async def reject(
    self,
    job_id: UUID,
    requeue: bool = True,
    error_message: str | None = None
) -> None:
    job = await self.get_job(job_id)
    if not job:
        return

    if requeue and job.retry_count < job.max_retries:
        # Requeue for retry
        updated_job = Job(
            **{**job.__dict__,
                "status": JobStatus.RETRYING,
                "retry_count": job.retry_count + 1,
                "error_message": error_message
            }
        )
        await self._requeue(updated_job)
    else:
        # Move to dead letter
        updated_job = Job(
            **{**job.__dict__,
                "status": JobStatus.DEAD_LETTER,
                "failed_at": datetime.now(UTC),
                "error_message": error_message
            }
        )
        await self._update_job(updated_job)

# ❌ BAD - No dead letter, retries forever
async def reject(self, job_id: UUID, requeue: bool = True, error_message: str | None = None) -> None:
    if requeue:
        job = await self.get_job(job_id)
        await self.enqueue(job)  # ❌ Infinite retries
```

### 5. ✅ Use Timeout for Blocking Dequeue

Support timeout to prevent workers from blocking indefinitely.

```python
# ✅ GOOD - Timeout support with asyncio
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    start_time = datetime.now(UTC)

    while (datetime.now(UTC) - start_time).seconds < timeout:
        async with self._lock:
            job = self._try_dequeue(queue_name)
            if job:
                return job

        # Wait for new job notification or timeout
        try:
            await asyncio.wait_for(
                self._job_available[queue_name].wait(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            pass

    return None  # Timeout reached

# ❌ BAD - Blocks forever, no timeout
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    while True:  # ❌ Infinite loop
        job = self._try_dequeue(queue_name)
        if job:
            return job
        await asyncio.sleep(0.1)
```

### 6. ✅ Track Job Timestamps

Update timestamps for observability and debugging.

```python
# ✅ GOOD - Track all relevant timestamps
async def enqueue(self, job: Job) -> Job:
    updated_job = Job(
        **{**job.__dict__,
            "created_at": datetime.now(UTC),  # When enqueued
            "status": JobStatus.PENDING
        }
    )
    await self._store(updated_job)
    return updated_job

async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    job = await self._get_next_job(queue_name)
    if job:
        updated_job = Job(
            **{**job.__dict__,
                "started_at": datetime.now(UTC),  # When started
                "status": JobStatus.RUNNING
            }
        )
        await self._update(updated_job)
        return updated_job

async def acknowledge(self, job_id: UUID) -> None:
    job = await self.get_job(job_id)
    updated_job = Job(
        **{**job.__dict__,
            "completed_at": datetime.now(UTC),  # When completed
            "status": JobStatus.COMPLETED
        }
    )
    await self._update(updated_job)

# ❌ BAD - No timestamp tracking
async def acknowledge(self, job_id: UUID) -> None:
    await self._update_status(job_id, JobStatus.COMPLETED)  # ❌ No timestamp
```

### 7. ✅ Cleanup Resources Properly

Implement initialize() and close() for resource management.

```python
# ✅ GOOD - Proper resource management
class DatabaseJobQueueAdapter(JobQueueAdapter):
    def __init__(self, database: Database):
        self.database = database
        self._initialized = False

    async def initialize(self) -> None:
        """Create job tables."""
        if not self._initialized:
            await self.database.create_tables()
            self._initialized = True

    async def close(self) -> None:
        """Close database connections."""
        await self.database.close()

# Usage
job_queue = DatabaseJobQueueAdapter(database)
try:
    await job_queue.initialize()
    # Use job queue...
finally:
    await job_queue.close()

# ❌ BAD - No resource cleanup
class DatabaseJobQueueAdapter(JobQueueAdapter):
    async def initialize(self) -> None:
        pass  # ❌ No-op

    async def close(self) -> None:
        pass  # ❌ Doesn't close connections
```

## FAQs

### How do I choose between memory, database, and Redis queue adapters?

**Memory Queue** (`MemoryJobQueueAdapter`):
- ✅ Perfect for development and testing
- ✅ No external dependencies
- ❌ Jobs lost on restart
- ❌ No persistence
- ❌ Single-process only

**Database Queue** (`DatabaseJobQueueAdapter`):
- ✅ Persistent jobs survive restarts
- ✅ Leverages existing database
- ✅ ACID guarantees
- ❌ Database becomes a bottleneck at scale
- ❌ Requires proper indexing for performance

**Redis Queue**:
- ✅ High performance
- ✅ Distributed workers
- ✅ Built-in pub/sub
- ✅ Scales horizontally
- ❌ Additional infrastructure
- ❌ Requires Redis setup

**Rule of thumb**:
- Development: Memory queue
- Small apps: Database queue
- Production/scale: Redis or dedicated queue system

### How do I prevent duplicate processing of the same job?

Use atomic operations and locking:

```python
# Database adapter - use SELECT FOR UPDATE
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    async with self.session() as session:
        stmt = (
            select(JobModel)
            .where(
                JobModel.queue_name == queue_name,
                JobModel.status == JobStatus.PENDING
            )
            .with_for_update(skip_locked=True)  # Critical!
        )

        job_model = await session.scalar(stmt)
        if job_model:
            job_model.status = JobStatus.RUNNING
            await session.commit()
            return job_model.to_domain()

# Memory adapter - use locks
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    async with self._lock:  # Atomic operation
        for job in self._queues[queue_name]:
            if job.id not in self._running:
                self._running.add(job.id)
                return job
```

### What happens to jobs that fail during processing (worker crash)?

Jobs that are RUNNING when a worker crashes need to be detected and requeued. Implement a "stale job detector":

```python
async def requeue_stale_jobs(job_queue: JobQueueAdapter, timeout_minutes: int = 10):
    """Requeue jobs that have been running too long (worker likely crashed)."""
    cutoff_time = datetime.now(UTC) - timedelta(minutes=timeout_minutes)

    # Find all running jobs started before cutoff
    stale_jobs = await get_stale_running_jobs(cutoff_time)

    for job in stale_jobs:
        logger.warning(
            "requeuing_stale_job",
            job_id=str(job.id),
            started_at=job.started_at.isoformat()
        )

        # Requeue with retry logic
        await job_queue.reject(
            job.id,
            requeue=True,
            error_message="Worker timeout - job requeued"
        )

# Run periodically
while True:
    await requeue_stale_jobs(job_queue, timeout_minutes=10)
    await asyncio.sleep(60)  # Check every minute
```

### How do I implement delayed job execution?

Use the `scheduled_at` field:

```python
from datetime import UTC, datetime, timedelta

# Schedule job for 1 hour from now
scheduled_time = datetime.now(UTC) + timedelta(hours=1)

job = Job(
    id=uuid4(),
    queue_name="reminders",
    job_type="reminder.send",
    payload={"message": "Your meeting starts in 10 minutes"},
    scheduled_at=scheduled_time
)

await job_queue.enqueue(job)

# Job won't be dequeued until scheduled_at time passes
```

Your queue adapter should check `scheduled_at` in the dequeue method:

```python
async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
    for job in pending_jobs:
        # Skip jobs scheduled for future
        if job.scheduled_at and job.scheduled_at > datetime.now(UTC):
            continue

        return job  # This job is ready now
```

### How do I handle jobs that need to run periodically (cron jobs)?

Use a scheduler that creates new jobs at regular intervals:

```python
import asyncio
from datetime import UTC, datetime, timedelta

async def schedule_periodic_job(
    job_queue: JobQueueAdapter,
    job_type: str,
    payload: dict,
    interval_minutes: int
):
    """Schedule a job to run periodically."""
    while True:
        # Create job for next execution
        next_run = datetime.now(UTC) + timedelta(minutes=interval_minutes)

        job = Job(
            id=uuid4(),
            queue_name="periodic",
            job_type=job_type,
            payload=payload,
            scheduled_at=next_run
        )

        await job_queue.enqueue(job)

        logger.info(
            "scheduled_periodic_job",
            job_type=job_type,
            next_run=next_run.isoformat()
        )

        # Wait until next interval
        await asyncio.sleep(interval_minutes * 60)

# Example: Send daily reports at 9 AM
asyncio.create_task(schedule_periodic_job(
    job_queue=job_queue,
    job_type="report.daily",
    payload={"report_type": "daily_summary"},
    interval_minutes=24 * 60  # 24 hours
))
```

### How do I implement priority-based processing across multiple queues?

Workers can check multiple queues in priority order:

```python
async def multi_queue_worker(
    job_queue: JobQueueAdapter,
    handlers: dict[str, JobHandler]
):
    """Worker that processes multiple queues by priority."""
    # Define queue priority order
    queue_priority = ["critical", "high", "default", "low"]

    while True:
        job_processed = False

        # Check queues in priority order
        for queue_name in queue_priority:
            job = await job_queue.dequeue(queue_name, timeout=1)

            if job:
                await process_job(job, handlers)
                job_processed = True
                break  # Start over from highest priority queue

        if not job_processed:
            # No jobs in any queue, wait a bit
            await asyncio.sleep(0.1)
```

### How do I test job queue adapters?

Test the adapter interface contract:

```python
import pytest
from uuid import uuid4
from portico.ports.job import Job, JobStatus

@pytest.mark.asyncio
async def test_enqueue_dequeue(job_queue_adapter):
    """Test basic enqueue/dequeue."""
    job = Job(
        id=uuid4(),
        queue_name="test",
        job_type="test.job",
        payload={"data": "test"}
    )

    # Enqueue
    enqueued = await job_queue_adapter.enqueue(job)
    assert enqueued.status == JobStatus.PENDING
    assert enqueued.created_at is not None

    # Dequeue
    dequeued = await job_queue_adapter.dequeue("test", timeout=1)
    assert dequeued is not None
    assert dequeued.id == job.id
    assert dequeued.status == JobStatus.RUNNING

@pytest.mark.asyncio
async def test_retry_logic(job_queue_adapter):
    """Test retry and dead letter logic."""
    job = Job(
        id=uuid4(),
        queue_name="test",
        job_type="test.job",
        payload={},
        max_retries=2
    )

    await job_queue_adapter.enqueue(job)

    # Fail job 3 times
    for i in range(3):
        dequeued = await job_queue_adapter.dequeue("test", timeout=1)
        assert dequeued is not None
        await job_queue_adapter.reject(
            dequeued.id,
            requeue=True,
            error_message=f"Attempt {i+1} failed"
        )

    # Should be in dead letter now
    final_job = await job_queue_adapter.get_job(job.id)
    assert final_job.status == JobStatus.DEAD_LETTER
    assert final_job.retry_count == 2

# Fixture for different adapters
@pytest.fixture(params=["memory", "database"])
async def job_queue_adapter(request):
    if request.param == "memory":
        return MemoryJobQueueAdapter()
    elif request.param == "database":
        db = await create_test_database()
        adapter = DatabaseJobQueueAdapter(db)
        await adapter.initialize()
        yield adapter
        await adapter.close()
```

### How do I monitor and debug job queue issues?

Use comprehensive logging and metrics:

```python
from portico.utils.logging import get_logger

logger = get_logger(__name__)

class InstrumentedJobQueue(JobQueueAdapter):
    """Job queue adapter with monitoring."""

    def __init__(self, adapter: JobQueueAdapter):
        self.adapter = adapter
        self.metrics = MetricsCollector()

    async def enqueue(self, job: Job) -> Job:
        start = time.time()

        try:
            result = await self.adapter.enqueue(job)

            duration_ms = (time.time() - start) * 1000
            self.metrics.record("job.enqueued", duration_ms)

            logger.info(
                "job_enqueued",
                job_id=str(job.id),
                queue=job.queue_name,
                job_type=job.job_type,
                priority=job.priority,
                duration_ms=duration_ms
            )

            return result
        except Exception as e:
            logger.error(
                "enqueue_failed",
                job_id=str(job.id),
                error=str(e),
                exc_info=True
            )
            self.metrics.increment("job.enqueue_error")
            raise

    async def dequeue(self, queue_name: str, timeout: int = 30) -> Job | None:
        start = time.time()

        job = await self.adapter.dequeue(queue_name, timeout)

        duration_ms = (time.time() - start) * 1000

        if job:
            self.metrics.record("job.dequeued", duration_ms)
            logger.info(
                "job_dequeued",
                job_id=str(job.id),
                queue=queue_name,
                job_type=job.job_type,
                wait_time_ms=duration_ms
            )
        else:
            logger.debug("dequeue_timeout", queue=queue_name)

        return job
```

## Related Ports

- **Job Creator Port** (`portico.ports.job_creator`) - Interface for creating jobs (implemented by JobService)
- **Job Handler Port** (`portico.ports.job_handler`) - Interface for processing jobs
- **[Audit Port](audit.md)** - Audit logging for job lifecycle events

## Related Kits

- **JobService** (`portico.kits.job`) - Uses JobQueueAdapter for job storage and retrieval
- **WorkerManager** (`portico.kits.job.worker_manager`) - Uses JobQueueAdapter to dequeue and process jobs

## Adapters

Available implementations:

- **MemoryJobQueueAdapter** (`portico.adapters.job_queue.memory_queue`) - In-memory queue for testing
- **DatabaseJobQueueAdapter** (`portico.adapters.job_queue.database_queue`) - Database-backed queue with persistence

## Architecture Notes

The Job Queue Port follows hexagonal architecture principles:

- **Queue adapters are infrastructure**: They provide queue backend implementations
- **Dependency inversion**: JobService and WorkerManager depend on the JobQueueAdapter interface, not concrete implementations
- **Pluggable backends**: Swap memory queue for database queue without changing application code
- **Testability**: Use memory queue in tests, database/Redis in production

This pattern enables:
- Testing with fast in-memory queues
- Production deployment with persistent queues
- Easy migration between queue backends
- Consistent job processing interface
