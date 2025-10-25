# Job Trigger Port

## Overview

The Job Trigger Port defines the contract for mechanisms that create jobs from external events. Triggers are adapters that respond to various event sources (HTTP webhooks, cron schedules, file system changes, message queues, etc.) and create background jobs using the JobCreator port.

**Purpose**: Enable pluggable event sources for job creation while maintaining hexagonal architecture by depending on the JobCreator port interface rather than concrete service implementations.

**Domain**: Background job automation, event-driven job creation, and scheduled task execution

**Key Capabilities**:

- Create jobs from external event sources (webhooks, schedules, queues, file systems)
- Lifecycle management (start/stop triggers)
- Decoupled from job execution (only creates jobs, doesn't process them)
- Integration with JobCreator port for dependency inversion

**Port Type**: Adapter

**When to Use**:

- When implementing event-driven job creation (webhooks, API callbacks)
- When building scheduled task systems (cron-like job scheduling)
- When integrating with external systems that should trigger job processing
- When you need automated job creation from various sources

## Port Interface

### JobTrigger

The `JobTrigger` abstract base class defines the contract for trigger implementations.

**Location**: `portico.ports.job_trigger.JobTrigger`

#### Methods

##### start

```python
@abstractmethod
async def start(self) -> None
```

Start the trigger to begin creating jobs from events.

**Returns**: None

**Purpose**: Initialize the trigger and begin listening for events. This might start:
- HTTP server listening for webhooks
- Cron scheduler polling for scheduled times
- File system watcher monitoring directories
- Message queue consumer reading from topics

**Example**:

```python
from portico.adapters.job_trigger.webhook_trigger import WebhookTrigger

# Create trigger
trigger = WebhookTrigger(job_creator=job_service)

# Start listening for webhook events
await trigger.start()

# Trigger is now running and creating jobs from webhooks
```

##### stop

```python
@abstractmethod
async def stop(self) -> None
```

Stop the trigger gracefully, ceasing job creation from events.

**Returns**: None

**Purpose**: Shutdown the trigger and stop listening for events. Should gracefully cleanup resources (close connections, unregister listeners, stop schedulers).

**Example**:

```python
# Gracefully shutdown trigger
await trigger.stop()

# Trigger has stopped and won't create new jobs
```

##### is_running

```python
@property
@abstractmethod
def is_running(self) -> bool
```

Check if trigger is currently running.

**Returns**: `bool` - True if trigger is active and creating jobs from events

**Purpose**: Query trigger status for monitoring, health checks, or preventing duplicate starts.

**Example**:

```python
if trigger.is_running:
    print("Trigger is active and processing events")
else:
    print("Trigger is stopped")
```

## Architectural Pattern

Job Triggers follow hexagonal architecture by depending on the **JobCreator port** rather than JobService directly:

```
┌────────────────────────────────┐
│  Job Triggers (Adapters)       │
│  - WebhookTrigger              │
│  - ScheduleTrigger             │
│  - FileTrigger                 │
│  - MessageQueueTrigger         │
└────────────────────────────────┘
              ↓ depends on
┌────────────────────────────────┐
│  JobCreator Port (Interface)   │
│  - create_job()                │
└────────────────────────────────┘
              ↑ implemented by
┌────────────────────────────────┐
│  JobService (Kit)              │
│  - Implements JobCreator       │
│  - Uses JobQueue adapter       │
└────────────────────────────────┘
```

**Why this matters:**
- ✅ Triggers don't depend on concrete JobService implementation
- ✅ Can test triggers with mock JobCreator
- ✅ Maintains clean separation of concerns
- ✅ Follows dependency inversion principle

## Common Patterns

### 1. Webhook Trigger for API Integration

```python
from fastapi import FastAPI
from portico.adapters.job_trigger.webhook_trigger import (
    WebhookTrigger,
    WebhookConfig
)
from portico.kits.job.job_service import JobService

# Create job service (implements JobCreator)
job_service = JobService(job_queue=job_queue_adapter)

# Create webhook trigger
webhook_config = WebhookConfig(
    prefix="/webhooks",
    allowed_job_types=["email.send", "report.generate", "data.import"]
)

webhook_trigger = WebhookTrigger(
    job_creator=job_service,
    config=webhook_config
)

# Integrate with FastAPI
app = FastAPI()
app.include_router(webhook_trigger.router)

# Start trigger
await webhook_trigger.start()

# External systems can now create jobs via HTTP POST
# POST /webhooks/jobs
# {
#   "job_type": "email.send",
#   "payload": {"recipient": "user@example.com"},
#   "queue_name": "emails",
#   "priority": 5
# }
```

### 2. Scheduled Trigger for Cron Jobs

```python
from portico.adapters.job_trigger.schedule_trigger import (
    ScheduleTrigger,
    ScheduleConfig
)

# Define schedule configurations
schedules = [
    # Daily report at 9 AM
    ScheduleConfig(
        cron="0 9 * * *",
        job_type="report.daily",
        payload={"report_type": "daily_summary"},
        queue_name="reports",
        priority=5
    ),

    # Cleanup every hour
    ScheduleConfig(
        cron="0 * * * *",
        job_type="cleanup.temp_files",
        payload={"max_age_hours": 24},
        queue_name="maintenance",
        priority=1
    ),

    # Weekly backup on Sundays at 2 AM
    ScheduleConfig(
        cron="0 2 * * 0",
        job_type="backup.full",
        payload={"backup_type": "weekly"},
        queue_name="backups",
        priority=10
    )
]

# Create schedule trigger
schedule_trigger = ScheduleTrigger(
    job_creator=job_service,
    schedules=schedules
)

# Start scheduler
await schedule_trigger.start()

# Jobs will now be created automatically on schedule
```

### 3. File System Trigger for Processing Uploads

```python
import asyncio
from pathlib import Path
from portico.ports.job_creator import JobCreator
from portico.ports.job_trigger import JobTrigger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileTrigger(JobTrigger):
    """Trigger that creates jobs when files are added to a directory."""

    def __init__(
        self,
        job_creator: JobCreator,
        watch_path: Path,
        job_type: str,
        queue_name: str = "file_processing"
    ):
        self.job_creator = job_creator
        self.watch_path = watch_path
        self.job_type = job_type
        self.queue_name = queue_name
        self._observer = None
        self._running = False

    async def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        self._running = True

        event_handler = FileCreatedHandler(
            self.job_creator,
            self.job_type,
            self.queue_name
        )

        self._observer = Observer()
        self._observer.schedule(
            event_handler,
            str(self.watch_path),
            recursive=True
        )
        self._observer.start()

    async def stop(self) -> None:
        """Stop watching the directory."""
        if not self._running:
            return

        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()

    @property
    def is_running(self) -> bool:
        return self._running


class FileCreatedHandler(FileSystemEventHandler):
    """Handler that creates jobs for new files."""

    def __init__(self, job_creator: JobCreator, job_type: str, queue_name: str):
        self.job_creator = job_creator
        self.job_type = job_type
        self.queue_name = queue_name

    def on_created(self, event):
        """Called when file is created."""
        if not event.is_directory:
            # Create job asynchronously
            asyncio.create_task(
                self.job_creator.create_job(
                    job_type=self.job_type,
                    payload={"file_path": event.src_path},
                    queue_name=self.queue_name
                )
            )


# Usage
file_trigger = FileTrigger(
    job_creator=job_service,
    watch_path=Path("/uploads"),
    job_type="file.process",
    queue_name="file_processing"
)

await file_trigger.start()

# Jobs will be created automatically when files are uploaded to /uploads
```

### 4. Message Queue Trigger for Event-Driven Processing

```python
import json
from portico.ports.job_creator import JobCreator
from portico.ports.job_trigger import JobTrigger

class RabbitMQTrigger(JobTrigger):
    """Trigger that creates jobs from RabbitMQ messages."""

    def __init__(
        self,
        job_creator: JobCreator,
        rabbitmq_url: str,
        queue_name: str,
        job_type_mapping: dict[str, str]
    ):
        self.job_creator = job_creator
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.job_type_mapping = job_type_mapping
        self._connection = None
        self._channel = None
        self._running = False

    async def start(self) -> None:
        """Start consuming messages from RabbitMQ."""
        if self._running:
            return

        self._running = True

        # Connect to RabbitMQ
        self._connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self._channel = await self._connection.channel()

        # Declare queue
        queue = await self._channel.declare_queue(
            self.queue_name,
            durable=True
        )

        # Start consuming
        await queue.consume(self._on_message)

    async def stop(self) -> None:
        """Stop consuming messages."""
        if not self._running:
            return

        self._running = False

        if self._connection:
            await self._connection.close()

    @property
    def is_running(self) -> bool:
        return self._running

    async def _on_message(self, message):
        """Handle incoming message by creating job."""
        async with message.process():
            try:
                # Parse message
                data = json.loads(message.body.decode())

                # Map event type to job type
                event_type = data.get("event_type")
                job_type = self.job_type_mapping.get(event_type)

                if job_type:
                    # Create job from message
                    await self.job_creator.create_job(
                        job_type=job_type,
                        payload=data.get("payload", {}),
                        queue_name=data.get("queue", "default"),
                        priority=data.get("priority", 0)
                    )

            except Exception as e:
                logger.error("error_processing_message", error=str(e))


# Usage
mq_trigger = RabbitMQTrigger(
    job_creator=job_service,
    rabbitmq_url="amqp://localhost",
    queue_name="job_events",
    job_type_mapping={
        "user.created": "email.welcome",
        "order.placed": "order.process",
        "payment.received": "invoice.send"
    }
)

await mq_trigger.start()

# Jobs will be created from RabbitMQ messages
```

### 5. Multiple Triggers with Application Lifecycle

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from portico.ports.job_trigger import JobTrigger

# Create multiple triggers
webhook_trigger = WebhookTrigger(job_creator=job_service)
schedule_trigger = ScheduleTrigger(job_creator=job_service, schedules=schedules)
file_trigger = FileTrigger(job_creator=job_service, watch_path=Path("/uploads"))

# Manage trigger lifecycle with application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: Start all triggers
    triggers = [webhook_trigger, schedule_trigger, file_trigger]

    for trigger in triggers:
        await trigger.start()
        logger.info(
            "trigger_started",
            trigger_type=type(trigger).__name__,
            is_running=trigger.is_running
        )

    yield  # Application runs

    # Shutdown: Stop all triggers
    for trigger in triggers:
        await trigger.stop()
        logger.info(
            "trigger_stopped",
            trigger_type=type(trigger).__name__,
            is_running=trigger.is_running
        )


app = FastAPI(lifespan=lifespan)
app.include_router(webhook_trigger.router)

# All triggers start with app and stop gracefully on shutdown
```

### 6. Custom Database Change Trigger

```python
from datetime import datetime, timedelta
from portico.ports.job_creator import JobCreator
from portico.ports.job_trigger import JobTrigger

class DatabasePollingTrigger(JobTrigger):
    """Trigger that polls database for new records and creates jobs."""

    def __init__(
        self,
        job_creator: JobCreator,
        database: Any,
        poll_interval_seconds: int = 60
    ):
        self.job_creator = job_creator
        self.database = database
        self.poll_interval = poll_interval_seconds
        self._task = None
        self._running = False

    async def start(self) -> None:
        """Start polling database."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling database."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    async def _poll_loop(self) -> None:
        """Poll database for new pending items."""
        while self._running:
            try:
                # Query for pending items
                pending_items = await self._get_pending_items()

                for item in pending_items:
                    # Create job for each pending item
                    await self.job_creator.create_job(
                        job_type="item.process",
                        payload={
                            "item_id": str(item.id),
                            "data": item.data
                        },
                        queue_name="processing"
                    )

                    # Mark as queued
                    await self._mark_as_queued(item.id)

                # Wait before next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("polling_error", error=str(e))
                await asyncio.sleep(self.poll_interval)

    async def _get_pending_items(self):
        """Get items that need processing."""
        async with self.database.session() as session:
            stmt = select(ItemModel).where(
                ItemModel.status == "pending",
                ItemModel.created_at > datetime.now() - timedelta(hours=1)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

    async def _mark_as_queued(self, item_id):
        """Mark item as queued for processing."""
        async with self.database.session() as session:
            stmt = (
                update(ItemModel)
                .where(ItemModel.id == item_id)
                .values(status="queued")
            )
            await session.execute(stmt)
            await session.commit()


# Usage
db_trigger = DatabasePollingTrigger(
    job_creator=job_service,
    database=database,
    poll_interval_seconds=30
)

await db_trigger.start()

# Polls database every 30 seconds and creates jobs for pending items
```

## Best Practices

### 1. Depend on JobCreator Port, Not JobService

Triggers should depend on the JobCreator interface for testability and flexibility.

```python
# ✅ GOOD - Depends on port interface
class WebhookTrigger(JobTrigger):
    def __init__(self, job_creator: JobCreator):
        self.job_creator = job_creator  # Interface dependency

    async def _create_job_endpoint(self, request):
        job = await self.job_creator.create_job(  # Uses interface
            job_type=request.job_type,
            payload=request.payload
        )

# ❌ BAD - Depends on concrete implementation
class WebhookTrigger(JobTrigger):
    def __init__(self, job_service: JobService):  # ❌ Concrete dependency
        self.job_service = job_service

    async def _create_job_endpoint(self, request):
        job = await self.job_service.create_job(...)  # Tight coupling
```

### 2. Implement Graceful Shutdown

Properly cleanup resources in stop() method.

```python
# ✅ GOOD - Graceful shutdown
class ScheduleTrigger(JobTrigger):
    async def stop(self) -> None:
        if not self._running:
            return

        logger.info("stopping_schedule_trigger")

        # Shutdown scheduler gracefully
        self._scheduler.shutdown(wait=True)  # Wait for running jobs

        self._running = False
        logger.info("schedule_trigger_stopped")

# ❌ BAD - Abrupt shutdown
class ScheduleTrigger(JobTrigger):
    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)  # ❌ Doesn't wait
        self._running = False  # ❌ No cleanup
```

### 3. Prevent Duplicate Starts

Check if already running before starting.

```python
# ✅ GOOD - Idempotent start
async def start(self) -> None:
    if self._running:
        logger.warning("trigger_already_running")
        return  # Don't start again

    logger.info("starting_trigger")
    self._running = True
    # Start logic...

# ❌ BAD - Can start multiple times
async def start(self) -> None:
    # ❌ No check, could start multiple schedulers/listeners
    self._running = True
    self._scheduler.start()
```

### 4. Handle Errors in Event Processing

Catch and log errors when creating jobs from events.

```python
# ✅ GOOD - Error handling
async def _on_webhook_request(self, request):
    try:
        job = await self.job_creator.create_job(
            job_type=request.job_type,
            payload=request.payload
        )

        logger.info("job_created_from_webhook", job_id=str(job.id))
        return {"job_id": str(job.id)}

    except ValidationError as e:
        logger.warning("invalid_webhook_request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(
            "webhook_job_creation_failed",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Job creation failed")

# ❌ BAD - No error handling
async def _on_webhook_request(self, request):
    job = await self.job_creator.create_job(...)  # ❌ Unhandled errors
    return {"job_id": str(job.id)}
```

### 5. Use Structured Logging

Log trigger lifecycle and job creation events with context.

```python
# ✅ GOOD - Structured logging
async def start(self) -> None:
    logger.info(
        "starting_webhook_trigger",
        prefix=self.config.prefix,
        allowed_types=self.config.allowed_job_types
    )

    self._running = True

    logger.info("webhook_trigger_started")

async def _create_job(self, config):
    try:
        job = await self.job_creator.create_job(...)

        logger.info(
            "job_created_from_schedule",
            job_id=str(job.id),
            job_type=config.job_type,
            cron=config.cron
        )

    except Exception as e:
        logger.error(
            "scheduled_job_creation_failed",
            job_type=config.job_type,
            error=str(e),
            exc_info=True
        )

# ❌ BAD - Unstructured logging
async def start(self) -> None:
    print("Starting trigger")  # ❌ Not structured
    self._running = True
```

### 6. Validate Input from External Sources

Validate data from webhooks, messages, files before creating jobs.

```python
# ✅ GOOD - Input validation
async def _on_webhook_request(self, request: WebhookJobRequest):
    # Validate job type allowlist
    if self.config.allowed_job_types:
        if request.job_type not in self.config.allowed_job_types:
            raise HTTPException(
                status_code=403,
                detail=f"Job type '{request.job_type}' not allowed"
            )

    # Validate payload is JSON-serializable
    try:
        json.dumps(request.payload)
    except TypeError:
        raise HTTPException(
            status_code=400,
            detail="Payload must be JSON-serializable"
        )

    # Create job
    job = await self.job_creator.create_job(...)

# ❌ BAD - No validation
async def _on_webhook_request(self, request):
    # ❌ Accepts any job type
    # ❌ Doesn't validate payload
    job = await self.job_creator.create_job(
        job_type=request.job_type,
        payload=request.payload  # Might not be serializable
    )
```

### 7. Use Dependency Injection for Testing

Accept JobCreator in constructor for easy testing.

```python
# ✅ GOOD - Dependency injection
class WebhookTrigger(JobTrigger):
    def __init__(self, job_creator: JobCreator):
        self.job_creator = job_creator

# Testing
from unittest.mock import AsyncMock

mock_creator = AsyncMock(spec=JobCreator)
trigger = WebhookTrigger(job_creator=mock_creator)

# Can verify job_creator.create_job was called
await trigger._create_job_endpoint(request)
mock_creator.create_job.assert_called_once()

# ❌ BAD - Creates dependencies internally
class WebhookTrigger(JobTrigger):
    def __init__(self, job_queue_url: str):
        # ❌ Hard to test
        self.job_service = JobService(JobQueueAdapter(job_queue_url))
```

## FAQs

### What's the difference between JobTrigger and JobHandler?

**JobTrigger** (creates jobs):
- **Input**: External events (webhooks, schedules, messages)
- **Output**: Jobs in queue
- **Purpose**: Trigger job creation from external sources
- **Depends on**: JobCreator port
- **Examples**: WebhookTrigger, ScheduleTrigger

**JobHandler** (processes jobs):
- **Input**: Jobs from queue
- **Output**: Job results
- **Purpose**: Execute business logic for job types
- **Depends on**: Business services
- **Examples**: EmailHandler, ReportHandler

```
External Event → JobTrigger → JobCreator → Job Queue → JobHandler → Result
```

### How do I test job triggers in isolation?

Use a mock JobCreator to verify trigger behavior:

```python
import pytest
from unittest.mock import AsyncMock
from portico.ports.job_creator import JobCreator

@pytest.mark.asyncio
async def test_webhook_trigger_creates_job():
    # Mock JobCreator
    mock_creator = AsyncMock(spec=JobCreator)
    mock_creator.create_job.return_value = Job(
        id=uuid4(),
        job_type="test.job",
        queue_name="default",
        payload={}
    )

    # Create trigger with mock
    trigger = WebhookTrigger(job_creator=mock_creator)

    # Simulate webhook request
    await trigger._create_job_endpoint(
        WebhookJobRequest(
            job_type="test.job",
            payload={"data": "test"}
        )
    )

    # Verify create_job was called
    mock_creator.create_job.assert_called_once_with(
        job_type="test.job",
        payload={"data": "test"},
        queue_name="default",
        priority=0
    )
```

### How do I prevent duplicate scheduled jobs?

Use idempotency keys or check for existing jobs before creating:

```python
class ScheduleTrigger(JobTrigger):
    async def _create_scheduled_job(self, config: ScheduleConfig):
        # Option 1: Use deterministic job ID
        job_id = uuid5(NAMESPACE_DNS, f"{config.job_type}:{config.cron}")

        # Check if job already exists
        existing = await self.job_creator.get_job(job_id)
        if existing and existing.status in [JobStatus.PENDING, JobStatus.SCHEDULED]:
            logger.debug("scheduled_job_already_exists", job_id=str(job_id))
            return

        # Create job with deterministic ID
        await self.job_creator.create_job(
            job_type=config.job_type,
            payload={**config.payload, "scheduled_run": datetime.now().isoformat()},
            metadata={"idempotency_key": str(job_id)}
        )
```

### Can triggers create jobs with different priorities?

Yes, triggers can set priority when creating jobs:

```python
# Webhook trigger with dynamic priority
async def _create_job_endpoint(self, request: WebhookJobRequest):
    # Map job type to priority
    priority_map = {
        "payment.process": 10,  # High priority
        "email.send": 5,        # Medium priority
        "cleanup.logs": 1       # Low priority
    }

    priority = priority_map.get(request.job_type, request.priority)

    job = await self.job_creator.create_job(
        job_type=request.job_type,
        payload=request.payload,
        priority=priority
    )

# Schedule trigger with configured priorities
schedules = [
    ScheduleConfig(
        cron="*/5 * * * *",
        job_type="health.check",
        payload={},
        priority=1  # Low priority
    ),
    ScheduleConfig(
        cron="0 9 * * *",
        job_type="report.daily",
        payload={},
        priority=10  # High priority
    )
]
```

### How do I handle rate limiting in triggers?

Implement rate limiting before creating jobs:

```python
from asyncio import Semaphore
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimitedWebhookTrigger(JobTrigger):
    def __init__(self, job_creator: JobCreator, max_requests_per_minute: int = 60):
        self.job_creator = job_creator
        self.max_requests = max_requests_per_minute
        self.request_counts = defaultdict(list)
        self._semaphore = Semaphore(max_requests_per_minute)

    async def _create_job_endpoint(self, request: WebhookJobRequest):
        # Check rate limit
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        self.request_counts[request.job_type] = [
            ts for ts in self.request_counts[request.job_type]
            if ts > one_minute_ago
        ]

        # Check limit
        if len(self.request_counts[request.job_type]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {request.job_type}"
            )

        # Record request
        self.request_counts[request.job_type].append(now)

        # Create job
        async with self._semaphore:
            return await self.job_creator.create_job(
                job_type=request.job_type,
                payload=request.payload
            )
```

### Should triggers validate job payloads?

Yes, triggers should validate inputs from external sources:

```python
from pydantic import BaseModel, validator

class EmailJobPayload(BaseModel):
    recipient: str
    subject: str
    body: str

    @validator('recipient')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v


class WebhookTrigger(JobTrigger):
    def __init__(self, job_creator: JobCreator, payload_schemas: dict):
        self.job_creator = job_creator
        self.payload_schemas = payload_schemas  # job_type -> Pydantic model

    async def _create_job_endpoint(self, request: WebhookJobRequest):
        # Validate payload against schema
        schema = self.payload_schemas.get(request.job_type)
        if schema:
            try:
                validated = schema(**request.payload)
                payload = validated.dict()
            except ValidationError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid payload: {e}"
                )
        else:
            payload = request.payload

        # Create job with validated payload
        return await self.job_creator.create_job(
            job_type=request.job_type,
            payload=payload
        )


# Usage
trigger = WebhookTrigger(
    job_creator=job_service,
    payload_schemas={
        "email.send": EmailJobPayload,
        "report.generate": ReportJobPayload
    }
)
```

### How do I monitor trigger health?

Implement health checks and metrics:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TriggerHealth:
    is_running: bool
    last_job_created: datetime | None
    jobs_created_count: int
    errors_count: int


class MonitoredWebhookTrigger(JobTrigger):
    def __init__(self, job_creator: JobCreator):
        self.job_creator = job_creator
        self._running = False
        self._last_job_created = None
        self._jobs_created = 0
        self._errors = 0

    async def _create_job_endpoint(self, request: WebhookJobRequest):
        try:
            job = await self.job_creator.create_job(...)

            # Update metrics
            self._last_job_created = datetime.now()
            self._jobs_created += 1

            logger.info(
                "job_created",
                total_jobs=self._jobs_created,
                total_errors=self._errors
            )

            return job

        except Exception as e:
            self._errors += 1
            logger.error("job_creation_error", total_errors=self._errors)
            raise

    def get_health(self) -> TriggerHealth:
        """Get trigger health status."""
        return TriggerHealth(
            is_running=self._running,
            last_job_created=self._last_job_created,
            jobs_created_count=self._jobs_created,
            errors_count=self._errors
        )


# Health check endpoint
@app.get("/health/triggers")
async def trigger_health():
    return {
        "webhook": webhook_trigger.get_health(),
        "schedule": schedule_trigger.get_health()
    }
```

### How do I handle trigger failures?

Implement retry logic and alerting:

```python
class ResilientScheduleTrigger(JobTrigger):
    async def _create_scheduled_job(self, config: ScheduleConfig):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                job = await self.job_creator.create_job(
                    job_type=config.job_type,
                    payload=config.payload
                )

                logger.info(
                    "scheduled_job_created",
                    job_id=str(job.id),
                    attempt=attempt + 1
                )

                return

            except Exception as e:
                logger.warning(
                    "scheduled_job_creation_failed",
                    job_type=config.job_type,
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # All retries exhausted, send alert
                    logger.error(
                        "scheduled_job_creation_failed_permanently",
                        job_type=config.job_type,
                        error=str(e),
                        exc_info=True
                    )

                    await self.alert_service.send_alert(
                        level="error",
                        message=f"Failed to create scheduled job: {config.job_type}",
                        context={"error": str(e)}
                    )
```

## Related Ports

- **Job Creator Port** (`portico.ports.job_creator`) - Interface for creating jobs (used by triggers)
- **Job Handler Port** (`portico.ports.job_handler`) - Interface for processing jobs
- **Job Queue Port** (`portico.ports.job_queue`) - Interface for queue adapters

## Related Kits

- **JobService** (`portico.kits.job`) - Implements JobCreator, used by triggers to create jobs

## Adapters

Available implementations:

- **WebhookTrigger** (`portico.adapters.job_trigger.webhook_trigger`) - HTTP webhook-based trigger
- **ScheduleTrigger** (`portico.adapters.job_trigger.schedule_trigger`) - Cron-style scheduled trigger

## Architecture Notes

The Job Trigger Port follows hexagonal architecture principles:

- **Triggers are adapters**: They adapt external event sources to job creation
- **Dependency inversion**: Triggers depend on JobCreator port, not JobService
- **Separation of concerns**: Job creation (triggers) is separated from job execution (handlers)
- **Testability**: Triggers can be tested with mock JobCreator

This pattern enables:
- Multiple event sources for job creation (webhooks, schedules, files, queues)
- Easy testing without running actual schedulers or servers
- Flexible job creation logic independent of execution
- Clean separation between infrastructure and domain logic
