# Job Handler Port

## Overview

The Job Handler Port defines the contract for implementing business logic that processes background jobs. This is the interface that application developers implement to execute specific job types.

**Purpose**: Enable custom business logic to process background jobs while maintaining separation between job orchestration (WorkerManager) and job execution (handler implementations).

**Domain**: Background job processing and asynchronous task execution

**Key Capabilities**:

- Process background jobs with custom business logic
- Handle job failures with custom cleanup/notification logic
- Type-safe job type identification
- Integration with WorkerManager for job orchestration

**Port Type**: Handler

**When to Use**:

- When implementing business logic to process specific job types (email sending, report generation, data import, etc.)
- When you need custom failure handling (cleanup, notifications, dead letter processing)
- When building asynchronous task processing systems
- When integrating with job queue systems

## Domain Models

The Job Handler Port uses domain models from `portico.ports.job`:

- **Job** - Represents a background job with type, payload, status, retry configuration
- **JobResult** - Result of job execution (success/failure, result data, error information)
- **JobStatus** - Enumeration of job statuses (PENDING, SCHEDULED, RUNNING, COMPLETED, FAILED, RETRYING, DEAD_LETTER)

These models are shared across the job processing system and are defined in `portico.ports.job`.

## Port Interface

### JobHandler

The `JobHandler` abstract base class defines the contract for job processing business logic.

**Location**: `portico.ports.job_handler.JobHandler`

#### Properties

##### job_type

```python
@property
@abstractmethod
def job_type(self) -> str
```

Returns the job type this handler processes (e.g., `"email.send"`, `"report.generate"`).

**Returns**: Job type string identifier

**Purpose**: Allows the WorkerManager to route jobs to the appropriate handler.

**Example**:

```python
from portico.ports.job_handler import JobHandler

class EmailJobHandler(JobHandler):
    @property
    def job_type(self) -> str:
        return "email.send"
```

#### Methods

##### handle

```python
@abstractmethod
async def handle(self, job: Job) -> JobResult
```

Process the job and return result.

**Parameters**:

- `job` (`Job`): Job to process with type, payload, and metadata

**Returns**: `JobResult` indicating success/failure with optional result data

**Raises**: Exceptions are caught by WorkerManager and trigger retry logic

**Purpose**: Contains the core business logic for processing a specific job type.

**Example**:

```python
async def handle(self, job: Job) -> JobResult:
    """Send email based on job payload."""
    try:
        recipient = job.payload["recipient"]
        subject = job.payload["subject"]
        body = job.payload["body"]

        await self.email_service.send_email(
            to=recipient,
            subject=subject,
            body=body
        )

        return JobResult(
            success=True,
            result_data={"sent_at": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error("email_send_failed", error=str(e))
        return JobResult(
            success=False,
            error=e,
            result_data={}
        )
```

##### on_failure

```python
@abstractmethod
async def on_failure(self, job: Job, error: Exception) -> None
```

Called when job fails (for cleanup, notifications, dead letter processing, etc.).

**Parameters**:

- `job` (`Job`): Job that failed
- `error` (`Exception`): Exception that caused the failure

**Returns**: None

**Purpose**: Perform cleanup actions, send notifications, or handle dead letter processing when a job fails.

**Important Notes**:

- Called on every failure, including retries
- Called even if the job will be retried
- Exceptions raised in `on_failure` are logged but don't affect job retry logic
- Useful for alerting, cleanup, or moving failed jobs to a dead letter queue

**Example**:

```python
async def on_failure(self, job: Job, error: Exception) -> None:
    """Log failure and send alert if job exhausted retries."""
    logger.error(
        "job_failed",
        job_id=str(job.id),
        job_type=job.job_type,
        error=str(error),
        retry_count=job.retry_count,
        max_retries=job.max_retries
    )

    # Send alert if job exhausted all retries
    if job.retry_count >= job.max_retries:
        await self.alert_service.send_alert(
            level="error",
            message=f"Job {job.id} failed after {job.max_retries} retries",
            context={
                "job_type": job.job_type,
                "error": str(error),
                "payload": job.payload
            }
        )
```

## Integration with WorkerManager

The `WorkerManager` orchestrates job processing by routing jobs to registered handlers:

```python
from portico.kits.job.worker_manager import WorkerManager

# Create handlers
email_handler = EmailJobHandler(email_service)
report_handler = ReportJobHandler(report_service)

# Register handlers with WorkerManager
manager = WorkerManager(
    job_queue=job_queue_adapter,
    handlers={
        "email.send": email_handler,
        "report.generate": report_handler,
    },
    concurrency=10,
    queues=["default", "high_priority"]
)

# Start processing jobs
await manager.start()

# ... application runs ...

# Gracefully shutdown workers
await manager.stop()
```

**How it works**:

1. WorkerManager dequeues jobs from configured queues
2. Looks up the handler for the job's `job_type`
3. Calls `handler.handle(job)` to process the job
4. If successful, acknowledges the job
5. If failed, calls `handler.on_failure(job, error)` and applies retry logic

## Common Patterns

### 1. Email Sending Handler

```python
from typing import Any
from portico.ports.job import Job, JobResult
from portico.ports.job_handler import JobHandler
from portico.utils.logging import get_logger

logger = get_logger(__name__)

class EmailJobHandler(JobHandler):
    """Handler for sending emails asynchronously."""

    def __init__(self, email_service: Any):
        self.email_service = email_service

    @property
    def job_type(self) -> str:
        return "email.send"

    async def handle(self, job: Job) -> JobResult:
        """Send email from job payload."""
        try:
            recipient = job.payload["recipient"]
            subject = job.payload["subject"]
            body = job.payload["body"]
            template_id = job.payload.get("template_id")

            logger.info(
                "sending_email",
                job_id=str(job.id),
                recipient=recipient
            )

            if template_id:
                # Use template
                await self.email_service.send_templated_email(
                    to=recipient,
                    template_id=template_id,
                    variables=job.payload.get("variables", {})
                )
            else:
                # Direct email
                await self.email_service.send_email(
                    to=recipient,
                    subject=subject,
                    body=body
                )

            return JobResult(
                success=True,
                result_data={
                    "sent_at": datetime.now().isoformat(),
                    "recipient": recipient
                }
            )

        except Exception as e:
            logger.error(
                "email_send_failed",
                job_id=str(job.id),
                error=str(e)
            )
            return JobResult(
                success=False,
                error=e,
                result_data={}
            )

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Log email send failures."""
        logger.error(
            "email_job_failed",
            job_id=str(job.id),
            recipient=job.payload.get("recipient"),
            error=str(error),
            retry_count=job.retry_count
        )

        # Alert if exhausted retries
        if job.retry_count >= job.max_retries:
            await self.email_service.send_admin_alert(
                subject="Email Job Failed",
                body=f"Failed to send email after {job.max_retries} retries"
            )
```

### 2. Data Processing Handler with Validation

```python
from portico.exceptions import ValidationError
from portico.ports.job import Job, JobResult
from portico.ports.job_handler import JobHandler

class DataImportJobHandler(JobHandler):
    """Handler for importing data from external sources."""

    def __init__(self, import_service: Any, storage_service: Any):
        self.import_service = import_service
        self.storage_service = storage_service

    @property
    def job_type(self) -> str:
        return "data.import"

    async def handle(self, job: Job) -> JobResult:
        """Import data from source URL."""
        try:
            # Validate payload
            source_url = job.payload.get("source_url")
            if not source_url:
                raise ValidationError("source_url is required")

            import_format = job.payload.get("format", "csv")
            user_id = job.payload.get("user_id")

            logger.info(
                "starting_data_import",
                job_id=str(job.id),
                source_url=source_url,
                format=import_format
            )

            # Download and parse data
            data = await self.import_service.fetch_data(source_url)
            parsed = await self.import_service.parse_data(data, import_format)

            # Store results
            result = await self.storage_service.store_import(
                data=parsed,
                user_id=user_id,
                metadata=job.metadata
            )

            logger.info(
                "data_import_completed",
                job_id=str(job.id),
                records_imported=len(parsed)
            )

            return JobResult(
                success=True,
                result_data={
                    "records_imported": len(parsed),
                    "import_id": str(result.id),
                    "completed_at": datetime.now().isoformat()
                }
            )

        except ValidationError as e:
            # Don't retry validation errors
            logger.error("validation_error", job_id=str(job.id), error=str(e))
            return JobResult(success=False, error=e, result_data={})

        except Exception as e:
            # Retry other errors
            logger.error("import_error", job_id=str(job.id), error=str(e))
            return JobResult(success=False, error=e, result_data={})

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Clean up and notify on import failure."""
        # Clean up partial imports
        if "import_id" in job.metadata:
            await self.storage_service.cleanup_failed_import(
                job.metadata["import_id"]
            )

        # Notify user if specified
        if "user_id" in job.payload:
            await self.notify_import_failure(
                job.payload["user_id"],
                job.payload.get("source_url"),
                error
            )
```

### 3. Report Generation Handler with Progress Tracking

```python
from datetime import datetime
from portico.ports.job import Job, JobResult
from portico.ports.job_handler import JobHandler

class ReportJobHandler(JobHandler):
    """Handler for generating reports."""

    def __init__(self, report_service: Any, file_storage: Any):
        self.report_service = report_service
        self.file_storage = file_storage

    @property
    def job_type(self) -> str:
        return "report.generate"

    async def handle(self, job: Job) -> JobResult:
        """Generate report and store file."""
        try:
            report_type = job.payload["report_type"]
            start_date = job.payload["start_date"]
            end_date = job.payload["end_date"]
            user_id = job.payload["user_id"]

            logger.info(
                "generating_report",
                job_id=str(job.id),
                report_type=report_type,
                date_range=f"{start_date} to {end_date}"
            )

            # Generate report (could take several minutes)
            report_data = await self.report_service.generate_report(
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
                filters=job.payload.get("filters", {})
            )

            # Store report file
            file_path = await self.file_storage.store_report(
                data=report_data,
                filename=f"{report_type}_{start_date}_{end_date}.pdf",
                user_id=user_id
            )

            logger.info(
                "report_generated",
                job_id=str(job.id),
                file_path=file_path
            )

            return JobResult(
                success=True,
                result_data={
                    "file_path": file_path,
                    "report_type": report_type,
                    "generated_at": datetime.now().isoformat(),
                    "size_bytes": len(report_data)
                }
            )

        except Exception as e:
            logger.error(
                "report_generation_failed",
                job_id=str(job.id),
                error=str(e)
            )
            return JobResult(success=False, error=e, result_data={})

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Notify user of report generation failure."""
        user_id = job.payload.get("user_id")
        if user_id and job.retry_count >= job.max_retries:
            # Send notification about failed report
            await self.notification_service.send_notification(
                user_id=user_id,
                type="report_failed",
                message=f"Report generation failed: {error}",
                metadata={"job_id": str(job.id)}
            )
```

### 4. Handler with External Service Retry Logic

```python
import asyncio
from portico.ports.job import Job, JobResult
from portico.ports.job_handler import JobHandler

class WebhookJobHandler(JobHandler):
    """Handler for sending webhooks to external services."""

    def __init__(self, http_client: Any):
        self.http_client = http_client

    @property
    def job_type(self) -> str:
        return "webhook.send"

    async def handle(self, job: Job) -> JobResult:
        """Send webhook with exponential backoff."""
        url = job.payload["url"]
        payload = job.payload["payload"]
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                logger.info(
                    "sending_webhook",
                    job_id=str(job.id),
                    url=url,
                    attempt=attempt + 1
                )

                response = await self.http_client.post(
                    url,
                    json=payload,
                    timeout=30
                )

                if response.status_code < 500:
                    # Success or client error (don't retry)
                    return JobResult(
                        success=response.status_code < 400,
                        result_data={
                            "status_code": response.status_code,
                            "response": response.text[:1000]
                        }
                    )

                # Server error - retry with backoff
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s

            except Exception as e:
                logger.warning(
                    "webhook_attempt_failed",
                    job_id=str(job.id),
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)

        # All attempts failed
        return JobResult(
            success=False,
            error=Exception("Webhook failed after all attempts"),
            result_data={"attempts": max_attempts}
        )

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Log webhook failures for monitoring."""
        logger.error(
            "webhook_permanently_failed",
            job_id=str(job.id),
            url=job.payload.get("url"),
            retry_count=job.retry_count,
            error=str(error)
        )
```

## Best Practices

### 1. ✅ Keep Handlers Stateless

Handlers should not maintain state between job executions. All necessary data should come from the job payload.

```python
# ✅ GOOD - Handler is stateless
class EmailHandler(JobHandler):
    def __init__(self, email_service: Any):
        self.email_service = email_service  # Dependency injection

    async def handle(self, job: Job) -> JobResult:
        # All data from job payload
        recipient = job.payload["recipient"]
        await self.email_service.send(recipient)
        return JobResult(success=True)

# ❌ BAD - Handler maintains state
class EmailHandler(JobHandler):
    def __init__(self):
        self.sent_count = 0  # ❌ State

    async def handle(self, job: Job) -> JobResult:
        self.sent_count += 1  # ❌ Won't work with multiple workers
        return JobResult(success=True)
```

### 2. ✅ Validate Payload Early

Validate job payload at the start of `handle()` to catch errors before doing expensive work.

```python
# ✅ GOOD - Validate immediately
async def handle(self, job: Job) -> JobResult:
    # Validate required fields
    if not job.payload.get("user_id"):
        return JobResult(
            success=False,
            error=ValidationError("user_id required")
        )

    if not job.payload.get("amount") or job.payload["amount"] <= 0:
        return JobResult(
            success=False,
            error=ValidationError("amount must be positive")
        )

    # Now proceed with expensive operations
    result = await self.process_payment(job.payload)
    return JobResult(success=True, result_data=result)

# ❌ BAD - Validation after expensive work
async def handle(self, job: Job) -> JobResult:
    # Do expensive work first
    data = await self.fetch_large_dataset()
    processed = await self.process_data(data)

    # Validate late ❌
    if not job.payload.get("user_id"):
        raise ValidationError("user_id required")
```

### 3. ✅ Use Structured Logging

Log key events with structured context for observability.

```python
# ✅ GOOD - Structured logging with context
async def handle(self, job: Job) -> JobResult:
    logger.info(
        "processing_job",
        job_id=str(job.id),
        job_type=job.job_type,
        retry_count=job.retry_count,
        user_id=job.payload.get("user_id")
    )

    try:
        result = await self.process(job)

        logger.info(
            "job_completed",
            job_id=str(job.id),
            duration_ms=result.duration
        )

        return JobResult(success=True, result_data=result)

    except Exception as e:
        logger.error(
            "job_failed",
            job_id=str(job.id),
            error=str(e),
            exc_info=True
        )
        return JobResult(success=False, error=e)

# ❌ BAD - Unstructured logging
async def handle(self, job: Job) -> JobResult:
    print(f"Processing job {job.id}")  # ❌ Not structured
    result = await self.process(job)
    return JobResult(success=True)
```

### 4. ✅ Return JobResult, Don't Raise Exceptions (Usually)

Return `JobResult` with `success=False` for expected failures. Only raise exceptions for unexpected errors.

```python
# ✅ GOOD - Return JobResult for expected failures
async def handle(self, job: Job) -> JobResult:
    try:
        user = await self.get_user(job.payload["user_id"])

        if not user:
            # Expected failure - return JobResult
            return JobResult(
                success=False,
                error=ResourceNotFoundError("User not found"),
                result_data={}
            )

        if user.balance < job.payload["amount"]:
            # Expected failure - return JobResult
            return JobResult(
                success=False,
                error=ValidationError("Insufficient balance"),
                result_data={}
            )

        # Process payment
        result = await self.process_payment(user, job.payload["amount"])
        return JobResult(success=True, result_data=result)

    except DatabaseError as e:
        # Unexpected error - re-raise for retry
        raise

# ❌ BAD - Raising for expected conditions
async def handle(self, job: Job) -> JobResult:
    user = await self.get_user(job.payload["user_id"])

    if not user:
        raise ResourceNotFoundError("User not found")  # ❌ Will retry unnecessarily

    if user.balance < job.payload["amount"]:
        raise ValidationError("Insufficient balance")  # ❌ Will retry unnecessarily
```

### 5. ✅ Use on_failure for Cleanup

Use `on_failure` for cleanup, alerting, and dead letter processing, not for core business logic.

```python
# ✅ GOOD - on_failure for cleanup
async def on_failure(self, job: Job, error: Exception) -> None:
    # Clean up resources
    if "temp_file" in job.metadata:
        await self.cleanup_temp_file(job.metadata["temp_file"])

    # Send alerts if exhausted retries
    if job.retry_count >= job.max_retries:
        await self.alert_service.send_alert(
            level="error",
            message=f"Job {job.id} failed permanently",
            context={"error": str(error)}
        )

    # Log for monitoring
    logger.error(
        "job_failed",
        job_id=str(job.id),
        error=str(error),
        will_retry=job.retry_count < job.max_retries
    )

# ❌ BAD - on_failure for business logic
async def on_failure(self, job: Job, error: Exception) -> None:
    # Don't do business logic in on_failure ❌
    await self.refund_payment(job.payload["payment_id"])

    # This runs on EVERY failure, including retries!
    # Could refund the same payment multiple times
```

### 6. ✅ Make Handlers Idempotent When Possible

Design handlers to safely retry without side effects.

```python
# ✅ GOOD - Idempotent handler
async def handle(self, job: Job) -> JobResult:
    payment_id = job.payload["payment_id"]

    # Check if already processed
    existing = await self.payment_service.get_payment(payment_id)
    if existing and existing.status == "completed":
        logger.info("payment_already_processed", payment_id=payment_id)
        return JobResult(
            success=True,
            result_data={"payment_id": payment_id, "already_processed": True}
        )

    # Process payment with idempotency key
    result = await self.payment_service.process_payment(
        payment_id=payment_id,
        idempotency_key=str(job.id)  # Use job ID
    )

    return JobResult(success=True, result_data=result)

# ❌ BAD - Not idempotent
async def handle(self, job: Job) -> JobResult:
    # Always creates new payment, even on retry ❌
    payment = await self.payment_service.create_payment(
        amount=job.payload["amount"]
    )
    return JobResult(success=True)
```

### 7. ✅ Use Dependency Injection

Inject dependencies through the constructor for testability.

```python
# ✅ GOOD - Dependencies injected
class EmailHandler(JobHandler):
    def __init__(
        self,
        email_service: Any,
        template_service: Any,
        audit_service: Any
    ):
        self.email_service = email_service
        self.template_service = template_service
        self.audit_service = audit_service

    async def handle(self, job: Job) -> JobResult:
        # Use injected services
        template = await self.template_service.get(job.payload["template_id"])
        await self.email_service.send(template)
        await self.audit_service.log("email_sent")
        return JobResult(success=True)

# ❌ BAD - Creating dependencies inside handler
class EmailHandler(JobHandler):
    async def handle(self, job: Job) -> JobResult:
        # Creating service instances ❌
        email_service = EmailService()
        template_service = TemplateService()

        # Hard to test, tight coupling
        template = await template_service.get(job.payload["template_id"])
        await email_service.send(template)
```

## FAQs

### When should I raise an exception vs return JobResult with success=False?

**Return `JobResult(success=False)`** for:
- Expected failures (validation errors, business rule violations)
- Failures that shouldn't be retried
- Client errors (400-level equivalents)

**Raise exceptions** for:
- Unexpected errors (database connection failures, external service timeouts)
- Errors that should trigger retries
- Infrastructure/transient errors (500-level equivalents)

```python
async def handle(self, job: Job) -> JobResult:
    # Expected failure - return JobResult
    if job.payload["amount"] < 0:
        return JobResult(
            success=False,
            error=ValidationError("Amount must be positive")
        )

    try:
        # Process payment
        result = await self.payment_api.charge(job.payload["amount"])
        return JobResult(success=True, result_data=result)
    except ConnectionError:
        # Transient error - raise for retry
        raise
    except PaymentDeclined as e:
        # Expected failure - return JobResult
        return JobResult(success=False, error=e)
```

### How do I pass context to handlers (database sessions, user context, etc.)?

Use dependency injection in the handler constructor. Pass services/adapters that provide the context you need.

```python
# Handler with database session factory
class DataHandler(JobHandler):
    def __init__(self, session_factory: Any):
        self.session_factory = session_factory

    async def handle(self, job: Job) -> JobResult:
        async with self.session_factory() as session:
            # Use session for database operations
            result = await session.execute(...)
            await session.commit()
        return JobResult(success=True)

# Handler with service that provides user context
class UserActionHandler(JobHandler):
    def __init__(self, user_service: Any, action_service: Any):
        self.user_service = user_service
        self.action_service = action_service

    async def handle(self, job: Job) -> JobResult:
        user = await self.user_service.get_user(job.payload["user_id"])
        result = await self.action_service.perform_action(user, job.payload["action"])
        return JobResult(success=True, result_data=result)
```

### What happens if on_failure raises an exception?

The WorkerManager logs the exception but continues with job retry logic. The `on_failure` exception doesn't affect whether the job is retried or moved to dead letter.

```python
async def on_failure(self, job: Job, error: Exception) -> None:
    try:
        # Attempt cleanup
        await self.cleanup_service.cleanup(job.id)
    except Exception as e:
        # Exception is logged but doesn't affect retry logic
        logger.error("cleanup_failed", error=str(e))
        # Job will still be retried or moved to dead letter based on retry count
```

### How do I test handlers in isolation?

Mock the dependencies injected into the handler constructor. The handler's business logic can be tested independently of the job queue.

```python
import pytest
from unittest.mock import AsyncMock, Mock
from portico.ports.job import Job, JobResult

@pytest.mark.asyncio
async def test_email_handler_success():
    # Mock dependencies
    email_service = AsyncMock()
    email_service.send_email.return_value = {"sent": True}

    # Create handler with mocks
    handler = EmailJobHandler(email_service)

    # Create test job
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={
            "recipient": "test@example.com",
            "subject": "Test",
            "body": "Test email"
        }
    )

    # Test handle method
    result = await handler.handle(job)

    # Verify
    assert result.success
    assert "sent_at" in result.result_data
    email_service.send_email.assert_called_once()

@pytest.mark.asyncio
async def test_email_handler_failure():
    # Mock service to raise exception
    email_service = AsyncMock()
    email_service.send_email.side_effect = Exception("SMTP error")

    handler = EmailJobHandler(email_service)

    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={"recipient": "test@example.com"}
    )

    # Test failure handling
    result = await handler.handle(job)

    assert not result.success
    assert result.error is not None
```

### How do I implement different handlers for the same job type?

You typically don't. Each job type should have one handler. If you need different processing logic, use different job types or use the job payload to control behavior.

```python
# ✅ GOOD - Different job types
class EmailJobHandler(JobHandler):
    @property
    def job_type(self) -> str:
        return "email.send"

class SmsJobHandler(JobHandler):
    @property
    def job_type(self) -> str:
        return "sms.send"

# Register both
manager = WorkerManager(
    job_queue=queue,
    handlers={
        "email.send": EmailJobHandler(email_service),
        "sms.send": SmsJobHandler(sms_service),
    }
)

# ✅ GOOD - Use payload to control behavior
class NotificationHandler(JobHandler):
    @property
    def job_type(self) -> str:
        return "notification.send"

    async def handle(self, job: Job) -> JobResult:
        notification_type = job.payload["type"]

        if notification_type == "email":
            return await self.send_email(job.payload)
        elif notification_type == "sms":
            return await self.send_sms(job.payload)
        elif notification_type == "push":
            return await self.send_push(job.payload)
```

### How do I handle long-running jobs that might exceed worker timeouts?

Break long jobs into smaller chunks, use progress tracking, or implement checkpoint/resume logic.

```python
# ✅ GOOD - Chunked processing
class DataProcessingHandler(JobHandler):
    @property
    def job_type(self) -> str:
        return "data.process_batch"

    async def handle(self, job: Job) -> JobResult:
        batch_id = job.payload["batch_id"]
        offset = job.payload.get("offset", 0)
        chunk_size = 1000

        # Process one chunk
        records = await self.fetch_records(batch_id, offset, chunk_size)
        processed = await self.process_records(records)

        # If more records exist, create next job
        if len(records) == chunk_size:
            await self.job_creator.create_job(
                job_type="data.process_batch",
                payload={
                    "batch_id": batch_id,
                    "offset": offset + chunk_size
                }
            )

        return JobResult(
            success=True,
            result_data={
                "processed": len(processed),
                "offset": offset
            }
        )
```

### How do I implement priority-based job processing?

Use different queues for different priorities and configure WorkerManager with multiple queues.

```python
# Create jobs with different queues
await job_service.create_job(
    job_type="email.send",
    payload={"recipient": "user@example.com"},
    queue_name="high_priority",  # High priority queue
    priority=10
)

await job_service.create_job(
    job_type="report.generate",
    payload={"report_id": "123"},
    queue_name="low_priority",  # Low priority queue
    priority=1
)

# Configure WorkerManager to process queues in order
manager = WorkerManager(
    job_queue=queue,
    handlers=handlers,
    queues=["high_priority", "default", "low_priority"],  # Order matters
    concurrency=10
)

# Workers will check high_priority first, then default, then low_priority
```

### How do I implement a dead letter queue handler?

Monitor dead letter jobs and create a handler to reprocess or analyze them.

```python
class DeadLetterHandler(JobHandler):
    """Handler for processing dead letter jobs."""

    @property
    def job_type(self) -> str:
        return "system.deadletter.process"

    async def handle(self, job: Job) -> JobResult:
        """Analyze and potentially requeue dead letter jobs."""
        dead_job_id = job.payload["dead_job_id"]

        # Get the dead letter job
        dead_job = await self.job_queue.get_job(dead_job_id)

        if not dead_job:
            return JobResult(success=True)  # Already processed

        # Analyze why it failed
        error_type = self._analyze_error(dead_job.error_message)

        # Log to monitoring system
        await self.monitoring_service.log_dead_letter(
            job_type=dead_job.job_type,
            error_type=error_type,
            payload=dead_job.payload
        )

        # Optionally requeue if error was transient
        if error_type == "transient":
            await self.job_queue.enqueue(dead_job)

        return JobResult(
            success=True,
            result_data={"analyzed": True, "requeued": error_type == "transient"}
        )

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Alert if dead letter processing fails."""
        await self.alert_service.send_critical_alert(
            "Dead letter handler failed",
            context={"job_id": str(job.id), "error": str(error)}
        )
```

## Related Ports

- **Job Creator Port** (`portico.ports.job_creator`) - Interface for creating jobs (implemented by JobService)
- **Job Queue Port** (`portico.ports.job_queue`) - Interface for queue adapters (memory, database, Redis)
- **[Audit Port](audit.md)** - Audit logging for job creation and execution

## Related Kits

- **JobService** (`portico.kits.job`) - Implements JobCreator, manages job lifecycle
- **WorkerManager** (`portico.kits.job.worker_manager`) - Orchestrates workers and routes jobs to handlers

## Architecture Notes

The Job Handler Port follows hexagonal architecture principles:

- **Handlers are adapters**: They adapt business logic to the job processing system
- **Dependency inversion**: WorkerManager depends on the JobHandler interface, not concrete implementations
- **Separation of concerns**: Job orchestration (WorkerManager) is separated from job execution (handlers)
- **Testability**: Handlers can be tested in isolation with mock dependencies

This pattern enables:
- Multiple handler implementations for different job types
- Easy testing of business logic without running workers
- Flexible job routing and processing strategies
- Clean separation between infrastructure and domain logic
