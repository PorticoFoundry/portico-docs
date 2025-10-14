"""Test examples for Job system documentation.

This module tests code examples from job-related documentation to ensure they remain
accurate and working.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

# --8<-- [start:imports]
from portico.ports.job import Job, JobResult, JobStatus, QueueStats
from portico.ports.job_handler import JobHandler

# --8<-- [end:imports]


# --8<-- [start:job-creation-basic]
def test_job_creation_basic():
    """Basic job creation."""
    job_id = uuid4()
    job = Job(
        id=job_id,
        queue_name="default",
        job_type="email.send",
        payload={
            "to": "user@example.com",
            "subject": "Welcome!",
            "body": "Thanks for signing up",
        },
        priority=5,
        max_retries=3,
    )

    assert job.id == job_id
    assert job.queue_name == "default"
    assert job.job_type == "email.send"
    assert job.payload["to"] == "user@example.com"
    assert job.priority == 5
    assert job.max_retries == 3
    assert job.status == JobStatus.PENDING


# --8<-- [end:job-creation-basic]


# --8<-- [start:job-with-schedule]
def test_job_with_schedule():
    """Job with scheduled execution time."""
    scheduled_time = datetime.now(UTC) + timedelta(hours=1)

    job = Job(
        id=uuid4(),
        queue_name="reports",
        job_type="report.generate",
        payload={"report_type": "monthly"},
        scheduled_at=scheduled_time,
    )

    assert job.status == JobStatus.PENDING
    assert job.scheduled_at == scheduled_time
    assert job.queue_name == "reports"


# --8<-- [end:job-with-schedule]


# --8<-- [start:job-status-enum]
def test_job_status_enum():
    """Job status enumeration."""
    assert JobStatus.PENDING == "pending"
    assert JobStatus.SCHEDULED == "scheduled"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"
    assert JobStatus.RETRYING == "retrying"
    assert JobStatus.DEAD_LETTER == "dead_letter"


# --8<-- [end:job-status-enum]


# --8<-- [start:job-result-success]
def test_job_result_success():
    """Successful job result."""
    result = JobResult(
        success=True,
        result_data={"sent_at": datetime.now(UTC).isoformat()},
    )

    assert result.success is True
    assert result.error is None
    assert "sent_at" in result.result_data


# --8<-- [end:job-result-success]


# --8<-- [start:job-result-failure]
def test_job_result_failure():
    """Failed job result."""
    error = ValueError("Invalid email address")

    result = JobResult(
        success=False,
        error=error,
    )

    assert result.success is False
    assert result.error == error
    assert isinstance(result.error, ValueError)


# --8<-- [end:job-result-failure]


# --8<-- [start:job-with-priority]
def test_job_with_priority():
    """Jobs with different priority levels."""
    # High priority job
    high_priority_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="alert.critical",
        payload={"message": "System down"},
        priority=10,  # Higher value = higher priority
    )

    # Normal priority job
    normal_priority_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={"to": "user@example.com"},
        priority=5,
    )

    # Low priority job
    low_priority_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="cleanup.old_data",
        payload={},
        priority=1,
    )

    assert high_priority_job.priority == 10
    assert normal_priority_job.priority == 5
    assert low_priority_job.priority == 1


# --8<-- [end:job-with-priority]


# --8<-- [start:job-with-metadata]
def test_job_with_metadata():
    """Job with metadata."""
    user_id = uuid4()

    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={"to": "user@example.com"},
        created_by=user_id,
        metadata={
            "source": "signup_flow",
            "campaign": "welcome_series",
            "variant": "A",
        },
    )

    assert job.created_by == user_id
    assert job.metadata["source"] == "signup_flow"
    assert job.metadata["campaign"] == "welcome_series"


# --8<-- [end:job-with-metadata]


# --8<-- [start:job-retry-configuration]
def test_job_retry_configuration():
    """Job with retry configuration."""
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="api.call",
        payload={"endpoint": "/users"},
        max_retries=5,  # Retry up to 5 times
        retry_count=0,
    )

    assert job.max_retries == 5
    assert job.retry_count == 0

    # Simulate a retry
    retried_job = replace(job, retry_count=1, status=JobStatus.RETRYING)

    assert retried_job.retry_count == 1
    assert retried_job.status == JobStatus.RETRYING


# --8<-- [end:job-retry-configuration]


# --8<-- [start:job-handler-interface]
@pytest.mark.asyncio
async def test_job_handler_interface():
    """Job handler interface implementation."""

    class EmailSendJob(JobHandler):
        """Send email job handler."""

        def __init__(self, email_service):
            self.email_service = email_service

        @property
        def job_type(self) -> str:
            return "email.send"

        async def handle(self, job: Job) -> JobResult:
            """Process email send job."""
            to = job.payload["to"]
            subject = job.payload["subject"]
            body = job.payload["body"]

            try:
                await self.email_service.send(to=to, subject=subject, body=body)
                return JobResult(
                    success=True,
                    result_data={"sent_at": datetime.now(UTC).isoformat()},
                )
            except Exception as e:
                return JobResult(success=False, error=e)

        async def on_failure(self, job: Job, error: Exception) -> None:
            """Handle job failure."""
            # Could log, send alerts, etc.

    # Mock email service
    email_service = AsyncMock()
    email_service.send = AsyncMock(return_value=None)

    # Create handler
    handler = EmailSendJob(email_service)

    # Verify job type
    assert handler.job_type == "email.send"

    # Create and handle job
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={
            "to": "user@example.com",
            "subject": "Test",
            "body": "Test email",
        },
    )

    result = await handler.handle(job)

    assert result.success is True
    assert "sent_at" in result.result_data
    email_service.send.assert_called_once()


# --8<-- [end:job-handler-interface]


# --8<-- [start:queue-stats]
def test_queue_stats():
    """Queue statistics."""
    stats = QueueStats(
        queue_name="default",
        pending_count=15,
        running_count=3,
        completed_count=127,
        failed_count=2,
        dead_letter_count=1,
    )

    assert stats.queue_name == "default"
    assert stats.pending_count == 15
    assert stats.running_count == 3
    assert stats.completed_count == 127
    assert stats.failed_count == 2
    assert stats.dead_letter_count == 1


# --8<-- [end:queue-stats]


# --8<-- [start:job-lifecycle-states]
def test_job_lifecycle_states():
    """Job lifecycle state transitions."""
    job_id = uuid4()

    # 1. Created - Pending
    job = Job(
        id=job_id,
        queue_name="default",
        job_type="task.process",
        payload={},
        status=JobStatus.PENDING,
        created_at=datetime.now(UTC),
    )
    assert job.status == JobStatus.PENDING

    # 2. Running
    job = replace(job, status=JobStatus.RUNNING, started_at=datetime.now(UTC))
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None

    # 3. Completed
    job = replace(job, status=JobStatus.COMPLETED, completed_at=datetime.now(UTC))
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None


# --8<-- [end:job-lifecycle-states]


# --8<-- [start:job-error-tracking]
def test_job_error_tracking():
    """Job error tracking."""
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="api.call",
        payload={},
        status=JobStatus.FAILED,
        error_message="Connection timeout after 30s",
        failed_at=datetime.now(UTC),
        retry_count=3,
    )

    assert job.status == JobStatus.FAILED
    assert job.error_message == "Connection timeout after 30s"
    assert job.failed_at is not None
    assert job.retry_count == 3


# --8<-- [end:job-error-tracking]


# --8<-- [start:job-dead-letter-queue]
def test_job_dead_letter_queue():
    """Job moved to dead letter queue after max retries."""
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="unreliable.task",
        payload={},
        status=JobStatus.DEAD_LETTER,
        max_retries=3,
        retry_count=3,  # Exhausted retries
        error_message="Failed after 3 retries",
        failed_at=datetime.now(UTC),
    )

    assert job.status == JobStatus.DEAD_LETTER
    assert job.retry_count == job.max_retries
    assert job.error_message is not None


# --8<-- [end:job-dead-letter-queue]


# --8<-- [start:multiple-queue-names]
def test_multiple_queue_names():
    """Jobs in different queues."""
    # Default queue for regular tasks
    default_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="email.send",
        payload={},
    )

    # High priority queue
    critical_job = Job(
        id=uuid4(),
        queue_name="critical",
        job_type="alert.send",
        payload={},
    )

    # Background processing queue
    background_job = Job(
        id=uuid4(),
        queue_name="background",
        job_type="report.generate",
        payload={},
    )

    assert default_job.queue_name == "default"
    assert critical_job.queue_name == "critical"
    assert background_job.queue_name == "background"


# --8<-- [end:multiple-queue-names]


# --8<-- [start:job-payload-patterns]
def test_job_payload_patterns():
    """Common job payload patterns."""
    # Email payload
    email_payload = {
        "to": "user@example.com",
        "subject": "Welcome",
        "body": "Thanks for signing up",
        "template_id": "welcome_v2",
    }

    # Document processing payload
    doc_payload = {
        "document_id": str(uuid4()),
        "operation": "extract_text",
        "options": {"language": "en", "ocr": True},
    }

    # API call payload
    api_payload = {
        "method": "POST",
        "endpoint": "/api/webhooks",
        "body": {"event": "user.created", "data": {}},
        "headers": {"Authorization": "Bearer token"},
    }

    # Report generation payload
    report_payload = {
        "report_type": "monthly_analytics",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "format": "pdf",
        "recipients": ["admin@example.com"],
    }

    job1 = Job(
        id=uuid4(), queue_name="default", job_type="email.send", payload=email_payload
    )
    job2 = Job(
        id=uuid4(), queue_name="default", job_type="doc.process", payload=doc_payload
    )
    job3 = Job(
        id=uuid4(), queue_name="default", job_type="api.call", payload=api_payload
    )
    job4 = Job(
        id=uuid4(), queue_name="default", job_type="report.gen", payload=report_payload
    )

    assert job1.payload["to"] == "user@example.com"
    assert job2.payload["operation"] == "extract_text"
    assert job3.payload["method"] == "POST"
    assert job4.payload["format"] == "pdf"


# --8<-- [end:job-payload-patterns]


# --8<-- [start:job-scheduled-future]
def test_job_scheduled_future():
    """Job scheduled for future execution."""
    future_time = datetime.now(UTC) + timedelta(days=7)

    job = Job(
        id=uuid4(),
        queue_name="scheduled",
        job_type="reminder.send",
        payload={"message": "Weekly report due"},
        status=JobStatus.SCHEDULED,
        scheduled_at=future_time,
    )

    assert job.status == JobStatus.SCHEDULED
    assert job.scheduled_at > datetime.now(UTC)
    assert (job.scheduled_at - datetime.now(UTC)).days >= 6


# --8<-- [end:job-scheduled-future]


# --8<-- [start:job-immutability]
def test_job_immutability():
    """Jobs are immutable - use replace() to modify."""
    original_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="task",
        payload={},
        status=JobStatus.PENDING,
    )

    # Create modified copy
    updated_job = replace(
        original_job,
        status=JobStatus.RUNNING,
        started_at=datetime.now(UTC),
    )

    # Original is unchanged
    assert original_job.status == JobStatus.PENDING
    assert original_job.started_at is None

    # New instance has updates
    assert updated_job.status == JobStatus.RUNNING
    assert updated_job.started_at is not None

    # Both share same ID
    assert original_job.id == updated_job.id


# --8<-- [end:job-immutability]


# --8<-- [start:job-type-naming]
def test_job_type_naming():
    """Job type naming conventions."""
    # Dot notation for namespacing
    email_job = Job(id=uuid4(), queue_name="default", job_type="email.send", payload={})
    doc_job = Job(
        id=uuid4(), queue_name="default", job_type="document.process", payload={}
    )
    report_job = Job(
        id=uuid4(), queue_name="default", job_type="report.generate", payload={}
    )

    # Action-oriented names
    cleanup_job = Job(
        id=uuid4(), queue_name="default", job_type="cleanup.old_files", payload={}
    )
    sync_job = Job(
        id=uuid4(), queue_name="default", job_type="sync.external_data", payload={}
    )
    notify_job = Job(
        id=uuid4(), queue_name="default", job_type="notify.admin", payload={}
    )

    assert "." in email_job.job_type
    assert "." in doc_job.job_type
    assert email_job.job_type.startswith("email")
    assert doc_job.job_type.startswith("document")


# --8<-- [end:job-type-naming]


# --8<-- [start:job-created-by-tracking]
def test_job_created_by_tracking():
    """Track which user created a job."""
    user_id = uuid4()

    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="export.data",
        payload={"format": "csv"},
        created_by=user_id,
        created_at=datetime.now(UTC),
    )

    assert job.created_by == user_id
    assert job.created_at is not None


# --8<-- [end:job-created-by-tracking]


# --8<-- [start:job-default-values]
def test_job_default_values():
    """Job default values."""
    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="task",
        payload={},
    )

    # Default values
    assert job.priority == 0
    assert job.max_retries == 3
    assert job.retry_count == 0
    assert job.status == JobStatus.PENDING
    assert job.scheduled_at is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.failed_at is None
    assert job.error_message is None
    assert job.created_by is None
    assert job.created_at is None
    assert job.metadata == {}


# --8<-- [end:job-default-values]


# --8<-- [start:queue-stats-calculations]
def test_queue_stats_calculations():
    """Calculate queue health metrics."""
    stats = QueueStats(
        queue_name="default",
        pending_count=50,
        running_count=10,
        completed_count=1000,
        failed_count=25,
        dead_letter_count=5,
    )

    # Calculate metrics
    total_processed = (
        stats.completed_count + stats.failed_count + stats.dead_letter_count
    )
    success_rate = stats.completed_count / total_processed if total_processed > 0 else 0
    failure_rate = (
        (stats.failed_count + stats.dead_letter_count) / total_processed
        if total_processed > 0
        else 0
    )

    assert total_processed == 1030
    assert success_rate > 0.97  # ~97% success rate
    assert failure_rate < 0.03  # ~3% failure rate


# --8<-- [end:queue-stats-calculations]


# --8<-- [start:job-result-with-data]
def test_job_result_with_data():
    """Job result with detailed result data."""
    result = JobResult(
        success=True,
        result_data={
            "processed_count": 150,
            "skipped_count": 3,
            "duration_ms": 1234,
            "output_file": "/tmp/report.pdf",
            "checksum": "abc123",
        },
    )

    assert result.success is True
    assert result.result_data["processed_count"] == 150
    assert result.result_data["skipped_count"] == 3
    assert result.result_data["output_file"] == "/tmp/report.pdf"


# --8<-- [end:job-result-with-data]


# --8<-- [start:job-timestamp-tracking]
def test_job_timestamp_tracking():
    """Track job execution timestamps."""
    created = datetime.now(UTC)
    started = created + timedelta(seconds=10)
    completed = started + timedelta(seconds=30)

    job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="task",
        payload={},
        created_at=created,
        started_at=started,
        completed_at=completed,
        status=JobStatus.COMPLETED,
    )

    # Calculate execution time
    execution_time = (job.completed_at - job.started_at).total_seconds()
    wait_time = (job.started_at - job.created_at).total_seconds()

    assert execution_time == 30
    assert wait_time == 10


# --8<-- [end:job-timestamp-tracking]


# --8<-- [start:job-handler-error-handling]
@pytest.mark.asyncio
async def test_job_handler_error_handling():
    """Job handler with error handling."""

    class ResilientHandler(JobHandler):
        @property
        def job_type(self) -> str:
            return "resilient.task"

        async def handle(self, job: Job) -> JobResult:
            try:
                # Simulated work that might fail
                if job.payload.get("fail"):
                    raise ValueError("Simulated error")

                return JobResult(success=True, result_data={"status": "ok"})

            except Exception as e:
                # Return failure result instead of raising
                return JobResult(
                    success=False,
                    error=e,
                )

        async def on_failure(self, job: Job, error: Exception) -> None:
            # Clean up resources, log, etc.
            pass

    handler = ResilientHandler()

    # Success case
    success_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="resilient.task",
        payload={"fail": False},
    )
    result = await handler.handle(success_job)
    assert result.success is True

    # Failure case
    fail_job = Job(
        id=uuid4(),
        queue_name="default",
        job_type="resilient.task",
        payload={"fail": True},
    )
    result = await handler.handle(fail_job)
    assert result.success is False
    assert isinstance(result.error, ValueError)


# --8<-- [end:job-handler-error-handling]
