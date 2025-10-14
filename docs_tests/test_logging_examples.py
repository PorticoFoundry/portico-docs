"""Test examples for Logging documentation.

These tests verify that all code examples in docs/reference/logging.md work correctly.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from portico.utils.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    log_context,
    reset_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_after_test():
    """Reset logging configuration after each test."""
    yield
    reset_logging()
    clear_context()


# --8<-- [start:manual-configuration]
def test_manual_configuration():
    """Manual logging configuration."""
    configure_logging(
        log_level="INFO",
        log_format="console",
        log_output="stdout",
        log_file_path=None,
    )

    logger = get_logger(__name__)
    logger.info("test_event", key="value")


# --8<-- [end:manual-configuration]


# --8<-- [start:get-logger]
def test_get_logger():
    """Get logger and log with structured fields."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    # Log with structured fields
    user = Mock(id=uuid4())
    cache_key = "user:123"
    e = Exception("Test error")
    query = "SELECT * FROM users"

    logger.info("user_created", user_id=str(user.id), email="user@example.com")
    logger.debug("cache_hit", key=cache_key, ttl=300)
    logger.warning("rate_limit_approaching", remaining=10)
    logger.error("database_error", error=str(e), query=query)


# --8<-- [end:get-logger]


# --8<-- [start:log-level-debug]
def test_log_level_debug():
    """DEBUG level logging examples."""
    configure_logging(log_level="DEBUG")

    logger = get_logger(__name__)
    key = "user:123"
    namespace = "users"
    user_id = uuid4()
    perm = "write"
    stmt = "SELECT * FROM users WHERE id = ?"
    params = {"id": user_id}

    logger.debug("cache_hit", key=key, namespace=namespace)
    logger.debug("permission_check", user_id=str(user_id), permission=perm)
    logger.debug("sql_query", query=stmt, params=params)


# --8<-- [end:log-level-debug]


# --8<-- [start:log-level-info]
def test_log_level_info():
    """INFO level logging examples."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    user = Mock(id=uuid4())
    job = Mock(id=uuid4())
    duration = 1500

    logger.info("user_created", user_id=str(user.id), email="user@example.com")
    logger.info("api_request", method="POST", path="/users", status=201)
    logger.info("job_completed", job_id=str(job.id), duration_ms=duration)


# --8<-- [end:log-level-info]


# --8<-- [start:log-level-warning]
def test_log_level_warning():
    """WARNING level logging examples."""
    configure_logging(log_level="WARNING")

    logger = get_logger(__name__)
    user = Mock(id=uuid4())

    logger.warning("rate_limit_approaching", user_id=str(user.id), remaining=10)
    logger.warning("cache_eviction", evicted_count=100)
    logger.warning("retry_attempt", attempt=2, max_retries=3)


# --8<-- [end:log-level-warning]


# --8<-- [start:log-level-error]
def test_log_level_error():
    """ERROR level logging examples."""
    configure_logging(log_level="ERROR")

    logger = get_logger(__name__)
    e = Exception("Connection failed")
    validation_errors = [{"field": "email", "message": "Invalid format"}]

    logger.error("database_error", error=str(e), operation="create_user")
    logger.error("api_timeout", service="OpenAI", timeout=30)
    logger.error("validation_failed", errors=validation_errors)


# --8<-- [end:log-level-error]


# --8<-- [start:log-level-critical]
def test_log_level_critical():
    """CRITICAL level logging examples."""
    configure_logging(log_level="CRITICAL")

    logger = get_logger(__name__)
    e = Exception("Connection lost")

    logger.critical("database_connection_lost", error=str(e))
    logger.critical("redis_unavailable", error=str(e))


# --8<-- [end:log-level-critical]


# --8<-- [start:context-async]
@pytest.mark.asyncio
async def test_manual_context_binding():
    """Manual context binding with async."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    async def do_work():
        logger.info("working")
        await asyncio.sleep(0.01)

    async def handle_request(request_id: str, user_id: UUID):
        # All logs within this context include request_id and user_id
        async with log_context(request_id=request_id, user_id=str(user_id)):
            logger.info("processing_request")
            await do_work()
            logger.info("request_completed")

    await handle_request("abc123", uuid4())


# --8<-- [end:context-async]


# --8<-- [start:context-manager]
def test_context_manager():
    """Context manager for binding and clearing context."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    request_id = "req-123"
    user = Mock(id=uuid4())

    # Bind context
    bind_context(request_id=request_id, user_id=str(user.id))

    # All subsequent logs include context
    logger.info("action_performed")

    # Clear context when done
    clear_context()


# --8<-- [end:context-manager]


# --8<-- [start:event-naming-good]
def test_event_naming_good():
    """Good event naming examples."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    # Good - Descriptive, past tense
    logger.info("user_created")
    logger.info("payment_processed")
    logger.info("document_uploaded")

    # Good - Present continuous for ongoing
    logger.info("processing_document")
    logger.info("generating_report")


# --8<-- [end:event-naming-good]


# --8<-- [start:structured-fields-good]
def test_structured_fields_good():
    """Good structured fields example."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    user = Mock(id=uuid4())
    request = Mock(client=Mock(host="192.168.1.1"), headers={"user-agent": "Mozilla"})

    # Good - Structured fields
    logger.info(
        "user_login",
        user_id=str(user.id),
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
    )


# --8<-- [end:structured-fields-good]


# --8<-- [start:error-logging]
@pytest.mark.asyncio
async def test_error_logging():
    """Error logging pattern."""
    from portico.exceptions import DatabaseError

    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    repository = Mock()
    request = Mock(email="test@example.com")

    # Simulate error
    repository.create_user = Mock(side_effect=DatabaseError("Connection failed"))

    try:
        user = await repository.create_user(request)
        logger.info("user_created", user_id=str(user.id), email=user.email)

    except DatabaseError as e:
        logger.error(
            "user_creation_failed",
            email=request.email,
            error=str(e),
            error_type=type(e).__name__,
        )


# --8<-- [end:error-logging]


# --8<-- [start:performance-monitoring]
@pytest.mark.asyncio
async def test_performance_monitoring():
    """Performance monitoring pattern."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    async def expensive_operation():
        await asyncio.sleep(0.01)
        return ["result1", "result2", "result3"]

    start_time = time.time()

    result = await expensive_operation()

    duration_ms = (time.time() - start_time) * 1000

    logger.info(
        "operation_completed",
        operation="expensive_operation",
        duration_ms=int(duration_ms),
        result_count=len(result),
    )


# --8<-- [end:performance-monitoring]


# --8<-- [start:audit-logging]
def test_audit_logging():
    """Audit logging pattern."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    user = Mock(id=uuid4())
    document = Mock(id=uuid4())
    request = Mock(client=Mock(host="192.168.1.1"))

    logger.info(
        "audit_event",
        user_id=str(user.id),
        action="document_deleted",
        resource_type="document",
        resource_id=str(document.id),
        ip_address=request.client.host,
    )


# --8<-- [end:audit-logging]


# --8<-- [start:cache-operations]
def test_cache_operations():
    """Cache operations logging."""
    configure_logging(log_level="DEBUG")

    logger = get_logger(__name__)
    key = "user:123"
    namespace = "users"
    ttl = 300
    count = 50

    # Cache hit
    logger.debug(
        "cache_hit",
        key=key,
        namespace=namespace,
        ttl=ttl,
    )

    # Cache miss
    logger.debug(
        "cache_miss",
        key=key,
        namespace=namespace,
    )

    # Cache eviction
    logger.warning(
        "cache_eviction",
        namespace=namespace,
        evicted_count=count,
        reason="memory_limit",
    )


# --8<-- [end:cache-operations]


# --8<-- [start:external-service-calls]
@pytest.mark.asyncio
async def test_external_service_calls():
    """External service call logging."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    logger.info(
        "external_api_request",
        service="OpenAI",
        model="gpt-4",
        endpoint="/chat/completions",
    )

    # Mock response
    response = Mock(model="gpt-4", usage=Mock(total_tokens=150))
    duration = 250

    try:
        # Simulate API call
        await asyncio.sleep(0.01)

        logger.info(
            "external_api_response",
            service="OpenAI",
            model=response.model,
            tokens_used=response.usage.total_tokens,
            duration_ms=duration,
        )

    except Exception as e:
        logger.error(
            "external_api_error",
            service="OpenAI",
            error=str(e),
            error_type=type(e).__name__,
        )


# --8<-- [end:external-service-calls]


# --8<-- [start:best-practice-structured]
def test_best_practice_structured():
    """Best practice: use structured fields."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    user = Mock(id=uuid4())
    ip_address = "192.168.1.1"

    # Good
    logger.info("user_login", user_id=str(user.id), ip=ip_address)


# --8<-- [end:best-practice-structured]


# --8<-- [start:best-practice-semantic]
def test_best_practice_semantic():
    """Best practice: use semantic event names."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    # Good
    logger.info("payment_processed", amount=100.00, currency="USD")


# --8<-- [end:best-practice-semantic]


# --8<-- [start:best-practice-context]
def test_best_practice_context():
    """Best practice: include relevant context."""
    configure_logging(log_level="ERROR")

    logger = get_logger(__name__)
    e = Exception("Internal server error")

    # Good
    logger.error(
        "api_request_failed",
        endpoint="/users",
        status_code=500,
        retry_count=3,
        error=str(e),
    )


# --8<-- [end:best-practice-context]


# --8<-- [start:best-practice-levels]
def test_best_practice_levels():
    """Best practice: use appropriate log levels."""
    configure_logging(log_level="DEBUG")

    logger = get_logger(__name__)
    key = "user:123"
    id = uuid4()
    e = Exception("Connection error")

    # Good
    logger.debug("cache_hit", key=key)  # Verbose, frequent
    logger.info("user_created", user_id=id)  # Important events
    logger.warning("rate_limit", remaining=5)  # Degraded service
    logger.error("database_error", error=e)  # Failures


# --8<-- [end:best-practice-levels]


# --8<-- [start:best-practice-no-sensitive]
def test_best_practice_no_sensitive():
    """Best practice: don't log sensitive data."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    email = "user@example.com"

    # Good - No sensitive data
    logger.info("user_created", email=email)


# --8<-- [end:best-practice-no-sensitive]


# --8<-- [start:best-practice-no-formatting]
def test_best_practice_no_formatting():
    """Best practice: don't use string formatting."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)
    user_id = uuid4()

    # Good - Structured
    logger.info("user_created", user_id=str(user_id))


# --8<-- [end:best-practice-no-formatting]


# --8<-- [start:best-practice-no-overlog]
def test_best_practice_no_overlog():
    """Best practice: don't over-log at high levels."""
    configure_logging(log_level="DEBUG")

    logger = get_logger(__name__)

    # Good - Use appropriate level
    logger.debug("page_viewed", page="/home")


# --8<-- [end:best-practice-no-overlog]


# --8<-- [start:best-practice-no-redundant]
def test_best_practice_no_redundant():
    """Best practice: don't log redundantly."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    # Good - Event name + structured fields
    logger.info("user_created", user_id="abc123")


# --8<-- [end:best-practice-no-redundant]


# --8<-- [start:config-development]
def test_config_development():
    """Development environment configuration."""
    configure_logging(
        log_level="DEBUG",  # Verbose logging
        log_format="console",  # Human-readable
        log_output="stdout",  # To terminal
    )

    logger = get_logger(__name__)
    logger.debug("test_event")


# --8<-- [end:config-development]


# --8<-- [start:config-production]
def test_config_production():
    """Production environment configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "app.log"

        configure_logging(
            log_level="INFO",  # Less verbose
            log_format="json",  # Machine-parseable
            log_output="both",  # Stdout + file
            log_file_path=str(log_file),
        )

        logger = get_logger(__name__)
        logger.info("test_event")

        assert log_file.exists()


# --8<-- [end:config-production]


# --8<-- [start:config-testing]
def test_config_testing():
    """Testing environment configuration."""
    configure_logging(
        log_level="ERROR",  # Quiet during tests
        log_format="console",
        log_output="stdout",
    )

    logger = get_logger(__name__)
    logger.error("test_error")


# --8<-- [end:config-testing]


# --8<-- [start:testing-caplog]
def test_logging_caplog(caplog):
    """Test that logging works with caplog."""
    configure_logging(log_level="INFO")

    logger = get_logger(__name__)

    with caplog.at_level(logging.INFO):
        logger.info("test_event", key="value")

    assert "test_event" in caplog.text
    assert "key" in caplog.text


# --8<-- [end:testing-caplog]


# --8<-- [start:testing-mock]
def test_with_mock_logger():
    """Test with mocked logger."""
    mock_logger = Mock()

    # Use mock
    mock_logger.info("event", key="value")

    # Verify
    mock_logger.info.assert_called_once_with("event", key="value")


# --8<-- [end:testing-mock]
