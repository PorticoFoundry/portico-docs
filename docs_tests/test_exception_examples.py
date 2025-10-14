"""Test examples for Exception documentation.

These tests verify that all code examples in docs/reference/exceptions.md work correctly.
"""

import asyncio
import pickle
import re
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from portico.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ConflictError,
    DatabaseError,
    ExternalServiceError,
    LLMError,
    PorticoError,
    RateLimitError,
    ResourceNotFoundError,
    ValidationError,
)


# --8<-- [start:import-exceptions]
def test_import_exceptions():
    """Import all Portico exceptions."""
    from portico.exceptions import (
        AuthenticationError,
        AuthorizationError,
        PorticoError,
        ValidationError,
    )

    # Verify all exceptions are subclasses of PorticoError
    assert issubclass(ValidationError, PorticoError)
    assert issubclass(AuthenticationError, PorticoError)
    assert issubclass(AuthorizationError, PorticoError)


# --8<-- [end:import-exceptions]


# --8<-- [start:portico-error-usage]
def test_portico_error_usage():
    """PorticoError base class usage."""
    original_exception = Exception("Something failed")

    error = PorticoError(
        message="Something went wrong",
        details={"context": "value"},
        cause=original_exception,
    )

    assert str(error) == "Something went wrong"
    assert error.details == {"context": "value"}
    assert error.cause == original_exception


# --8<-- [end:portico-error-usage]


# --8<-- [start:validation-error-missing-field]
def test_validation_error_missing_field():
    """ValidationError for missing required field."""
    request = Mock(email=None)

    # Missing required field
    if not request.email:
        with pytest.raises(ValidationError, match="Email is required"):
            raise ValidationError("Email is required")


# --8<-- [end:validation-error-missing-field]


# --8<-- [start:validation-error-invalid-format]
def test_validation_error_invalid_format():
    """ValidationError for invalid format."""
    email = "invalid-email"

    # Invalid format
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        with pytest.raises(ValidationError, match="Invalid email format"):
            raise ValidationError("Invalid email format")


# --8<-- [end:validation-error-invalid-format]


# --8<-- [start:validation-error-business-rule]
def test_validation_error_business_rule():
    """ValidationError for business rule violation."""
    request = Mock(age=16)

    # Business rule
    if request.age < 18:
        with pytest.raises(ValidationError, match="Must be 18 or older"):
            raise ValidationError("Must be 18 or older")


# --8<-- [end:validation-error-business-rule]


# --8<-- [start:validation-error-with-details]
def test_validation_error_with_details():
    """ValidationError with detailed field errors."""
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError(
            message="Invalid user data",
            details={
                "errors": [
                    {"field": "email", "message": "Email is required"},
                    {"field": "username", "message": "Too short"},
                ]
            },
        )

    assert "Invalid user data" in str(exc_info.value)
    assert exc_info.value.details["errors"][0]["field"] == "email"


# --8<-- [end:validation-error-with-details]


# --8<-- [start:authentication-error-invalid-credentials]
@pytest.mark.asyncio
async def test_authentication_error_invalid_credentials():
    """AuthenticationError for invalid credentials."""
    email = "test@example.com"
    password = "wrong"

    async def authenticate(email, password):
        return None  # Simulate failed authentication

    # Invalid credentials
    user = await authenticate(email, password)
    if not user:
        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            raise AuthenticationError("Invalid email or password")


# --8<-- [end:authentication-error-invalid-credentials]


# --8<-- [start:authentication-error-missing-token]
def test_authentication_error_missing_token():
    """AuthenticationError for missing token."""
    token = None

    # Missing token
    if not token:
        with pytest.raises(AuthenticationError, match="Authentication required"):
            raise AuthenticationError("Authentication required")


# --8<-- [end:authentication-error-missing-token]


# --8<-- [start:authentication-error-expired-session]
def test_authentication_error_expired_session():
    """AuthenticationError for expired session."""
    session = Mock(is_expired=lambda: True)

    # Expired session
    if session.is_expired():
        with pytest.raises(AuthenticationError, match="Session expired"):
            raise AuthenticationError("Session expired. Please login again.")


# --8<-- [end:authentication-error-expired-session]


# --8<-- [start:authorization-error-missing-permission]
def test_authorization_error_missing_permission():
    """AuthorizationError for missing permission."""
    user = Mock(has_permission=lambda p: False)

    # Missing permission
    if not user.has_permission("documents.delete"):
        with pytest.raises(
            AuthorizationError, match="Permission denied: documents.delete required"
        ):
            raise AuthorizationError("Permission denied: documents.delete required")


# --8<-- [end:authorization-error-missing-permission]


# --8<-- [start:authorization-error-wrong-role]
def test_authorization_error_wrong_role():
    """AuthorizationError for wrong role."""
    user = Mock(role="user")

    # Wrong role
    if user.role != "admin":
        with pytest.raises(AuthorizationError, match="Admin access required"):
            raise AuthorizationError("Admin access required")


# --8<-- [end:authorization-error-wrong-role]


# --8<-- [start:authorization-error-resource-ownership]
def test_authorization_error_resource_ownership():
    """AuthorizationError for resource ownership."""
    user = Mock(id=uuid4())
    document = Mock(owner_id=uuid4())

    # Resource ownership
    if document.owner_id != user.id:
        with pytest.raises(
            AuthorizationError, match="Can only delete your own documents"
        ):
            raise AuthorizationError("Can only delete your own documents")


# --8<-- [end:authorization-error-resource-ownership]


# --8<-- [start:resource-not-found-by-id]
@pytest.mark.asyncio
async def test_resource_not_found_by_id():
    """ResourceNotFoundError for entity not found by ID."""
    user_id = uuid4()
    repository = Mock()
    repository.get_by_id = AsyncMock(return_value=None)

    # Not found by ID
    user = await repository.get_by_id(user_id)
    if not user:
        with pytest.raises(ResourceNotFoundError, match=f"User {user_id} not found"):
            raise ResourceNotFoundError(f"User {user_id} not found")


# --8<-- [end:resource-not-found-by-id]


# --8<-- [start:resource-not-found-by-email]
@pytest.mark.asyncio
async def test_resource_not_found_by_email():
    """ResourceNotFoundError for entity not found by email."""
    email = "test@example.com"
    repository = Mock()
    repository.get_by_email = AsyncMock(return_value=None)

    # Not found by email
    user = await repository.get_by_email(email)
    if not user:
        with pytest.raises(
            ResourceNotFoundError, match=f"User with email {email} not found"
        ):
            raise ResourceNotFoundError(f"User with email {email} not found")


# --8<-- [end:resource-not-found-by-email]


# --8<-- [start:conflict-error-duplicate-email]
@pytest.mark.asyncio
async def test_conflict_error_duplicate_email():
    """ConflictError for duplicate email."""
    email = "test@example.com"
    repository = Mock()
    repository.get_by_email = AsyncMock(return_value=Mock(email=email))

    # Duplicate email
    existing = await repository.get_by_email(email)
    if existing:
        with pytest.raises(ConflictError, match=f"Email {email} already exists"):
            raise ConflictError(f"Email {email} already exists")


# --8<-- [end:conflict-error-duplicate-email]


# --8<-- [start:conflict-error-state-transition]
def test_conflict_error_state_transition():
    """ConflictError for state transition conflict."""
    order = Mock(status="shipped")

    # State conflict
    if order.status == "shipped":
        with pytest.raises(ConflictError, match="Cannot cancel shipped order"):
            raise ConflictError("Cannot cancel shipped order")


# --8<-- [end:conflict-error-state-transition]


# --8<-- [start:conflict-error-optimistic-locking]
def test_conflict_error_optimistic_locking():
    """ConflictError for optimistic locking failure."""
    document = Mock(version=2)
    expected_version = 1

    # Optimistic locking
    if document.version != expected_version:
        with pytest.raises(ConflictError, match="Document was modified"):
            raise ConflictError(
                "Document was modified by another user. Please refresh and try again."
            )


# --8<-- [end:conflict-error-optimistic-locking]


# --8<-- [start:rate-limit-error-login-attempts]
def test_rate_limit_error_login_attempts():
    """RateLimitError for too many login attempts."""
    attempts = 10
    MAX_ATTEMPTS = 5
    lockout_minutes = 15

    # Login attempts
    if attempts > MAX_ATTEMPTS:
        with pytest.raises(RateLimitError, match="Too many login attempts"):
            raise RateLimitError(
                f"Too many login attempts. Try again in {lockout_minutes} minutes"
            )


# --8<-- [end:rate-limit-error-login-attempts]


# --8<-- [start:rate-limit-error-api-limit]
def test_rate_limit_error_api_limit():
    """RateLimitError for API rate limit."""
    requests_per_minute = 150
    LIMIT = 100

    # API rate limit
    if requests_per_minute > LIMIT:
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            raise RateLimitError("Rate limit exceeded. Please slow down.")


# --8<-- [end:rate-limit-error-api-limit]


# --8<-- [start:rate-limit-error-with-retry]
def test_rate_limit_error_with_retry():
    """RateLimitError with retry information."""
    with pytest.raises(RateLimitError) as exc_info:
        # With retry information
        raise RateLimitError(
            message="Rate limit exceeded",
            details={"retry_after": 60},  # Seconds
        )

    assert exc_info.value.details["retry_after"] == 60


# --8<-- [end:rate-limit-error-with-retry]


# --8<-- [start:database-error-connection]
@pytest.mark.asyncio
async def test_database_error_connection():
    """DatabaseError for connection failure."""
    from sqlalchemy.exc import OperationalError

    query = "SELECT 1"

    # Connection error
    try:
        # Simulate connection failure
        raise OperationalError("Connection refused", None, None)
    except OperationalError as e:
        with pytest.raises(DatabaseError, match="Database connection failed"):
            raise DatabaseError(f"Database connection failed: {e}", cause=e) from e


# --8<-- [end:database-error-connection]


# --8<-- [start:external-service-error-http]
@pytest.mark.asyncio
async def test_external_service_error_http():
    """ExternalServiceError for HTTP API failure."""

    class HTTPError(Exception):
        pass

    url = "https://api.payment.com"
    data = {"amount": 100}
    httpx_client = Mock()

    async def failing_post(url, json):
        raise HTTPError("Connection timeout")

    httpx_client.post = failing_post

    # HTTP API error
    try:
        response = await httpx_client.post(url, json=data)
        response.raise_for_status()
    except HTTPError as e:
        with pytest.raises(ExternalServiceError):
            raise ExternalServiceError(
                service_name="PaymentAPI",
                original_error=e,
            ) from e


# --8<-- [end:external-service-error-http]


# --8<-- [start:external-service-error-timeout]
@pytest.mark.asyncio
async def test_external_service_error_timeout():
    """ExternalServiceError for timeout."""
    external_api = Mock()

    async def timeout_call(timeout):
        await asyncio.sleep(0.01)
        raise asyncio.TimeoutError()

    external_api.call = timeout_call

    # Timeout
    try:
        response = await external_api.call(timeout=5.0)
    except asyncio.TimeoutError as e:
        with pytest.raises(ExternalServiceError):
            raise ExternalServiceError(
                service_name="ExternalAPI",
                original_error=e,
            ) from e


# --8<-- [end:external-service-error-timeout]


# --8<-- [start:external-service-error-unavailable]
def test_external_service_error_unavailable():
    """ExternalServiceError for service unavailable."""
    response = Mock(status_code=503)

    # Service unavailable
    if response.status_code == 503:
        with pytest.raises(ExternalServiceError):
            raise ExternalServiceError(
                service_name="ExternalAPI",
                original_error=Exception("Service temporarily unavailable"),
            )


# --8<-- [end:external-service-error-unavailable]


# --8<-- [start:llm-error-token-limit]
def test_llm_error_token_limit():
    """LLMError for token limit exceeded."""
    token_count = 10000
    MAX_TOKENS = 8000

    # Token limit
    if token_count > MAX_TOKENS:
        with pytest.raises(LLMError, match="Input too long"):
            raise LLMError(
                f"Input too long: {token_count} tokens exceeds limit of {MAX_TOKENS}"
            )


# --8<-- [end:llm-error-token-limit]


# --8<-- [start:cache-error-serialization]
@pytest.mark.asyncio
async def test_cache_error_serialization():
    """CacheError for serialization failure."""
    key = "test_key"

    # Create an object that can't be pickled
    class UnpicklableClass:
        def __getstate__(self):
            raise pickle.PickleError("Cannot pickle this object")

    value = UnpicklableClass()
    cache = Mock()

    # Serialization error
    try:
        serialized = pickle.dumps(value)
        await cache.set(key, serialized)
    except pickle.PickleError as e:
        with pytest.raises(CacheError, match="Failed to serialize value"):
            raise CacheError(f"Failed to serialize value: {e}") from e


# --8<-- [end:cache-error-serialization]


# --8<-- [start:cache-error-full]
def test_cache_error_full():
    """CacheError for cache full."""
    cache = Mock()
    cache.size = Mock(return_value=1000)
    MAX_SIZE = 1000

    # Cache full
    if cache.size() >= MAX_SIZE:
        with pytest.raises(CacheError, match="Cache is full"):
            raise CacheError("Cache is full. Cannot store more items.")


# --8<-- [end:cache-error-full]


# --8<-- [start:cache-error-invalid-key]
def test_cache_error_invalid_key():
    """CacheError for invalid key."""
    key = "x" * 300
    MAX_KEY_LENGTH = 255

    # Invalid key
    if len(key) > MAX_KEY_LENGTH:
        with pytest.raises(CacheError, match="Cache key too long"):
            raise CacheError(f"Cache key too long: {len(key)} > {MAX_KEY_LENGTH}")


# --8<-- [end:cache-error-invalid-key]


# --8<-- [start:best-practice-specific-exceptions]
def test_best_practice_specific_exceptions():
    """Best practice: use specific exceptions."""
    user = None
    user_id = uuid4()

    # Good
    if not user:
        with pytest.raises(ResourceNotFoundError, match=f"User {user_id} not found"):
            raise ResourceNotFoundError(f"User {user_id} not found")


# --8<-- [end:best-practice-specific-exceptions]


# --8<-- [start:best-practice-add-context]
def test_best_practice_add_context():
    """Best practice: add context with details."""
    with pytest.raises(ValidationError) as exc_info:
        # Good
        raise ValidationError(
            message="Invalid request",
            details={
                "errors": [
                    {"field": "email", "message": "Invalid format"},
                    {"field": "age", "message": "Must be 18+"},
                ]
            },
        )

    assert len(exc_info.value.details["errors"]) == 2


# --8<-- [end:best-practice-add-context]


# --8<-- [start:best-practice-preserve-stack]
@pytest.mark.asyncio
async def test_best_practice_preserve_stack():
    """Best practice: preserve stack trace with 'from'."""

    class OriginalError(Exception):
        pass

    async def operation():
        raise OriginalError("Original failure")

    # Good - preserves stack trace
    try:
        result = await operation()
    except OriginalError as e:
        with pytest.raises(PorticoError) as exc_info:
            raise PorticoError("Operation failed") from e  # Preserves stack

        assert exc_info.value.__cause__ is e


# --8<-- [end:best-practice-preserve-stack]


# --8<-- [start:testing-exceptions]
@pytest.mark.asyncio
async def test_testing_exceptions():
    """Test exceptions with pytest.raises."""
    from uuid import uuid4

    service = Mock()
    service.get_user = AsyncMock(side_effect=ResourceNotFoundError("User .* not found"))

    # Test getting non-existent user raises exception
    with pytest.raises(ResourceNotFoundError, match="User .* not found"):
        await service.get_user(uuid4())


# --8<-- [end:testing-exceptions]


# --8<-- [start:testing-validation-error]
@pytest.mark.asyncio
async def test_testing_validation_error():
    """Test validation error details."""
    from pydantic import BaseModel

    class CreateUserRequest(BaseModel):
        email: str

    service = Mock()
    service.create_user = AsyncMock(side_effect=ValidationError("Email is required"))

    # Test validation error
    with pytest.raises(ValidationError) as exc_info:
        await service.create_user(CreateUserRequest(email=""))

    assert "Email is required" in str(exc_info.value)


# --8<-- [end:testing-validation-error]
