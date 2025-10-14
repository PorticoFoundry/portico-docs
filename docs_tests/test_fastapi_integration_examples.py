"""Test examples for FastAPI Integration documentation.

This module tests code examples from FastAPI-related documentation to ensure
they remain accurate and working.

Note: These tests focus on the API contracts and types rather than full
integration testing which would require complex async setup.
"""

from unittest.mock import MagicMock

from portico.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    PorticoError,
    RateLimitError,
    ResourceNotFoundError,
    ValidationError,
)

# --8<-- [start:imports]
from portico.kits.fastapi import (
    CoreDependencies,
    create_all_dependencies,
    create_exception_handlers,
    register_exception_handlers,
)

# --8<-- [end:imports]


# --8<-- [start:exception-handler-registration]
def test_exception_handler_registration():
    """Register exception handlers with FastAPI."""
    from fastapi import FastAPI

    app = FastAPI()

    # Register all Portico exception handlers
    register_exception_handlers(app)

    # Verify handlers are registered
    assert len(app.exception_handlers) > 0
    assert PorticoError in app.exception_handlers


# --8<-- [end:exception-handler-registration]


# --8<-- [start:create-exception-handlers]
def test_create_exception_handlers():
    """Create exception handlers dictionary."""
    handlers = create_exception_handlers()

    # Verify all expected handlers exist
    assert AuthenticationError in handlers
    assert AuthorizationError in handlers
    assert ResourceNotFoundError in handlers
    assert ValidationError in handlers
    assert ConflictError in handlers
    assert RateLimitError in handlers
    assert PorticoError in handlers

    # Handlers should be callables
    assert callable(handlers[AuthenticationError])
    assert callable(handlers[PorticoError])


# --8<-- [end:create-exception-handlers]


# --8<-- [start:core-dependencies-class]
def test_core_dependencies_class():
    """CoreDependencies factory class."""
    mock_core = MagicMock()
    mock_core.db = MagicMock()

    # Create dependencies
    deps = CoreDependencies(mock_core)

    # Verify it's a CoreDependencies instance
    assert isinstance(deps, CoreDependencies)
    assert deps.core == mock_core

    # Verify it has expected methods
    assert hasattr(deps, "get_session")
    assert hasattr(deps, "get_current_user")
    assert hasattr(deps, "get_optional_user")
    assert callable(deps.get_session)
    assert callable(deps.get_current_user)


# --8<-- [end:core-dependencies-class]


# --8<-- [start:create-all-dependencies]
def test_create_all_dependencies():
    """Create all dependencies with authentication."""
    mock_core = MagicMock()
    mock_core.db = MagicMock()
    mock_core.db.session_manager = MagicMock()

    # Create dependencies with auth
    dependencies = create_all_dependencies(mock_core, include_auth=True)

    # Verify all expected dependencies exist
    assert "get_session" in dependencies
    assert "get_current_user" in dependencies
    assert "get_optional_user" in dependencies
    assert "core" in dependencies

    # Core should be accessible
    assert dependencies["core"] == mock_core


# --8<-- [end:create-all-dependencies]


# --8<-- [start:dependencies-without-auth]
def test_dependencies_without_auth():
    """Dependencies without authentication."""
    mock_core = MagicMock()
    mock_core.db = MagicMock()
    mock_core.db.session_manager = MagicMock()

    # Create dependencies without auth
    dependencies = create_all_dependencies(mock_core, include_auth=False)

    # Should have session and core, but not user dependencies
    assert "get_session" in dependencies
    assert "core" in dependencies
    assert "get_current_user" not in dependencies
    assert "get_optional_user" not in dependencies


# --8<-- [end:dependencies-without-auth]


# --8<-- [start:exception-inheritance]
def test_exception_inheritance():
    """Portico exceptions inherit from PorticoError."""
    # All specific exceptions inherit from PorticoError
    assert issubclass(AuthenticationError, PorticoError)
    assert issubclass(AuthorizationError, PorticoError)
    assert issubclass(ResourceNotFoundError, PorticoError)
    assert issubclass(ValidationError, PorticoError)
    assert issubclass(ConflictError, PorticoError)
    assert issubclass(RateLimitError, PorticoError)


# --8<-- [end:exception-inheritance]


# --8<-- [start:exception-handler-types]
def test_exception_handler_types():
    """Exception handlers are async callables."""
    handlers = create_exception_handlers()

    # All handlers should be async functions
    import inspect

    for exc_type, handler in handlers.items():
        assert callable(handler)
        # Handler should be async
        assert inspect.iscoroutinefunction(handler)


# --8<-- [end:exception-handler-types]


# --8<-- [start:authentication-error-creation]
def test_authentication_error_creation():
    """Creating AuthenticationError."""
    error = AuthenticationError("Authentication required")

    assert isinstance(error, AuthenticationError)
    assert isinstance(error, PorticoError)
    assert str(error) == "Authentication required"


# --8<-- [end:authentication-error-creation]


# --8<-- [start:authorization-error-creation]
def test_authorization_error_creation():
    """Creating AuthorizationError."""
    error = AuthorizationError("Permission denied")

    assert isinstance(error, AuthorizationError)
    assert isinstance(error, PorticoError)
    assert str(error) == "Permission denied"


# --8<-- [end:authorization-error-creation]


# --8<-- [start:resource-not-found-error-creation]
def test_resource_not_found_error_creation():
    """Creating ResourceNotFoundError."""
    error = ResourceNotFoundError("User not found")

    assert isinstance(error, ResourceNotFoundError)
    assert isinstance(error, PorticoError)
    assert str(error) == "User not found"


# --8<-- [end:resource-not-found-error-creation]


# --8<-- [start:validation-error-creation]
def test_validation_error_creation():
    """Creating ValidationError."""
    error = ValidationError("Invalid input data")

    assert isinstance(error, ValidationError)
    assert isinstance(error, PorticoError)
    assert str(error) == "Invalid input data"


# --8<-- [end:validation-error-creation]


# --8<-- [start:conflict-error-creation]
def test_conflict_error_creation():
    """Creating ConflictError."""
    error = ConflictError("Email already exists")

    assert isinstance(error, ConflictError)
    assert isinstance(error, PorticoError)
    assert str(error) == "Email already exists"


# --8<-- [end:conflict-error-creation]


# --8<-- [start:rate-limit-error-creation]
def test_rate_limit_error_creation():
    """Creating RateLimitError."""
    error = RateLimitError("Rate limit exceeded")

    assert isinstance(error, RateLimitError)
    assert isinstance(error, PorticoError)
    assert str(error) == "Rate limit exceeded"


# --8<-- [end:rate-limit-error-creation]


# --8<-- [start:exception-with-details]
def test_exception_with_details():
    """Exception with additional details."""
    error = ResourceNotFoundError(
        "User not found",
        details={"user_id": "123", "action": "fetch"},
    )

    assert isinstance(error, ResourceNotFoundError)
    assert str(error) == "User not found"
    assert error.details == {"user_id": "123", "action": "fetch"}


# --8<-- [end:exception-with-details]


# --8<-- [start:portico-error-base]
def test_portico_error_base():
    """PorticoError as base exception."""
    error = PorticoError("Generic error message")

    assert isinstance(error, PorticoError)
    assert isinstance(error, Exception)
    assert str(error) == "Generic error message"


# --8<-- [end:portico-error-base]


# --8<-- [start:core-dependencies-methods]
def test_core_dependencies_methods():
    """CoreDependencies provides dependency factory methods."""
    from fastapi.params import Depends as DependsType

    mock_core = MagicMock()
    mock_core.db = MagicMock()

    deps = CoreDependencies(mock_core)

    # Get dependency factories
    session_dep = deps.get_session()
    user_dep = deps.get_current_user()
    optional_user_dep = deps.get_optional_user()

    # All should be Depends instances (wrapped callables)
    assert isinstance(session_dep, DependsType)
    assert isinstance(user_dep, DependsType)
    assert isinstance(optional_user_dep, DependsType)


# --8<-- [end:core-dependencies-methods]


# --8<-- [start:handler-registration-completeness]
def test_handler_registration_completeness():
    """All Portico exceptions have handlers."""
    from fastapi import FastAPI

    app = FastAPI()
    register_exception_handlers(app)

    # Check that all exception types are registered
    assert AuthenticationError in app.exception_handlers
    assert AuthorizationError in app.exception_handlers
    assert ResourceNotFoundError in app.exception_handlers
    assert ValidationError in app.exception_handlers
    assert ConflictError in app.exception_handlers
    assert RateLimitError in app.exception_handlers
    assert PorticoError in app.exception_handlers


# --8<-- [end:handler-registration-completeness]


# --8<-- [start:dependencies-structure]
def test_dependencies_structure():
    """Dependencies dictionary structure."""
    mock_core = MagicMock()
    mock_core.db = MagicMock()
    mock_core.db.session_manager = MagicMock()

    dependencies = create_all_dependencies(mock_core, include_auth=True)

    # Should be a dictionary
    assert isinstance(dependencies, dict)

    # Should have string keys
    for key in dependencies.keys():
        assert isinstance(key, str)

    # Core value should be the core instance
    assert dependencies["core"] == mock_core


# --8<-- [end:dependencies-structure]


# --8<-- [start:exception-error-codes]
def test_exception_error_codes():
    """Exceptions have error_code attributes."""
    # Each exception type has an error_code
    auth_error = AuthenticationError("test")
    assert hasattr(auth_error, "error_code")

    authz_error = AuthorizationError("test")
    assert hasattr(authz_error, "error_code")

    not_found = ResourceNotFoundError("test")
    assert hasattr(not_found, "error_code")

    validation = ValidationError("test")
    assert hasattr(validation, "error_code")


# --8<-- [end:exception-error-codes]


# --8<-- [start:core-dependencies-core-access]
def test_core_dependencies_core_access():
    """Accessing core instance from CoreDependencies."""
    mock_core = MagicMock()
    mock_core.db = MagicMock()

    deps = CoreDependencies(mock_core)

    # Core should be accessible
    assert deps.core is mock_core


# --8<-- [end:core-dependencies-core-access]


# --8<-- [start:exception-handlers-dict-keys]
def test_exception_handlers_dict_keys():
    """Exception handlers dictionary uses exception classes as keys."""
    handlers = create_exception_handlers()

    # Keys should be exception classes
    for key in handlers.keys():
        assert isinstance(key, type)
        assert issubclass(key, Exception)


# --8<-- [end:exception-handlers-dict-keys]
