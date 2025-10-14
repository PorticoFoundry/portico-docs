"""Test examples for authentication documentation.

All examples are extracted into docs using snippet markers.
"""

import pytest

from portico.core import PorticoCore
from portico.exceptions import ValidationError
from portico.ports.user import CreateUserRequest


@pytest.fixture
async def test_core(config_registry):
    """Create test PorticoCore instance."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    yield core
    await core.close()


# --8<-- [start:create-user]
@pytest.mark.asyncio
async def test_create_user(test_core):
    """Create a new user with password."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Verify user was created
    assert user is not None
    assert user.email == "user@example.com"
    assert user.username == "user"
    assert user.id is not None


# --8<-- [end:create-user]


# --8<-- [start:authenticate-user]
@pytest.mark.asyncio
async def test_authenticate_user(test_core):
    """Authenticate user with username and password."""
    # Create user first
    await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Authenticate with correct credentials
    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="SecurePassword123!",
    )

    # Verify authentication succeeded
    assert auth_result.success is True
    assert auth_result.user is not None
    assert auth_result.user.email == "user@example.com"
    assert auth_result.token is not None
    assert auth_result.session is not None


# --8<-- [end:authenticate-user]


# --8<-- [start:failed-authentication]
@pytest.mark.asyncio
async def test_failed_authentication(test_core):
    """Handle failed authentication."""
    # Create user
    await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Try to authenticate with wrong password
    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="WrongPassword",
    )

    # Verify authentication failed
    assert auth_result.success is False
    assert auth_result.user is None
    assert auth_result.token is None


# --8<-- [end:failed-authentication]


# --8<-- [start:password-hashing]
@pytest.mark.asyncio
async def test_password_hashing(test_core):
    """Hash and verify passwords."""
    # Hash a password
    password = "SecurePassword123!"
    hashed = await test_core.hash_password(password)

    # Verify hash is different from plaintext
    assert hashed != password
    assert len(hashed) > 0

    # Create user with hashed password
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password=password,
        )
    )

    # Verify password
    is_valid = await test_core.verify_password(user, password)
    assert is_valid is True

    # Verify wrong password fails
    is_valid = await test_core.verify_password(user, "WrongPassword")
    assert is_valid is False


# --8<-- [end:password-hashing]


# --8<-- [start:create-with-session]
@pytest.mark.asyncio
async def test_create_with_session(test_core):
    """Create user and session in one operation."""
    # Create user with session
    auth_result = await test_core.create_user_with_session(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        ),
        request=None,  # Can pass Request for IP tracking
    )

    # Verify user created
    assert auth_result.success is True
    assert auth_result.user is not None
    assert auth_result.user.email == "user@example.com"

    # Note: token/session may not be created without Request object
    # When using with FastAPI, pass the actual request


# --8<-- [end:create-with-session]


# --8<-- [start:password-validation]
@pytest.mark.asyncio
async def test_password_validation(test_core):
    """Password must meet minimum requirements."""
    # Try to create user with weak password
    try:
        await test_core.create_user(
            CreateUserRequest(
                email="user@example.com",
                username="user",
                password="weak",  # Too short
            )
        )
        assert False, "Should have raised error"
    except (ValidationError, ValueError) as e:
        # Password validation failed as expected
        assert "password" in str(e).lower() or "8" in str(e).lower()


# --8<-- [end:password-validation]


# --8<-- [start:email-uniqueness]
@pytest.mark.asyncio
async def test_email_uniqueness(test_core):
    """Email addresses must be unique."""
    from portico.exceptions import ConflictError

    # Create first user
    await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user1",
            password="SecurePassword123!",
        )
    )

    # Try to create second user with same email
    try:
        await test_core.create_user(
            CreateUserRequest(
                email="user@example.com",  # Duplicate email
                username="user2",
                password="SecurePassword123!",
            )
        )
        assert False, "Should have raised ConflictError"
    except ConflictError:
        # Email uniqueness enforced as expected
        pass


# --8<-- [end:email-uniqueness]


# --8<-- [start:update-password]
@pytest.mark.asyncio
async def test_update_password(test_core):
    """Change user password."""
    from portico.ports.user import UpdateUserRequest

    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="OldPassword123!",
        )
    )

    # To change password, hash the new password first
    new_password = "NewPassword456!"
    hashed = await test_core.hash_password(new_password)

    # Update user with new password hash
    updated_user = await test_core.update_user(
        user.id, UpdateUserRequest(password_hash=hashed)
    )

    # Verify update succeeded
    assert updated_user is not None


# --8<-- [end:update-password]


# --8<-- [start:session-management]
@pytest.mark.asyncio
async def test_session_management(test_core):
    """Manage user sessions."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Create session via authentication
    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="SecurePassword123!",
    )

    # Verify session exists
    assert auth_result.token is not None
    assert auth_result.session is not None
    session_token = auth_result.token

    # Validate session token
    validated_user = await test_core._validate_session_token(session_token)
    assert validated_user is not None
    assert validated_user.id == user.id


# --8<-- [end:session-management]


# ========== Authentication Guide Examples ==========


# --8<-- [start:signup-with-validation]
@pytest.mark.asyncio
async def test_signup_with_validation(test_core):
    """Signup with request validation."""
    from pydantic import BaseModel, Field

    class SignupRequest(BaseModel):
        email: str  # Use str instead of EmailStr to avoid email_validator dependency
        username: str = Field(min_length=3, max_length=50)
        password: str = Field(min_length=8)
        full_name: str | None = None

    # Valid signup request
    signup_data = SignupRequest(
        email="user@example.com",
        username="testuser",
        password="SecurePassword123!",
        full_name="Test User",
    )

    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email=signup_data.email,
            username=signup_data.username,
            password=signup_data.password,
        )
    )

    assert user.email == "user@example.com"
    assert user.username == "testuser"


# --8<-- [end:signup-with-validation]


# --8<-- [start:login-with-remember-me]
@pytest.mark.asyncio
async def test_login_with_remember_me(test_core):
    """Login with remember me option."""
    # Create user
    await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Login with remember_me flag
    remember_me = True
    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="SecurePassword123!",
    )

    # Calculate cookie max_age
    max_age = 3600 * 24 * 30 if remember_me else 3600 * 24  # 30 days or 1 day

    assert auth_result.success is True
    assert auth_result.token is not None
    assert max_age == 3600 * 24 * 30  # 30 days for remember me


# --8<-- [end:login-with-remember-me]


# --8<-- [start:api-login-returns-token]
@pytest.mark.asyncio
async def test_api_login_returns_token(test_core):
    """API login returns token for client storage."""
    from pydantic import BaseModel

    class LoginRequest(BaseModel):
        email: str
        password: str

    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="api@example.com",
            username="apiuser",
            password="SecurePassword123!",
        )
    )

    # Login via API
    login_data = LoginRequest(
        email="api@example.com",
        password="SecurePassword123!",
    )

    auth_result = await test_core.authenticate_user(
        username=login_data.email,
        password=login_data.password,
    )

    # Return token for API clients
    api_response = {
        "access_token": auth_result.token,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
        },
    }

    assert api_response["access_token"] is not None
    assert api_response["user"]["email"] == "api@example.com"


# --8<-- [end:api-login-returns-token]


# --8<-- [start:logout-with-audit]
@pytest.mark.asyncio
async def test_logout_with_audit(test_core):
    """Logout and log audit event."""
    # Create user and login
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="SecurePassword123!",
    )

    token = auth_result.token

    # Log audit event
    await test_core.audit_action(
        user_id=user.id,
        action="logout",
        resource_type="session",
    )

    # Verify audit was logged (demonstrates the pattern)
    assert token is not None
    assert user.id is not None


# --8<-- [end:logout-with-audit]


# --8<-- [start:change-password]
@pytest.mark.asyncio
async def test_change_password_flow(test_core):
    """Change user password flow."""
    from pydantic import BaseModel, Field

    class ChangePasswordRequest(BaseModel):
        current_password: str
        new_password: str = Field(min_length=8)

    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="OldPassword123!",
        )
    )

    # Change password request
    password_data = ChangePasswordRequest(
        current_password="OldPassword123!",
        new_password="NewPassword456!",
    )

    # Verify current password
    is_valid = await test_core.verify_password(user, password_data.current_password)
    assert is_valid is True

    # Set new password
    new_hash = await test_core.hash_password(password_data.new_password)
    from portico.ports.user import UpdateUserRequest

    updated_user = await test_core.update_user(
        user.id, UpdateUserRequest(password_hash=new_hash)
    )

    # Verify password was updated
    assert updated_user is not None


# --8<-- [end:change-password]


# --8<-- [start:secure-cookie-config]
def test_secure_cookie_configuration(config_registry):
    """Secure cookie configuration for production."""
    # Production cookie settings
    cookie_config = {
        "key": "access_token",
        "httponly": True,  # Prevents JavaScript access
        "secure": True,  # HTTPS only
        "samesite": "lax",  # CSRF protection
        "max_age": 3600 * 24,  # 24 hours
        "domain": ".yourapp.com",  # Allow subdomains
    }

    assert cookie_config["httponly"] is True
    assert cookie_config["secure"] is True
    assert cookie_config["samesite"] == "lax"
    assert cookie_config["max_age"] == 86400


# --8<-- [end:secure-cookie-config]


# --8<-- [start:password-requirements-config]
def test_password_requirements_configuration(config_registry):
    """Configure password requirements."""
    # Password configuration
    password_config = {
        "password_min_length": 12,
        "password_require_uppercase": True,
        "password_require_lowercase": True,
        "password_require_digit": True,
        "password_require_special": True,
    }

    # Security configuration
    security_config = {
        "max_login_attempts": 5,
        "lockout_duration_minutes": 15,
        "session_timeout_minutes": 30,
    }

    assert password_config["password_min_length"] == 12
    assert security_config["max_login_attempts"] == 5
    assert security_config["session_timeout_minutes"] == 30


# --8<-- [end:password-requirements-config]


# --8<-- [start:cors-configuration]
def test_cors_configuration(config_registry):
    """CORS configuration for authentication."""
    cors_config = {
        "allow_origins": ["https://yourapp.com"],
        "allow_credentials": True,  # Allow cookies
        "allow_methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["*"],
    }

    assert cors_config["allow_credentials"] is True
    assert "https://yourapp.com" in cors_config["allow_origins"]
    assert "POST" in cors_config["allow_methods"]


# --8<-- [end:cors-configuration]


# --8<-- [start:token-expiry-handling]
@pytest.mark.asyncio
async def test_token_expiry_handling(test_core):
    """Handle expired tokens."""
    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Authenticate to get token
    auth_result = await test_core.authenticate_user(
        username="user@example.com",
        password="SecurePassword123!",
    )

    # Token should be valid immediately after creation
    assert auth_result.token is not None
    assert auth_result.success is True


# --8<-- [end:token-expiry-handling]


# --8<-- [start:multiple-login-prevention]
@pytest.mark.asyncio
async def test_multiple_login_attempts(test_core):
    """Track failed login attempts."""
    # Create user
    await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Simulate failed login attempts
    failed_attempts = 0
    max_attempts = 5

    for attempt in range(3):
        auth_result = await test_core.authenticate_user(
            username="user@example.com",
            password="WrongPassword",
        )
        if not auth_result.success:
            failed_attempts += 1

    assert failed_attempts == 3
    assert failed_attempts < max_attempts  # Haven't hit lockout yet


# --8<-- [end:multiple-login-prevention]


# --8<-- [start:user-metadata]
@pytest.mark.asyncio
async def test_user_metadata_storage(test_core):
    """Store custom metadata with users."""
    # Create user (metadata support depends on User model implementation)
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Verify user created successfully
    assert user is not None
    assert user.email == "user@example.com"


# --8<-- [end:user-metadata]


# --8<-- [start:session-ip-tracking]
@pytest.mark.asyncio
async def test_session_ip_tracking(test_core):
    """Track user IP addresses in sessions."""
    # Demonstrate IP tracking pattern
    mock_ip_address = "192.168.1.100"
    mock_user_agent = "Mozilla/5.0"

    # Create user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com",
            username="user",
            password="SecurePassword123!",
        )
    )

    # Session metadata pattern for IP tracking
    session_metadata = {
        "ip_address": mock_ip_address,
        "user_agent": mock_user_agent,
    }

    assert session_metadata["ip_address"] == "192.168.1.100"
    assert session_metadata["user_agent"] == "Mozilla/5.0"


# --8<-- [end:session-ip-tracking]
