"""Test examples for dependency injection documentation.

All examples are extracted into docs using snippet markers.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from portico.core import PorticoCore
from portico.kits.fastapi import create_all_dependencies
from portico.ports.user import CreateUserRequest, User


@pytest.fixture
async def test_core(config_registry):
    """Create test PorticoCore instance."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    yield core
    await core.close()


# --8<-- [start:basic-dependencies]
@pytest.mark.asyncio
async def test_basic_dependencies(config_registry):
    """Create basic dependencies dict."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create dependencies
    dependencies = create_all_dependencies(core, include_auth=True)

    # Dependencies dict contains pre-wrapped dependencies
    assert "get_session" in dependencies
    assert "get_current_user" in dependencies
    assert "get_optional_user" in dependencies
    assert "core" in dependencies

    await core.close()


# --8<-- [end:basic-dependencies]


# --8<-- [start:session-dependency]
@pytest.mark.asyncio
async def test_session_dependency(test_core):
    """Use session dependency in route."""
    dependencies = create_all_dependencies(test_core, include_auth=False)

    # Verify dependencies dict structure
    assert "get_session" in dependencies
    assert "core" in dependencies

    # Show correct usage pattern (function definition)
    # @app.get("/users")
    # async def get_users(session: AsyncSession = dependencies["get_session"]):
    #     users = await core.get_all_users()
    #     return {"count": len(users)}


# --8<-- [end:session-dependency]


# --8<-- [start:auth-dependencies]
@pytest.mark.asyncio
async def test_auth_dependencies(test_core):
    """Use authentication dependencies."""
    dependencies = create_all_dependencies(test_core, include_auth=True)

    # Verify auth dependencies are included
    assert "get_current_user" in dependencies
    assert "get_optional_user" in dependencies

    # Show correct usage patterns (function definitions)
    # Protected route with required auth:
    # @app.get("/protected")
    # async def protected_route(current_user: User = dependencies["get_current_user"]):
    #     return {"email": current_user.email}

    # Optional auth route:
    # @app.get("/optional")
    # async def optional_route(current_user: Optional[User] = dependencies["get_optional_user"]):
    #     if current_user:
    #         return {"email": current_user.email}
    #     return {"email": None}


# --8<-- [end:auth-dependencies]


# --8<-- [start:custom-dependency]
class UserService:
    """Example custom service."""

    def __init__(self, core: PorticoCore, current_user: Optional[User] = None):
        self.core = core
        self.current_user = current_user

    async def get_users(self):
        """Get all users."""
        return await self.core.get_all_users()


@pytest.mark.asyncio
async def test_custom_dependency(test_core):
    """Create custom service dependency."""
    dependencies = create_all_dependencies(test_core, include_auth=True)

    # Factory function for custom service
    def get_user_service(
        current_user: Optional[User] = dependencies["get_optional_user"],
    ) -> UserService:
        return UserService(test_core, current_user)

    # Verify factory function is callable
    assert callable(get_user_service)

    # Show correct usage pattern (function definition)
    # @app.post("/users")
    # async def create_user(service: UserService = Depends(get_user_service)):
    #     users = await service.get_users()
    #     return {"count": len(users)}


# --8<-- [end:custom-dependency]


# --8<-- [start:dependency-override]
class MockUserService:
    """Mock service for testing."""

    async def get_users(self):
        # Mock response with 1 fake user
        class MockUser:
            email = "mock@example.com"

        return [MockUser()]


@pytest.mark.asyncio
async def test_dependency_override(test_core):
    """Override dependencies for testing."""
    dependencies = create_all_dependencies(test_core, include_auth=False)

    def get_user_service() -> UserService:
        return UserService(test_core)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/users")
    async def get_users(service: UserService = Depends(get_user_service)):
        users = await service.get_users()
        return {"count": len(users)}

    # Override for testing
    app.dependency_overrides[get_user_service] = lambda: MockUserService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/users")
        assert response.status_code == 200
        assert response.json()["count"] == 1


# --8<-- [end:dependency-override]


# --8<-- [start:core-access]
@pytest.mark.asyncio
async def test_core_access(test_core):
    """Access core directly from dependencies."""
    dependencies = create_all_dependencies(test_core, include_auth=False)

    # Access core from dependencies dict
    core_instance = dependencies["core"]

    # Verify core is accessible
    assert core_instance is not None
    assert core_instance == test_core

    # Show correct usage pattern (function definition)
    # @app.get("/users")
    # async def get_users():
    #     # Use core methods directly
    #     users = await core_instance.get_all_users()
    #     return {"count": len(users)}


# --8<-- [end:core-access]


# --8<-- [start:lifespan-pattern]
@pytest.mark.asyncio
async def test_lifespan_pattern(config_registry):
    """Demonstrate lifespan pattern with dependencies."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create dependencies
    dependencies = create_all_dependencies(core, include_auth=True)

    # Verify pattern works
    assert "get_session" in dependencies
    assert "get_current_user" in dependencies

    # Show correct usage pattern:
    # @asynccontextmanager
    # async def lifespan(app: FastAPI):
    #     await core.initialize()
    #     dependencies = create_all_dependencies(core, include_auth=True)
    #     app.state.dependencies = dependencies
    #     yield
    #     await core.close()
    #
    # app = FastAPI(lifespan=lifespan)

    await core.close()


# --8<-- [end:lifespan-pattern]


# --8<-- [start:shared-config]
@pytest.mark.asyncio
async def test_shared_config(config_registry):
    """Demonstrate shared configuration pattern."""
    # This would be in shared.py
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create dependencies once
    dependencies = create_all_dependencies(core, include_auth=True)

    # Use in multiple modules/routes
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/route1")
    async def route1(session: AsyncSession = dependencies["get_session"]):
        return {"route": "route1"}

    @app.get("/route2")
    async def route2(current_user: User = dependencies["get_current_user"]):
        return {"route": "route2"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/route1")
        assert response.status_code == 200

    await core.close()


# --8<-- [end:shared-config]


# --8<-- [start:environment-config]
@pytest.mark.asyncio
async def test_environment_config(config_registry):
    """Create dependencies from environment variables."""
    # Set environment variables
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    os.environ["SECRET_KEY"] = "test-secret-key-minimum-32-characters-from-env"

    # Load config from environment
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    # Create dependencies
    dependencies = create_all_dependencies(core, include_auth=True)

    assert "get_session" in dependencies
    assert "get_current_user" in dependencies

    await core.close()

    # Cleanup
    del os.environ["DATABASE_URL"]
    del os.environ["SECRET_KEY"]


# --8<-- [end:environment-config]


# --8<-- [start:authentication-patterns]
@pytest.mark.asyncio
async def test_authentication_patterns(test_core):
    """Test authentication dependency patterns."""
    dependencies = create_all_dependencies(test_core, include_auth=True)

    # Create test app
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)

    # Protected route - 401 if not authenticated
    @app.get("/dashboard")
    async def dashboard(current_user: User = dependencies["get_current_user"]):
        return {"user": current_user.email}

    # Optional auth - works with or without authentication
    @app.get("/home")
    async def home(current_user: Optional[User] = dependencies["get_optional_user"]):
        if current_user:
            return {"message": f"Welcome back, {current_user.email}"}
        return {"message": "Welcome, guest"}

    # Public route - no authentication required
    @app.get("/public")
    async def public():
        return {"message": "Public content"}

    # Test endpoints
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Public route works
        response = await client.get("/public")
        assert response.status_code == 200
        assert response.json()["message"] == "Public content"

        # Optional auth works without auth
        response = await client.get("/home")
        assert response.status_code == 200
        assert "guest" in response.json()["message"]


# --8<-- [end:authentication-patterns]


# --8<-- [start:factory-pattern-best-practices]
@pytest.mark.asyncio
async def test_factory_pattern_best_practices(test_core):
    """Factory pattern best practices."""
    dependencies = create_all_dependencies(test_core, include_auth=False)

    class MyService:
        def __init__(self, core: PorticoCore):
            self.core = core

        async def get_count(self):
            users = await self.core.list_users()
            return len(users)

    # ✅ Good - Simple factory
    def get_service() -> MyService:
        return MyService(test_core)

    # ✅ Good - Factory with additional logic
    def get_service_with_validation() -> MyService:
        # Can add validation or setup here
        return MyService(test_core)

    # Verify factory works
    service = get_service()
    count = await service.get_count()
    assert count >= 0


# --8<-- [end:factory-pattern-best-practices]


# --8<-- [start:when-to-use-core-directly]
@pytest.mark.asyncio
async def test_when_to_use_core_directly(test_core):
    """When to use core directly vs session."""
    dependencies = create_all_dependencies(test_core, include_auth=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)

    # ✅ Good - Use core for high-level operations
    @app.post("/users")
    async def create_user(email: str, username: str):
        core = dependencies["core"]
        user = await core.create_user(
            CreateUserRequest(email=email, username=username, password="Test123!")
        )
        return {"id": str(user.id)}

    # Verify pattern works
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/users?email=test@example.com&username=testuser")
        assert response.status_code == 200
        assert "id" in response.json()


# --8<-- [end:when-to-use-core-directly]


# --8<-- [start:complete-application-setup]
@pytest.mark.asyncio
async def test_complete_application_setup(config_registry):
    """Complete application setup with lifecycle."""
    # Global core instance
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    dependencies = create_all_dependencies(core, include_auth=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        # Startup
        await core.initialize()
        yield
        # Shutdown
        await core.close()

    # Create app
    app = FastAPI(title="My Application", lifespan=lifespan)

    # Routes
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    # Test application
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# --8<-- [end:complete-application-setup]


# --8<-- [start:testing-with-real-dependencies]
@pytest.fixture
async def test_app_with_real_deps(config_registry):
    """Create test app with real dependencies."""
    core = PorticoCore(
        config_registry=config_registry, log_level="INFO", environment="testing"
    )
    await core.initialize()

    dependencies = create_all_dependencies(core, include_auth=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/users")
    async def get_users():
        users = await core.list_users()
        return {"count": len(users)}

    yield app, core

    await core.close()


@pytest.mark.asyncio
async def test_real_dependencies(test_app_with_real_deps):
    """Test with real dependencies."""
    app, core = test_app_with_real_deps

    # Create test user
    await core.create_user(
        CreateUserRequest(
            email="test@example.com", username="test", password="Test123!"
        )
    )

    # Test endpoint
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/users")
        assert response.status_code == 200
        assert response.json()["count"] == 1


# --8<-- [end:testing-with-real-dependencies]


# --8<-- [start:dependency-organization]
@pytest.mark.asyncio
async def test_dependency_organization(test_core):
    """Centralize custom dependencies."""
    dependencies = create_all_dependencies(test_core, include_auth=True)

    class BlogService:
        def __init__(self, core: PorticoCore, current_user: Optional[User] = None):
            self.core = core
            self.current_user = current_user

    class CommentService:
        def __init__(self, core: PorticoCore, current_user: User):
            self.core = core
            self.current_user = current_user

    def create_app_dependencies(core: PorticoCore):
        """Create all application dependencies."""
        # Get framework dependencies
        deps = create_all_dependencies(core, include_auth=True)

        # Add custom factories
        def get_blog_service(
            current_user: Optional[User] = deps["get_optional_user"],
        ) -> BlogService:
            return BlogService(core, current_user)

        def get_comment_service(
            current_user: User = deps["get_current_user"],
        ) -> CommentService:
            return CommentService(core, current_user)

        # Return combined dict
        return {
            **deps,
            "get_blog_service": get_blog_service,
            "get_comment_service": get_comment_service,
        }

    # Create combined dependencies
    app_deps = create_app_dependencies(test_core)

    # Verify both framework and custom deps available
    assert "get_session" in app_deps
    assert "get_current_user" in app_deps
    assert "get_blog_service" in app_deps
    assert "get_comment_service" in app_deps


# --8<-- [end:dependency-organization]


# --8<-- [start:error-handling-in-factory]
@pytest.mark.asyncio
async def test_error_handling_in_factory(test_core):
    """Error handling in dependency factories."""
    from portico.exceptions import AuthorizationError

    dependencies = create_all_dependencies(test_core, include_auth=True)

    class AdminService:
        def __init__(self, core: PorticoCore, current_user: User):
            self.core = core
            self.current_user = current_user

    # Create test user
    user = await test_core.create_user(
        CreateUserRequest(
            email="user@example.com", username="user", password="Test123!"
        )
    )

    def get_admin_service(current_user: User) -> AdminService:
        """Factory for admin-only service."""
        # Validate permissions in factory
        if current_user.global_role != "admin":
            raise AuthorizationError("Admin access required")
        return AdminService(test_core, current_user)

    # Non-admin user should raise error
    with pytest.raises(AuthorizationError):
        get_admin_service(user)


# --8<-- [end:error-handling-in-factory]
