"""Test examples for Architecture documentation.

All examples are extracted into docs using snippet markers.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel


# --8<-- [start:port-interface-definition]
def test_port_interface_definition():
    """Define a port interface."""

    class CacheAdapter(ABC):
        """Cache port - defines interface, not implementation."""

        @abstractmethod
        async def get(self, key: str, namespace: str = "default") -> str | None:
            """Get value from cache."""

        @abstractmethod
        async def set(
            self,
            key: str,
            value: str,
            ttl: int | None = None,
            namespace: str = "default",
        ) -> None:
            """Set value in cache."""

        @abstractmethod
        async def delete(self, key: str, namespace: str = "default") -> None:
            """Delete value from cache."""

    # Verify interface defined
    assert hasattr(CacheAdapter, "get")
    assert hasattr(CacheAdapter, "set")
    assert hasattr(CacheAdapter, "delete")


# --8<-- [end:port-interface-definition]


# --8<-- [start:domain-models]
def test_domain_models():
    """Domain models in ports."""

    class User(BaseModel):
        """Domain model - business representation of a user."""

        id: UUID
        email: str
        username: str
        is_active: bool = True

        # Domain methods allowed
        def has_role(self, role: str) -> bool:
            """Check if user has a specific role."""
            return self.global_role == role if hasattr(self, "global_role") else False

    class CreateUserRequest(BaseModel):
        """Request to create a user."""

        email: str
        username: str
        password: str | None = None

    # Test domain model
    user = User(id=uuid4(), email="test@example.com", username="test")
    assert user.email == "test@example.com"
    assert user.is_active is True


# --8<-- [end:domain-models]


# --8<-- [start:crud-interface-pattern]
def test_crud_interface_pattern():
    """CRUD interface pattern for repositories."""

    class CreateUserRequest(BaseModel):
        email: str
        username: str

    class User(BaseModel):
        id: UUID
        email: str
        username: str

    class UserRepository(ABC):
        """Repository pattern for users."""

        @abstractmethod
        async def create(self, request: CreateUserRequest) -> User:
            pass

        @abstractmethod
        async def get_by_id(self, user_id: UUID) -> User | None:
            pass

        @abstractmethod
        async def update(self, user_id: UUID, updates: dict) -> User:
            pass

        @abstractmethod
        async def delete(self, user_id: UUID) -> bool:
            pass

        @abstractmethod
        async def list(self, limit: int = 100, offset: int = 0) -> list[User]:
            pass

    # Verify CRUD methods defined
    assert hasattr(UserRepository, "create")
    assert hasattr(UserRepository, "get_by_id")
    assert hasattr(UserRepository, "update")
    assert hasattr(UserRepository, "delete")
    assert hasattr(UserRepository, "list")


# --8<-- [end:crud-interface-pattern]


# --8<-- [start:service-interface-pattern]
def test_service_interface_pattern():
    """Service interface pattern for external services."""

    class LLMResponse(BaseModel):
        message: str
        model: str
        usage: dict

    class LLMProvider(ABC):
        """LLM service interface."""

        @abstractmethod
        async def generate(
            self,
            messages: list[dict[str, str]],
            model: str | None = None,
            max_tokens: int = 1000,
            temperature: float = 0.7,
        ) -> LLMResponse:
            """Generate completion from messages."""

        @abstractmethod
        async def stream_generate(
            self,
            messages: list[dict[str, str]],
            **kwargs,
        ) -> AsyncIterator[str]:
            """Stream completion tokens."""

    # Verify service methods defined
    assert hasattr(LLMProvider, "generate")
    assert hasattr(LLMProvider, "stream_generate")


# --8<-- [end:service-interface-pattern]


# --8<-- [start:memory-cache-adapter]
@pytest.mark.asyncio
async def test_memory_cache_adapter():
    """In-memory cache adapter implementation."""

    class CacheAdapter(ABC):
        @abstractmethod
        async def get(self, key: str, namespace: str = "default") -> str | None:
            pass

        @abstractmethod
        async def set(
            self,
            key: str,
            value: str,
            ttl: int | None = None,
            namespace: str = "default",
        ) -> None:
            pass

    class MemoryCacheAdapter(CacheAdapter):
        """In-memory cache implementation."""

        def __init__(self):
            self.data: dict[str, str] = {}

        async def get(self, key: str, namespace: str = "default") -> str | None:
            namespaced_key = f"{namespace}:{key}"
            return self.data.get(namespaced_key)

        async def set(
            self,
            key: str,
            value: str,
            ttl: int | None = None,
            namespace: str = "default",
        ) -> None:
            namespaced_key = f"{namespace}:{key}"
            self.data[namespaced_key] = value

    # Test adapter
    cache = MemoryCacheAdapter()
    await cache.set("key", "value")
    result = await cache.get("key")
    assert result == "value"


# --8<-- [end:memory-cache-adapter]


# --8<-- [start:mock-adapter]
@pytest.mark.asyncio
async def test_mock_adapter():
    """Mock adapter for testing."""

    class CacheAdapter(ABC):
        @abstractmethod
        async def get(self, key: str, namespace: str = "default") -> str | None:
            pass

        @abstractmethod
        async def set(
            self,
            key: str,
            value: str,
            ttl: int | None = None,
            namespace: str = "default",
        ) -> None:
            pass

    class MockCacheAdapter(CacheAdapter):
        """Mock cache for testing."""

        async def get(self, key: str, namespace: str = "default") -> str | None:
            return None  # Always miss

        async def set(
            self,
            key: str,
            value: str,
            ttl: int | None = None,
            namespace: str = "default",
        ) -> None:
            pass  # No-op

    # Use mock
    cache = MockCacheAdapter()
    await cache.set("key", "value")
    result = await cache.get("key")
    assert result is None  # Mock always returns None


# --8<-- [end:mock-adapter]


# --8<-- [start:user-service-business-logic]
@pytest.mark.asyncio
async def test_user_service_business_logic():
    """User service with business logic."""
    from portico.exceptions import ValidationError

    class User(BaseModel):
        id: UUID
        email: str
        username: str

    class CreateUserRequest(BaseModel):
        email: str
        username: str

    class UserRepository(ABC):
        @abstractmethod
        async def create(self, request: CreateUserRequest) -> User:
            pass

    class MockUserRepository(UserRepository):
        async def create(self, request: CreateUserRequest) -> User:
            return User(id=uuid4(), email=request.email, username=request.username)

    class UserService:
        """User business logic."""

        def __init__(self, user_repository: UserRepository):
            self.users = user_repository

        async def create_user(self, request: CreateUserRequest) -> User:
            """Create user with business rules."""
            # Business rule: Email required
            if not request.email:
                raise ValidationError("Email is required")

            # Business rule: Username required
            if not request.username or len(request.username) < 3:
                raise ValidationError("Username must be at least 3 characters")

            # Create user via repository
            user = await self.users.create(request)

            return user

    # Test service
    service = UserService(MockUserRepository())
    user = await service.create_user(
        CreateUserRequest(email="test@example.com", username="testuser")
    )
    assert user.email == "test@example.com"


# --8<-- [end:user-service-business-logic]


# --8<-- [start:dependency-injection-factory]
def test_dependency_injection_factory():
    """Factory pattern for dependency injection."""

    class User(BaseModel):
        id: UUID
        email: str

    class CreateUserRequest(BaseModel):
        email: str
        username: str

    class UserRepository(ABC):
        @abstractmethod
        async def create(self, request: CreateUserRequest) -> User:
            pass

    class MockUserRepository(UserRepository):
        async def create(self, request: CreateUserRequest) -> User:
            return User(id=uuid4(), email=request.email)

    class UserService:
        def __init__(self, user_repository: UserRepository):
            self.users = user_repository

    def create_user_service() -> UserService:
        """Factory function for UserService."""
        user_repository = MockUserRepository()
        return UserService(user_repository=user_repository)

    # Create service using factory
    service = create_user_service()
    assert service is not None
    assert isinstance(service, UserService)


# --8<-- [end:dependency-injection-factory]


# --8<-- [start:mock-unit-test]
@pytest.mark.asyncio
async def test_mock_unit_test():
    """Unit test with mock adapters."""

    class User(BaseModel):
        id: UUID
        email: str
        username: str

    class CreateUserRequest(BaseModel):
        email: str
        username: str

    class UserRepository(ABC):
        @abstractmethod
        async def create(self, request: CreateUserRequest) -> User:
            pass

    class MockUserRepository(UserRepository):
        async def create(self, request: CreateUserRequest) -> User:
            return User(id=uuid4(), email=request.email, username=request.username)

    class UserService:
        def __init__(self, user_repository: UserRepository):
            self.users = user_repository

        async def create_user(self, request: CreateUserRequest) -> User:
            return await self.users.create(request)

    # Unit test - mock all adapters
    service = UserService(user_repository=MockUserRepository())

    user = await service.create_user(
        CreateUserRequest(email="test@example.com", username="test")
    )

    assert user.email == "test@example.com"


# --8<-- [end:mock-unit-test]


# --8<-- [start:repository-pattern]
def test_repository_pattern():
    """Repository pattern for data access."""

    class Entity(BaseModel):
        id: UUID
        name: str

    class Repository(ABC):
        @abstractmethod
        async def create(self, entity: Entity) -> Entity:
            pass

        @abstractmethod
        async def get(self, id: UUID) -> Entity | None:
            pass

    class MemoryRepository(Repository):
        def __init__(self):
            self.data: dict[UUID, Entity] = {}

        async def create(self, entity: Entity) -> Entity:
            self.data[entity.id] = entity
            return entity

        async def get(self, id: UUID) -> Entity | None:
            return self.data.get(id)

    # Verify pattern
    assert hasattr(Repository, "create")
    assert hasattr(Repository, "get")


# --8<-- [end:repository-pattern]


# --8<-- [start:strategy-pattern]
def test_strategy_pattern():
    """Strategy pattern for multiple implementations."""

    class PaymentResult(BaseModel):
        success: bool
        transaction_id: str

    class PaymentProvider(ABC):
        @abstractmethod
        async def process_payment(self, amount: float) -> PaymentResult:
            pass

    class StripePaymentProvider(PaymentProvider):
        async def process_payment(self, amount: float) -> PaymentResult:
            return PaymentResult(success=True, transaction_id="stripe_123")

    class PayPalPaymentProvider(PaymentProvider):
        async def process_payment(self, amount: float) -> PaymentResult:
            return PaymentResult(success=True, transaction_id="paypal_456")

    class PaymentService:
        def __init__(self, providers: dict[str, PaymentProvider]):
            self.providers = providers

        async def pay(self, amount: float, method: str) -> PaymentResult:
            provider = self.providers[method]
            return await provider.process_payment(amount)

    # Verify strategy pattern
    assert hasattr(PaymentProvider, "process_payment")
    assert hasattr(PaymentService, "pay")


# --8<-- [end:strategy-pattern]


# --8<-- [start:port-rules]
def test_port_rules():
    """Rules for port definitions."""

    # ✅ Ports should be abstract
    class GoodPort(ABC):
        @abstractmethod
        async def operation(self) -> str:
            pass

    # ✅ Ports should use domain models
    class DomainModel(BaseModel):
        id: UUID
        name: str

    class PortWithDomainModel(ABC):
        @abstractmethod
        async def get(self, id: UUID) -> DomainModel:
            pass

    # Verify rules followed
    assert hasattr(GoodPort, "operation")
    assert hasattr(PortWithDomainModel, "get")


# --8<-- [end:port-rules]


# --8<-- [start:adapter-rules]
def test_adapter_rules():
    """Rules for adapter implementations."""

    class Port(ABC):
        @abstractmethod
        async def operation(self) -> str:
            pass

    # ✅ Adapters implement port interface
    class GoodAdapter(Port):
        async def operation(self) -> str:
            return "result"

    # Verify adapter implements interface
    adapter = GoodAdapter()
    assert isinstance(adapter, Port)


# --8<-- [end:adapter-rules]


# --8<-- [start:hexagonal-benefits]
@pytest.mark.asyncio
async def test_hexagonal_benefits():
    """Benefits of hexagonal architecture."""

    class User(BaseModel):
        id: UUID
        email: str

    class UserAdapter(ABC):
        @abstractmethod
        async def get_user(self, user_id: UUID) -> User | None:
            pass

    # Benefit 1: Test with mock adapter
    class MockAdapter(UserAdapter):
        async def get_user(self, user_id: UUID) -> User | None:
            return User(id=user_id, email="test@example.com")

    # Benefit 2: Swap adapters without changing service
    class MemoryAdapter(UserAdapter):
        def __init__(self):
            self.users: dict[UUID, User] = {}

        async def get_user(self, user_id: UUID) -> User | None:
            return self.users.get(user_id)

    # Same interface, different implementations
    mock = MockAdapter()
    memory = MemoryAdapter()

    user_id = uuid4()
    mock_user = await mock.get_user(user_id)
    assert mock_user is not None


# --8<-- [end:hexagonal-benefits]


# --8<-- [start:best-practices-do]
def test_best_practices_do():
    """Best practices: What to do."""

    class User(BaseModel):
        id: UUID
        email: str
        username: str

    # ✅ Define clear interfaces
    class UserAdapter(ABC):
        @abstractmethod
        async def create_user(self, email: str, username: str) -> User:
            """Create a new user."""

    # ✅ Depend on interfaces
    class UserService:
        def __init__(self, users: UserAdapter):  # Interface, not concrete class
            self.users = users

    # Verify best practices
    assert hasattr(UserAdapter, "create_user")
    assert hasattr(UserService, "__init__")


# --8<-- [end:best-practices-do]


# --8<-- [start:dependency-flow]
def test_dependency_flow():
    """Dependency flow in hexagonal architecture."""

    # Ports (no dependencies)
    class DomainPort(ABC):
        @abstractmethod
        async def operation(self) -> str:
            pass

    # Adapters depend on ports
    class Adapter(DomainPort):
        async def operation(self) -> str:
            return "implementation"

    # Kits depend on ports
    class Service:
        def __init__(self, port: DomainPort):
            self.port = port

        async def execute(self) -> str:
            return await self.port.operation()

    # Dependencies point inward:
    # - Adapters depend on ports
    # - Kits depend on ports
    # - Ports depend on nothing
    assert issubclass(Adapter, DomainPort)
    service = Service(Adapter())
    assert service.port is not None


# --8<-- [end:dependency-flow]


# --8<-- [start:testing-benefits]
@pytest.mark.asyncio
async def test_testing_benefits():
    """Testing benefits of hexagonal architecture."""

    class CreateUserRequest(BaseModel):
        email: str
        username: str

    class User(BaseModel):
        id: UUID
        email: str
        username: str

    class UserAdapter(ABC):
        @abstractmethod
        async def create_user(self, request: CreateUserRequest) -> User:
            pass

    # Test service without any infrastructure
    class MockUserAdapter(UserAdapter):
        async def create_user(self, request: CreateUserRequest) -> User:
            return User(id=uuid4(), email=request.email, username=request.username)

    class UserService:
        def __init__(self, adapter: UserAdapter):
            self.adapter = adapter

        async def create(self, request: CreateUserRequest) -> User:
            return await self.adapter.create_user(request)

    # Unit test - no database needed
    service = UserService(MockUserAdapter())
    user = await service.create(
        CreateUserRequest(email="test@example.com", username="test")
    )
    assert user.email == "test@example.com"


# --8<-- [end:testing-benefits]


# --8<-- [start:flexibility-benefits]
@pytest.mark.asyncio
async def test_flexibility_benefits():
    """Flexibility benefits of hexagonal architecture."""

    class CacheAdapter(ABC):
        @abstractmethod
        async def set(self, key: str, value: str) -> None:
            pass

        @abstractmethod
        async def get(self, key: str) -> str | None:
            pass

    class MemoryCacheAdapter(CacheAdapter):
        def __init__(self):
            self.data: dict[str, str] = {}

        async def set(self, key: str, value: str) -> None:
            self.data[key] = value

        async def get(self, key: str) -> str | None:
            return self.data.get(key)

    class MockCacheAdapter(CacheAdapter):
        async def set(self, key: str, value: str) -> None:
            pass

        async def get(self, key: str) -> str | None:
            return None

    # Development - use memory
    cache: CacheAdapter = MemoryCacheAdapter()

    # Testing - use mock
    # cache = MockCacheAdapter()

    # Interface stays the same!
    await cache.set("key", "value")
    result = await cache.get("key")
    assert result == "value"


# --8<-- [end:flexibility-benefits]


# --8<-- [start:interface-segregation]
def test_interface_segregation():
    """Interface segregation principle."""

    # Split large interfaces into smaller, focused ones
    class Readable(ABC):
        @abstractmethod
        async def read(self, id: UUID) -> dict:
            pass

    class Writable(ABC):
        @abstractmethod
        async def write(self, id: UUID, data: dict) -> None:
            pass

    class Deletable(ABC):
        @abstractmethod
        async def delete(self, id: UUID) -> bool:
            pass

    # Implement only what you need
    class ReadOnlyRepository(Readable):
        async def read(self, id: UUID) -> dict:
            return {"id": str(id)}

    class FullRepository(Readable, Writable, Deletable):
        async def read(self, id: UUID) -> dict:
            return {}

        async def write(self, id: UUID, data: dict) -> None:
            pass

        async def delete(self, id: UUID) -> bool:
            return True

    # Verify segregation
    assert hasattr(ReadOnlyRepository, "read")
    assert hasattr(FullRepository, "read")
    assert hasattr(FullRepository, "write")


# --8<-- [end:interface-segregation]
