# FastAPI Kit

## Overview

**Purpose**: Provide seamless integration between Portico and FastAPI with dependency injection, route decorators, exception handling, and middleware for building web applications.

**Key Features**:

- Type-safe dependency injection for services and authentication
- Route protection decorators for auth and permissions
- Automatic exception handling (converts Portico exceptions to HTTP responses)
- Request logging middleware with structured logging
- Jinja2 template integration helpers
- Support for both cookie and Bearer token authentication

**Dependencies**:

- **Injected services**: None (provides access to all kit services)
- **Port dependencies**: None (utility kit)
- **Note**: This is a utility kit that doesn't follow the standard kit pattern (no service, repository, or models). It provides FastAPI integration helpers.

## Quick Start

```python
from fastapi import FastAPI, Request
from portico import compose
from portico.kits.fastapi import (
    Dependencies,
    register_exception_handlers,
    RequestLoggingMiddleware,
)

# Create Portico app
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(),
        compose.auth(session_secret="your-secret-key"),
        compose.rbac(),
    ]
)

# Create FastAPI app
fastapi_app = FastAPI()

# Register Portico exception handlers
register_exception_handlers(fastapi_app)

# Add request logging middleware
fastapi_app.add_middleware(RequestLoggingMiddleware)

# Create dependencies helper
deps = Dependencies(app)

# Protected route with dependency injection
@fastapi_app.get("/dashboard")
async def dashboard(current_user = deps.current_user):
    return {"email": current_user.email}
```

## Core Concepts

### Dependency Injection with Dependencies Class

The `Dependencies` class provides type-safe access to common dependencies:

```python
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@fastapi_app.get("/users")
async def list_users(
    current_user = deps.current_user,  # Authenticated user (raises 401 if not auth)
    session = deps.session,  # Database session
):
    # Access services directly
    user_service = deps.user_service
    users = await user_service.list_users()
    return {"users": users}

@fastapi_app.get("/public")
async def public_page(
    user = deps.optional_user  # None if not authenticated, no 401
):
    if user:
        return {"message": f"Welcome back, {user.email}"}
    return {"message": "Welcome, guest"}
```

**Available dependencies:**

- `deps.current_user` - Authenticated user (raises 401)
- `deps.optional_user` - User or None (no error)
- `deps.session` - Database session
- `deps.user_service` - UserManagementService
- `deps.group_service` - GroupManagementService
- `deps.auth_service` - AuthenticationService
- `deps.rbac_service` - AuthorizationService
- `deps.webapp` - WebApp instance

### Route Protection Decorators

Simplify auth and permission checking with decorators:

```python
from portico.kits.fastapi import (
    requires_auth,
    requires_permission,
    requires_role,
    requires_group_permission,
)

# Require authentication
@fastapi_app.get("/dashboard")
@requires_auth(app)
async def dashboard(current_user):
    return {"email": current_user.email}

# Require specific permission
@fastapi_app.delete("/users/{user_id}")
@requires_permission(app, "users.delete")
async def delete_user(user_id: UUID, current_user):
    await deps.user_service.delete_user(user_id)
    return {"status": "deleted"}

# Require role
@fastapi_app.get("/admin")
@requires_role(app, "admin")
async def admin_panel(current_user):
    return {"message": "Admin access"}

# Require multiple acceptable roles
@fastapi_app.get("/staff")
@requires_role(app, ["admin", "moderator", "staff"])
async def staff_dashboard(current_user):
    return {"message": "Staff access"}

# Require group-specific permission
@fastapi_app.post("/groups/{group_id}/files")
@requires_group_permission(app, "files.write")
async def upload_file(group_id: UUID, file: UploadFile, current_user):
    return await file_service.upload(group_id, file)
```

### Automatic Exception Handling

Convert Portico domain exceptions to proper HTTP responses:

```python
from portico.kits.fastapi import register_exception_handlers
from portico.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
)

# Register handlers once at startup
register_exception_handlers(fastapi_app)

# Now exceptions automatically convert to HTTP responses
@fastapi_app.get("/users/{user_id}")
async def get_user(user_id: UUID):
    user = await user_service.get_by_id(user_id)
    if not user:
        # Automatically returns 404 with proper JSON
        raise ResourceNotFoundError(f"User {user_id} not found")
    return user

@fastapi_app.post("/admin/action")
async def admin_action(current_user = deps.current_user):
    if not await rbac_service.check_permission(current_user.id, "admin.action"):
        # Automatically returns 403 with proper JSON
        raise AuthorizationError("Admin permission required")
    # Perform action
```

**Exception mappings:**

- `AuthenticationError` → 401 Unauthorized
- `AuthorizationError` → 403 Forbidden
- `ResourceNotFoundError` → 404 Not Found
- `ValidationError` → 400 Bad Request
- `ConflictError` → 409 Conflict
- `RateLimitError` → 429 Too Many Requests
- `PorticoError` → Uses exception's status_code (default 500)

### Request Logging Middleware

Add structured logging with request IDs:

```python
from portico.kits.fastapi import RequestLoggingMiddleware

fastapi_app.add_middleware(RequestLoggingMiddleware)

# Now all requests are logged with:
# - Unique request ID
# - HTTP method and path
# - Response status code
# - Errors (if any)

@fastapi_app.get("/api/data")
async def get_data(request: Request):
    # request.state.request_id is available
    logger.info("processing_data", request_id=request.state.request_id)
    return {"data": "value"}
```

Logs example:

```
INFO: http_request method=GET path=/api/data request_id=abc-123
INFO: processing_data request_id=abc-123
INFO: http_response method=GET path=/api/data status_code=200 request_id=abc-123
```

### Template Integration

Use Portico features in Jinja2 templates:

```python
from fastapi.templating import Jinja2Templates
from portico.kits.fastapi import setup_template_globals, create_template_context

templates = Jinja2Templates(directory="templates")
setup_template_globals(templates, app)

@fastapi_app.get("/dashboard")
async def dashboard(request: Request, user = deps.current_user):
    context = create_template_context(
        request,
        user=user,
        page_title="Dashboard"
    )
    return templates.TemplateResponse("dashboard.html", context)
```

In templates:

```jinja2
{% raw %}
<!DOCTYPE html>
<html>
<head>
    <title>{{ page_title }}</title>
    {{ app.render_assets('head')|safe }}
</head>
<body>
    <h1>Welcome, {{ user.email }}</h1>
    <p>App version: {{ app.version }}</p>
</body>
</html>
{% endraw %}
```

## Configuration

This kit has no configuration. It's a collection of utilities that integrate Portico with FastAPI.

## Usage Examples

### Example 1: Complete FastAPI Application Setup

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from portico import compose
from portico.kits.fastapi import (
    Dependencies,
    register_exception_handlers,
    RequestLoggingMiddleware,
)

# Create Portico app
portico_app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(),
        compose.auth(session_secret="your-secret"),
        compose.group(),
        compose.rbac(),
    ]
)

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    await portico_app.initialize()
    yield
    await portico_app.close()

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Register exception handlers and middleware
register_exception_handlers(app)
app.add_middleware(RequestLoggingMiddleware)

# Create dependencies
deps = Dependencies(portico_app)

# Routes
@app.get("/")
async def home():
    return {"message": "Welcome to Portico"}

@app.post("/login")
async def login(email: str, password: str):
    result = await deps.auth_service.authenticate(email, password)
    return {"token": result.token}

@app.get("/dashboard")
async def dashboard(user = deps.current_user):
    return {"email": user.email}
```

### Example 2: Protected Routes with RBAC

```python
from portico.kits.fastapi import requires_permission, requires_role
from portico.ports.user import CreateUserRequest

# Admin-only route
@app.get("/admin/users")
@requires_role(portico_app, "admin")
async def list_all_users(current_user):
    users = await deps.user_service.list_users()
    return {"users": users, "count": len(users)}

# Permission-based route
@app.post("/users")
@requires_permission(portico_app, "users.create")
async def create_user(
    user_data: CreateUserRequest,
    current_user
):
    user = await deps.user_service.create_user(user_data)
    return {"user_id": str(user.id)}

# Group-specific permission
@app.get("/groups/{group_id}/members")
@requires_group_permission(portico_app, "members.read")
async def list_members(group_id: UUID, current_user):
    members = await deps.group_service.get_members(group_id)
    return {"members": members}
```

### Example 3: Custom Authentication Flow

```python
from fastapi import Response
from portico.exceptions import AuthenticationError

@app.post("/login")
async def login(
    response: Response,
    email: str,
    password: str
):
    try:
        result = await deps.auth_service.authenticate(email, password)

        # Set cookie
        response.set_cookie(
            key="access_token",
            value=result.token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600
        )

        return {
            "success": True,
            "user": {
                "id": str(result.user.id),
                "email": result.user.email
            }
        }
    except AuthenticationError as e:
        # Exception handler converts to 401 automatically
        raise

@app.post("/logout")
async def logout(response: Response, user = deps.optional_user):
    if user:
        # Get token from cookie or header
        token = response.cookies.get("access_token")
        if token:
            await deps.auth_service.logout(token)

    response.delete_cookie("access_token")
    return {"success": True}
```

### Example 4: API with Error Handling

```python
from portico.exceptions import (
    ResourceNotFoundError,
    ValidationError,
    AuthorizationError,
)

@app.get("/api/users/{user_id}")
async def get_user(user_id: UUID, current_user = deps.current_user):
    # Check permission
    if not await deps.rbac_service.check_permission(
        current_user.id, "users.read"
    ):
        raise AuthorizationError("Cannot read user data")

    # Get user
    user = await deps.user_service.get_by_id(user_id)
    if not user:
        raise ResourceNotFoundError(f"User {user_id} not found")

    return {"user": user}

@app.put("/api/users/{user_id}")
async def update_user(
    user_id: UUID,
    update_data: dict,
    current_user = deps.current_user
):
    # Validate input
    if "email" in update_data and not "@" in update_data["email"]:
        raise ValidationError("Invalid email format")

    # Update user
    user = await deps.user_service.update_user(user_id, update_data)
    return {"user": user}
```

### Example 5: Template-Based Web App

```python
from fastapi import Request
from fastapi.templating import Jinja2Templates
from portico.kits.fastapi import (
    setup_template_globals,
    create_template_context,
)

templates = Jinja2Templates(directory="templates")
setup_template_globals(templates, portico_app)

@app.get("/")
async def home(request: Request, user = deps.optional_user):
    context = create_template_context(
        request,
        user=user,
        page_title="Home"
    )
    return templates.TemplateResponse("home.html", context)

@app.get("/dashboard")
async def dashboard(request: Request, user = deps.current_user):
    # Get user's data
    groups = await deps.group_service.get_user_groups(user.id)

    context = create_template_context(
        request,
        user=user,
        groups=groups,
        page_title="Dashboard"
    )
    return templates.TemplateResponse("dashboard.html", context)
```

## Best Practices

### 1. Use Dependencies Class Instead of Dictionary

The Dependencies class provides type safety and IDE autocomplete:

```python
# ✅ GOOD - Type-safe dependencies
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@app.get("/users")
async def list_users(user = deps.current_user):  # Autocomplete works
    return {"user": user.email}

# ❌ BAD - Deprecated dictionary pattern (no type safety)
from portico.kits.fastapi import create_all_dependencies

dependencies = create_all_dependencies(app)

@app.get("/users")
async def list_users(user = dependencies["get_current_user"]):  # No autocomplete
    return {"user": user.email}
```

### 2. Register Exception Handlers Early

Register handlers before defining routes:

```python
# ✅ GOOD - Register handlers at startup
from portico.kits.fastapi import register_exception_handlers

app = FastAPI()
register_exception_handlers(app)  # Register first

@app.get("/users")
async def list_users():
    # Exceptions automatically handled
    pass

# ❌ BAD - Registering after routes might miss some handlers
@app.get("/users")
async def list_users():
    pass

register_exception_handlers(app)  # Too late
```

### 3. Use Decorators for Simple Auth Checks

Decorators are cleaner than manual checks:

```python
# ✅ GOOD - Declarative with decorator
from portico.kits.fastapi import requires_permission

@app.delete("/users/{user_id}")
@requires_permission(app, "users.delete")
async def delete_user(user_id: UUID, current_user):
    await user_service.delete_user(user_id)
    return {"deleted": True}

# ❌ BAD - Manual permission check (verbose)
@app.delete("/users/{user_id}")
async def delete_user(user_id: UUID, user = deps.current_user):
    has_perm = await deps.rbac_service.check_permission(
        user.id, "users.delete"
    )
    if not has_perm:
        raise HTTPException(403, "Permission denied")
    await user_service.delete_user(user_id)
    return {"deleted": True}
```

### 4. Add Request Logging Middleware

Enable structured logging for all requests:

```python
# ✅ GOOD - Request logging enabled
from portico.kits.fastapi import RequestLoggingMiddleware

app = FastAPI()
app.add_middleware(RequestLoggingMiddleware)

# All requests automatically logged with request IDs

# ❌ BAD - No structured logging
app = FastAPI()
# Requests not logged, no request IDs
```

### 5. Use optional_user for Public/Private Content

Differentiate between public and auth-required routes:

```python
# ✅ GOOD - Public route with optional auth
@app.get("/")
async def home(user = deps.optional_user):
    if user:
        return {"message": f"Welcome back, {user.email}"}
    return {"message": "Welcome, guest"}

# ✅ GOOD - Auth required
@app.get("/dashboard")
async def dashboard(user = deps.current_user):
    # Only authenticated users can access
    return {"dashboard": "data"}

# ❌ BAD - Using current_user for public route
@app.get("/")
async def home(user = deps.current_user):
    # Raises 401 for unauthenticated users!
    return {"message": "Welcome"}
```

### 6. Leverage Template Globals

Make app features available in all templates:

```python
# ✅ GOOD - Setup globals once
from portico.kits.fastapi import setup_template_globals

templates = Jinja2Templates(directory="templates")
setup_template_globals(templates, app)

# Now all templates can access app
# { { app.render_assets('head')|safe } }
# { { app.version } }

# ❌ BAD - Manually passing app to every route
@app.get("/page1")
async def page1(request: Request):
    return templates.TemplateResponse("page1.html", {
        "request": request,
        "app": app  # Repetitive
    })

@app.get("/page2")
async def page2(request: Request):
    return templates.TemplateResponse("page2.html", {
        "request": request,
        "app": app  # Repetitive
    })
```

### 7. Use create_template_context Helper

Simplify template context creation:

```python
# ✅ GOOD - Helper function
from portico.kits.fastapi import create_template_context

@app.get("/dashboard")
async def dashboard(request: Request, user = deps.current_user):
    context = create_template_context(
        request,
        user=user,
        page_title="Dashboard"
    )
    return templates.TemplateResponse("dashboard.html", context)

# ❌ BAD - Manual dictionary construction
@app.get("/dashboard")
async def dashboard(request: Request, user = deps.current_user):
    context = {
        "request": request,  # Must remember this
        "user": user,
        "page_title": "Dashboard"
    }
    return templates.TemplateResponse("dashboard.html", context)
```

## Security Considerations

### Authentication Token Storage

Use secure cookie settings for session tokens:

```python
response.set_cookie(
    key="access_token",
    value=token,
    httponly=True,   # Prevent JavaScript access (XSS protection)
    secure=True,      # HTTPS only
    samesite="lax",   # CSRF protection
    max_age=3600      # 1 hour expiration
)
```

### Permission Checking

Always check permissions before performing sensitive operations:

```python
# Check permission before action
@app.delete("/users/{user_id}")
@requires_permission(app, "users.delete")
async def delete_user(user_id: UUID, current_user):
    # Permission already validated by decorator
    await user_service.delete_user(user_id)
    return {"deleted": True}
```

### Input Validation

Validate user input before processing:

```python
from pydantic import BaseModel, EmailStr, validator

class CreateUserRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v

@app.post("/users")
async def create_user(user_data: CreateUserRequest):
    # Pydantic validates input automatically
    user = await deps.user_service.create_user(user_data)
    return user
```

### Rate Limiting

Implement rate limiting for sensitive endpoints:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, email: str, password: str):
    # Limited to 5 attempts per minute per IP
    result = await deps.auth_service.authenticate(email, password)
    return {"token": result.token}
```

## FAQs

### Q: How do I access the current user in non-route functions?

A: Pass the user as a parameter from your route handler:

```python
async def send_notification(user: User, message: str):
    # Business logic that needs user context
    pass

@app.post("/notify")
async def notify(message: str, user = deps.current_user):
    await send_notification(user, message)
    return {"sent": True}
```

### Q: Can I use multiple decorators on the same route?

A: Yes, stack decorators in order (authentication first, then authorization):

```python
@app.delete("/admin/users/{user_id}")
@requires_auth(app)  # First: check authentication
@requires_role(app, "admin")  # Second: check role
@requires_permission(app, "users.delete")  # Third: check permission
async def delete_user(user_id: UUID, current_user):
    await user_service.delete_user(user_id)
    return {"deleted": True}
```

### Q: How do I customize exception responses?

A: Create custom handlers or modify existing ones:

```python
from portico.exceptions import ValidationError

async def custom_validation_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_failed",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

app.add_exception_handler(ValidationError, custom_validation_handler)
```

### Q: How do I use Bearer tokens instead of cookies?

A: The Dependencies class automatically checks both:

```python
# Supports both:
# Cookie: access_token=<token>
# Header: Authorization: Bearer <token>

@app.get("/api/data")
async def get_data(user = deps.current_user):
    # Works with either auth method
    return {"data": "value"}
```

### Q: Can I use the FastAPI Kit without AuthKit?

A: Yes, but authentication features (`current_user`, `optional_user`, auth decorators) won't work. Other features (exception handlers, middleware, template helpers) work independently.

### Q: How do I test routes that use dependencies?

A: Override dependencies in tests:

```python
from fastapi.testclient import TestClient

def test_protected_route():
    # Create test user
    test_user = User(id=uuid4(), email="test@example.com")

    # Override dependency
    app.dependency_overrides[deps.current_user] = lambda: test_user

    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "test@example.com" in response.text
```

### Q: How do I add custom middleware?

A: Add it alongside the request logging middleware:

```python
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Portico request logging
app.add_middleware(RequestLoggingMiddleware)
```

### Q: Can I use the Dependencies class with API routes and HTML routes?

A: Yes, it works with both JSON APIs and template-based routes:

```python
# JSON API
@app.get("/api/users")
async def api_users(user = deps.current_user):
    return {"users": await user_service.list_users()}

# HTML template
@app.get("/users")
async def html_users(request: Request, user = deps.current_user):
    context = create_template_context(request, user=user)
    return templates.TemplateResponse("users.html", context)
```

### Q: How do I handle different permission levels for the same resource?

A: Use multiple decorators or manual checks:

```python
# Read permission
@app.get("/documents/{doc_id}")
@requires_permission(app, "documents.read")
async def read_document(doc_id: UUID, current_user):
    return await document_service.get(doc_id)

# Write permission
@app.put("/documents/{doc_id}")
@requires_permission(app, "documents.write")
async def update_document(doc_id: UUID, data: dict, current_user):
    return await document_service.update(doc_id, data)

# Delete permission (requires admin)
@app.delete("/documents/{doc_id}")
@requires_role(app, "admin")
async def delete_document(doc_id: UUID, current_user):
    return await document_service.delete(doc_id)
```
