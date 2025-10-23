# Documentation Templates

This directory contains reusable templates for Portico documentation. These files are excluded from the built documentation site.

## Available Templates

### `port_template.md`

Comprehensive template for documenting Portico ports (domain interfaces).

**Usage:**

1. Create a new port documentation file (e.g., `docs/ports/cache/index.md`)
2. Copy the template and customize sections:

```markdown
# Cache Port

## Overview

**Purpose**: Provides caching capabilities for application data with TTL and tag-based invalidation.

**Domain**: Performance optimization and data caching

**Key Capabilities**:
- Store and retrieve cached values with TTL
- Tag-based cache invalidation
- Namespace isolation for multi-tenancy
- Cache statistics and monitoring

**Port Type**: Service

## When to Use

Use this port when you need to:
- Improve application performance with caching
- Reduce database load for frequently accessed data
- Implement multi-tenant caching with namespaces

[Continue filling out other sections...]
```

### `kit_template.md`

Comprehensive template for documenting Portico kits.

**Usage:**

1. Create a new kit documentation file (e.g., `docs/kits/auth/index.md`)
2. Include the template and customize sections:

```markdown
# Auth Kit

## Overview

**Purpose**: Provides authentication and session management for web applications.

**Key Features**:
- Login/logout functionality
- Session token management
- Password hashing and validation
- Cookie-based session storage

**Dependencies**:
- Required kits: User Kit
- Required ports: `AuthPort`, `UserRepository`
- Optional kits: Audit Kit (for login tracking)

## Quick Start

\`\`\`python
from portico import compose

app_services = compose.webapp(
    auth={
        "session_secret": "your-secret-key",
        "session_cookie_name": "session_token",
    }
)

auth_kit = app_services.auth
\`\`\`

[Continue filling out other sections...]
```

## Template Customization

### Port Template Placeholders

The port template includes placeholders in the format `{PLACEHOLDER_NAME}`. Replace these with actual values:

- `{PORT_NAME}`: Display name (e.g., "Cache", "User", "LLM")
- `{port_name}`: Lowercase/snake_case name for imports (e.g., "cache", "user", "llm")
- `{ModelName}`: PascalCase name for domain models (e.g., "User", "CacheEntry", "Template")
- `{RequestModelName}`: PascalCase request model name (e.g., "CreateUserRequest", "UpdateCacheEntryRequest")
- `{InterfaceName}`: PascalCase interface name (e.g., "CacheAdapter", "UserRepository", "LLMProvider")
- `{EnumName}`: PascalCase enum name (e.g., "MessageRole", "CachePolicy")
- `{AdapterName}`: PascalCase adapter class name (e.g., "RedisCacheAdapter", "OpenAIProvider")
- `{KitName}`: PascalCase kit name (e.g., "Cache", "User", "LLM")
- `{kit_name}`: Lowercase kit name for config (e.g., "cache", "user", "llm")

### Kit Template Placeholders

The kit template includes placeholders in the format `{PLACEHOLDER_NAME}`. Replace these with actual values:

- `{KIT_NAME}`: Display name (e.g., "Auth")
- `{KIT_NAME_LOWER}`: Lowercase name for config (e.g., "auth")
- `{kit_name}`: Snake_case name for variables (e.g., "auth_kit")
- `{KitName}`: PascalCase name for classes (e.g., "AuthKit")
- `{PortName}`: Port interface name (e.g., "AuthPort")
- `{AdapterName}`: Adapter class name (e.g., "CookieSessionAdapter")

## Optional Sections

### Port Template Sections

You can omit sections that don't apply to your port:

- **Enumerations**: Only if the port defines enums (e.g., MessageRole, CachePolicy)
- **Helper Functions**: Only if the port provides utility functions beyond domain models
- **Type Aliases**: Only if the port defines type aliases for backward compatibility
- **Constants**: Only if the port defines constant values or configuration classes
- **Migration Guide**: Only when there are breaking changes between versions
- **Advanced Usage**: Optional, include if there are complex patterns worth documenting

**Required Sections** (always include):
- Overview
- Domain Models (at least one)
- Port Interfaces (at least one)
- Available Adapters
- Integration with Kits
- Complete Example
- Best Practices

### Kit Template Sections

You can omit sections that don't apply to your kit:

- **Events**: Only if the kit publishes domain events
- **Database Models**: Only if the kit has database tables
- **Migration Guide**: Only when there are breaking changes
- **Performance Considerations**: Optional, add if relevant
- **Security Considerations**: Optional, but recommended for security-sensitive kits

## Snippets

You can also use the template as a snippet reference and include only specific sections using pymdownx.snippets if needed.
