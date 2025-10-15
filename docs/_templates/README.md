# Documentation Templates

This directory contains reusable templates for Portico documentation. These files are excluded from the built documentation site.

## Available Templates

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

The template includes placeholders in the format `{PLACEHOLDER_NAME}`. Replace these with actual values:

- `{KIT_NAME}`: Display name (e.g., "Auth")
- `{KIT_NAME_LOWER}`: Lowercase name for config (e.g., "auth")
- `{kit_name}`: Snake_case name for variables (e.g., "auth_kit")
- `{KitName}`: PascalCase name for classes (e.g., "AuthKit")
- `{PortName}`: Port interface name (e.g., "AuthPort")
- `{AdapterName}`: Adapter class name (e.g., "CookieSessionAdapter")

## Sections

You can omit sections that don't apply to your kit:

- **Events**: Only if the kit publishes domain events
- **Database Models**: Only if the kit has database tables
- **Migration Guide**: Only when there are breaking changes
- **Performance Considerations**: Optional, add if relevant
- **Security Considerations**: Optional, but recommended for security-sensitive kits

## Snippets

You can also use the template as a snippet reference and include only specific sections using pymdownx.snippets if needed.
