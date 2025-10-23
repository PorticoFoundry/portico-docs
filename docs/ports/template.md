# Template Port

## Overview

The Template Port defines the contract for storing, managing, and rendering reusable templates in Portico applications.

**Purpose**: Abstract template storage and rendering operations to enable consistent text generation across different use cases (LLM prompts, emails, notifications, etc.).

**Domain**: Template management, content generation, variable substitution

**Key Capabilities**:

- Jinja2-based template storage and rendering
- Template versioning with full history
- User ownership and access control
- Type-specific template organization (LLM prompts, emails, SMS, webhooks)
- Variable extraction and validation
- Template search and filtering
- Public and private template sharing
- Version restoration and rollback

**Port Type**: Repository + Adapter (dual interface)

**When to Use**:

- LLM prompt template management
- Email and notification template systems
- Webhook payload templating
- Multi-tenant applications with user-specific templates
- Content generation with dynamic variable substitution
- Applications requiring template version control

## Domain Models

### Template

Represents a reusable template with type-specific metadata and optional user ownership. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | `uuid4()` | Unique template identifier |
| `name` | `str` | Yes | - | Template name (unique within type) |
| `description` | `Optional[str]` | No | `None` | Template description |
| `template_type` | `str` | Yes | - | Type identifier (e.g., "llm_prompt", "notification_email") |
| `content` | `str` | Yes | - | Jinja2 template content |
| `variables` | `List[str]` | No | `[]` | List of variable names used in template |
| `metadata` | `Dict[str, Any]` | No | `{}` | Type-specific configuration |
| `tags` | `List[str]` | No | `[]` | Tags for organization and search |
| `user_id` | `Optional[UUID]` | No | `None` | Owner user ID (None = system template) |
| `is_public` | `bool` | No | `False` | Whether template is visible to all users |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |
| `updated_at` | `datetime` | No | Current UTC time | Last update timestamp |

**Methods**:

- `get_metadata_field(field: str, default: Any = None) -> Any` - Safely get metadata field with default
- `is_owned_by(user_id: UUID) -> bool` - Check if template is owned by user
- `is_accessible_by(user_id: Optional[UUID]) -> bool` - Check if template is accessible by user (public, owned, or system)

**Example**:

```python
from portico.ports.template import Template, TemplateTypes

# LLM prompt template
template = Template(
    name="customer_support_assistant",
    description="Friendly customer support assistant prompt",
    template_type=TemplateTypes.LLM_PROMPT,
    content="You are a helpful customer support assistant for {{ company_name }}. "
            "Be friendly and professional. Customer question: {{ question }}",
    variables=["company_name", "question"],
    metadata={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 500
    },
    tags=["customer-support", "assistant"],
    user_id=None,  # System template
    is_public=True
)

# Check access
if template.is_accessible_by(user_id):
    print("User can access this template")

# Get metadata
model = template.get_metadata_field("model", "gpt-3.5-turbo")
```

### TemplateVersion

A version snapshot of a template, enabling version history and restoration. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | `uuid4()` | Unique version identifier |
| `template_id` | `UUID` | Yes | - | Parent template ID |
| `name` | `str` | Yes | - | Template name (snapshot) |
| `description` | `Optional[str]` | No | `None` | Template description (snapshot) |
| `template_type` | `str` | Yes | - | Template type (snapshot) |
| `content` | `str` | Yes | - | Template content (snapshot) |
| `variables` | `List[str]` | No | `[]` | Variable names (snapshot) |
| `metadata` | `Dict[str, Any]` | No | `{}` | Metadata (snapshot) |
| `tags` | `List[str]` | No | `[]` | Tags (snapshot) |
| `version_number` | `int` | Yes | - | Sequential version number (1, 2, 3...) |
| `created_by` | `Optional[UUID]` | No | `None` | User who created this version |
| `change_note` | `Optional[str]` | No | `None` | Optional note describing changes |
| `restored_from_version` | `Optional[int]` | No | `None` | If restored, which version number |
| `created_at` | `datetime` | No | Current UTC time | Version creation timestamp |
| `updated_at` | `datetime` | No | Current UTC time | Version update timestamp |

**Example**:

```python
from portico.ports.template import TemplateVersion

# Get version history
versions = await template_registry.list_versions(template_id)

for version in versions:
    print(f"Version {version.version_number}")
    print(f"  Created by: {version.created_by}")
    print(f"  Note: {version.change_note}")
    if version.restored_from_version:
        print(f"  Restored from v{version.restored_from_version}")
```

### CreateTemplateRequest

Request for creating a new template.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | - | Template name |
| `description` | `Optional[str]` | No | `None` | Template description |
| `template_type` | `str` | Yes | - | Template type identifier |
| `content` | `str` | Yes | - | Jinja2 template content |
| `variables` | `List[str]` | No | `[]` | Variable names |
| `metadata` | `Dict[str, Any]` | No | `{}` | Type-specific metadata |
| `tags` | `List[str]` | No | `[]` | Tags |
| `user_id` | `Optional[UUID]` | No | `None` | Owner user ID |
| `is_public` | `bool` | No | `False` | Public visibility |

**Example**:

```python
from portico.ports.template import CreateTemplateRequest, TemplateTypes

request = CreateTemplateRequest(
    name="welcome_email",
    description="Welcome email for new users",
    template_type=TemplateTypes.NOTIFICATION_EMAIL,
    content="Welcome {{ user_name }}! Thanks for joining {{ app_name }}.",
    variables=["user_name", "app_name"],
    tags=["onboarding", "email"],
    user_id=admin_user_id,
    is_public=True
)

template = await template_registry.create(request)
```

### UpdateTemplateRequest

Request for updating an existing template. All fields optional for partial updates.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `Optional[str]` | No | `None` | New template name |
| `description` | `Optional[str]` | No | `None` | New description |
| `content` | `Optional[str]` | No | `None` | New template content |
| `variables` | `Optional[List[str]]` | No | `None` | New variable list |
| `metadata` | `Optional[Dict[str, Any]]` | No | `None` | New metadata |
| `tags` | `Optional[List[str]]` | No | `None` | New tags |
| `is_public` | `Optional[bool]` | No | `None` | New public status |

**Example**:

```python
from portico.ports.template import UpdateTemplateRequest

# Update content and make public
request = UpdateTemplateRequest(
    content="Updated template content with {{ new_variable }}",
    variables=["new_variable"],
    is_public=True
)

updated = await template_registry.update(template_id, request, user_id)
```

### TemplateQuery

Query parameters for searching templates. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `template_type` | `Optional[str]` | No | `None` | Filter by single template type |
| `template_types` | `Optional[List[str]]` | No | `None` | Filter by multiple template types |
| `tags` | `Optional[List[str]]` | No | `None` | Filter by tags (any match) |
| `name_contains` | `Optional[str]` | No | `None` | Filter by name substring |
| `limit` | `int` | No | `100` | Maximum results (1-1000) |
| `offset` | `int` | No | `0` | Results to skip (≥0) |

**Example**:

```python
from portico.ports.template import TemplateQuery, TemplateTypes

# Search for email templates
query = TemplateQuery(
    template_types=[TemplateTypes.NOTIFICATION_EMAIL, TemplateTypes.NOTIFICATION_SMS],
    tags=["onboarding"],
    name_contains="welcome",
    limit=10
)

templates = await template_registry.search_templates(query)
```

## Enumerations

### TemplateTypes

Standard template type constants.

| Constant | Value | Description |
|----------|-------|-------------|
| `LLM_PROMPT` | `"llm_prompt"` | LLM system/user prompts |
| `NOTIFICATION_EMAIL` | `"notification_email"` | Email notification templates |
| `NOTIFICATION_SMS` | `"notification_sms"` | SMS notification templates |
| `WEBHOOK` | `"webhook"` | Webhook payload templates |
| `SLACK` | `"slack"` | Slack message templates |

**Example**:

```python
from portico.ports.template import TemplateTypes

# Use constants instead of strings
template = CreateTemplateRequest(
    name="my_prompt",
    template_type=TemplateTypes.LLM_PROMPT,  # ✅ Type-safe
    content="..."
)

# ❌ Avoid magic strings
template_type = "llm_prompt"
```

## Port Interfaces

### TemplateRegistry

The `TemplateRegistry` abstract base class defines the contract for template storage and retrieval.

**Location**: `portico.ports.template.TemplateRegistry`

#### Key Methods

##### create

```python
async def create(template_data: CreateTemplateRequest) -> Template
```

Create a new template. Primary method for storing templates.

**Parameters**:

- `template_data`: Template creation data

**Returns**: Created Template object.

**Example**:

```python
from portico.ports.template import CreateTemplateRequest, TemplateTypes

# Create LLM prompt template
request = CreateTemplateRequest(
    name="code_reviewer",
    description="Code review assistant prompt",
    template_type=TemplateTypes.LLM_PROMPT,
    content="""You are an expert code reviewer. Review this code for:
- Bugs and errors
- Performance issues
- Best practices

Code:
{{ code }}

Language: {{ language }}""",
    variables=["code", "language"],
    metadata={
        "model": "gpt-4",
        "temperature": 0.3
    },
    tags=["code-review", "development"],
    user_id=user_id,
    is_public=False
)

template = await template_registry.create(request)
print(f"Created template: {template.id}")
```

##### get_by_id

```python
async def get_by_id(template_id: UUID, user_id: Optional[UUID] = None) -> Optional[Template]
```

Retrieve a template by ID with access control. Primary method for loading templates.

**Parameters**:

- `template_id`: Template identifier
- `user_id`: Optional user ID for access control (checks if user can access template)

**Returns**: Template object if found and accessible, None otherwise.

**Example**:

```python
# Get template with access control
template = await template_registry.get_by_id(template_id, user_id=current_user_id)

if template:
    # User has access
    rendered = template_renderer.render(template.content, {"code": code, "language": "python"})
else:
    # Template not found or access denied
    print("Template not accessible")

# Get system template (no access control)
system_template = await template_registry.get_by_id(template_id)
```

#### Other Methods

##### get_by_name

```python
async def get_by_name(name: str, template_type: Optional[str] = None) -> Optional[Template]
```

Retrieve a template by name and optional type. Returns first match if type not specified.

##### update

```python
async def update(
    template_id: UUID,
    update_data: UpdateTemplateRequest,
    user_id: Optional[UUID] = None
) -> Optional[Template]
```

Update an existing template. If user_id provided, verifies ownership. Returns updated template or None.

##### delete

```python
async def delete(template_id: UUID, user_id: Optional[UUID] = None) -> bool
```

Delete a template. If user_id provided, verifies ownership. Returns True if deleted, False if not found.

##### list_templates

```python
async def list_templates(
    template_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    user_id: Optional[UUID] = None,
    include_public: bool = True,
    limit: int = 100,
    offset: int = 0
) -> List[Template]
```

List templates with filtering and pagination. If user_id provided, returns user's templates + public + system templates.

##### search_templates

```python
async def search_templates(query: TemplateQuery) -> List[Template]
```

Search templates with advanced filters. Returns list of templates matching query.

##### get_latest_version

```python
async def get_latest_version(template_id: UUID) -> Optional[TemplateVersion]
```

Get the most recent version of a template. Returns latest TemplateVersion or None.

##### get_version

```python
async def get_version(template_id: UUID, version_number: int) -> Optional[TemplateVersion]
```

Get a specific version of a template. Returns TemplateVersion or None.

##### list_versions

```python
async def list_versions(template_id: UUID, limit: int = 100, offset: int = 0) -> List[TemplateVersion]
```

List all versions of a template, newest first. Returns list of TemplateVersion objects.

##### restore_version

```python
async def restore_version(
    template_id: UUID,
    version_number: int,
    created_by: Optional[UUID] = None,
    change_note: Optional[str] = None
) -> Template
```

Restore template to a previous version. Creates new version with old content (preserves history). Raises ResourceNotFoundError if not found.

##### get_version_count

```python
async def get_version_count(template_id: UUID) -> int
```

Get total number of versions for a template. Returns version count.

### TemplateRenderer

The `TemplateRenderer` abstract base class defines the contract for rendering templates with variables.

**Location**: `portico.ports.template.TemplateRenderer`

#### Key Methods

##### render

```python
def render(template: str, variables: Dict[str, Any]) -> str
```

Render a template with variables. Primary method for template rendering.

**Parameters**:

- `template`: Jinja2 template string
- `variables`: Dictionary of variable values

**Returns**: Rendered template string.

**Raises**:
- `TemplateRenderError` if rendering fails
- `TemplateValidationError` if template has syntax errors

**Example**:

```python
from portico.adapters.template import Jinja2TemplateRenderer

renderer = Jinja2TemplateRenderer()

# Simple variable substitution
template = "Hello {{ name }}!"
rendered = renderer.render(template, {"name": "Alice"})
print(rendered)  # "Hello Alice!"

# With conditionals
template = """
{% if is_premium %}
Premium user: {{ name }}
{% else %}
Free user: {{ name }}
{% endif %}
"""
rendered = renderer.render(template, {"name": "Bob", "is_premium": True})

# With loops
template = """
Items:
{% for item in items %}
- {{ item }}
{% endfor %}
"""
rendered = renderer.render(template, {"items": ["apple", "banana", "cherry"]})
```

##### extract_variables

```python
def extract_variables(template: str) -> List[str]
```

Extract variable names from a template. Returns list of unique variable names found.

**Example**:

```python
template = "Hello {{ name }}, your score is {{ score }}!"
variables = renderer.extract_variables(template)
print(variables)  # ["name", "score"]
```

#### Other Methods

##### validate_variables

```python
def validate_variables(template: str, variables: Dict[str, Any]) -> List[str]
```

Check if all required variables are provided. Returns list of missing variable names (empty if all provided).

## Common Patterns

### LLM Prompt Management

```python
from portico.ports.template import (
    TemplateRegistry,
    TemplateRenderer,
    CreateTemplateRequest,
    TemplateTypes
)

async def create_prompt_template(
    template_registry: TemplateRegistry,
    name: str,
    content: str,
    model: str = "gpt-4",
    temperature: float = 0.7
):
    """Create an LLM prompt template with metadata."""

    # Extract variables from template
    from portico.adapters.template import Jinja2TemplateRenderer
    renderer = Jinja2TemplateRenderer()
    variables = renderer.extract_variables(content)

    # Create template
    request = CreateTemplateRequest(
        name=name,
        template_type=TemplateTypes.LLM_PROMPT,
        content=content,
        variables=variables,
        metadata={
            "model": model,
            "temperature": temperature,
            "max_tokens": 1000
        }
    )

    template = await template_registry.create(request)
    return template

async def render_prompt(
    template_registry: TemplateRegistry,
    template_renderer: TemplateRenderer,
    template_name: str,
    variables: dict
) -> str:
    """Load and render a prompt template."""

    # Get template
    template = await template_registry.get_by_name(
        template_name,
        template_type=TemplateTypes.LLM_PROMPT
    )

    if not template:
        raise ValueError(f"Template '{template_name}' not found")

    # Validate variables
    missing = template_renderer.validate_variables(template.content, variables)
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    # Render
    return template_renderer.render(template.content, variables)

# Usage
template = await create_prompt_template(
    template_registry,
    name="code_explainer",
    content="Explain this {{ language }} code:\n\n{{ code }}"
)

prompt = await render_prompt(
    template_registry,
    template_renderer,
    template_name="code_explainer",
    variables={"language": "Python", "code": "def hello(): print('Hi')"}
)
```

### Template Versioning and Rollback

```python
from portico.ports.template import UpdateTemplateRequest

async def update_template_with_versioning(
    template_registry: TemplateRegistry,
    template_id: UUID,
    new_content: str,
    user_id: UUID,
    change_note: str
):
    """Update template content and create version snapshot."""

    # Update creates a new version automatically
    request = UpdateTemplateRequest(content=new_content)
    updated = await template_registry.update(template_id, request, user_id)

    # Get version history
    versions = await template_registry.list_versions(template_id)
    latest_version = versions[0]

    logger.info(
        "template_updated",
        template_id=str(template_id),
        version=latest_version.version_number,
        note=change_note
    )

    return updated

async def rollback_template(
    template_registry: TemplateRegistry,
    template_id: UUID,
    version_number: int,
    user_id: UUID
):
    """Rollback template to a previous version."""

    # Restore creates a new version with old content
    restored = await template_registry.restore_version(
        template_id=template_id,
        version_number=version_number,
        created_by=user_id,
        change_note=f"Rolled back to version {version_number}"
    )

    logger.info(
        "template_restored",
        template_id=str(template_id),
        restored_from=version_number
    )

    return restored

# Usage
# Update template (creates version 2)
await update_template_with_versioning(
    template_registry,
    template_id,
    new_content="Updated prompt: {{ variable }}",
    user_id=admin_id,
    change_note="Improved clarity"
)

# Oops, rollback to version 1
await rollback_template(template_registry, template_id, version_number=1, user_id=admin_id)
```

## Integration with Kits

The Template Port is used by the **LLM Kit** for prompt template management and by applications for general template needs.

```python
from portico import compose
from portico.adapters.template import Jinja2TemplateRenderer
from portico.ports.template import CreateTemplateRequest, TemplateTypes

# Templates are stored in the database
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[compose.llm(provider="openai", api_key="sk-...")]
)

await app.initialize()

# Access template registry through database adapter
# (Note: Template kit not yet implemented, use adapters directly)
from portico.adapters.storage import SqlAlchemyTemplateRegistry

template_registry = SqlAlchemyTemplateRegistry(database=app.database)
template_renderer = Jinja2TemplateRenderer()

# Create prompt template
request = CreateTemplateRequest(
    name="chat_assistant",
    template_type=TemplateTypes.LLM_PROMPT,
    content="You are a {{ persona }}. Answer: {{ question }}",
    variables=["persona", "question"],
    metadata={"model": "gpt-4", "temperature": 0.7}
)

template = await template_registry.create(request)

# Render and use with LLM
prompt = template_renderer.render(
    template.content,
    {"persona": "helpful assistant", "question": "What is Python?"}
)

response = await app.kits["llm"].service.complete(prompt)
```

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Extract Variables Automatically**: Use `extract_variables()` to avoid manual variable tracking

   ```python
   # ✅ GOOD: Auto-extract variables
   renderer = Jinja2TemplateRenderer()
   variables = renderer.extract_variables(content)

   template = CreateTemplateRequest(
       name="my_template",
       content=content,
       variables=variables  # Automatically tracked
   )

   # ❌ BAD: Manually specify (error-prone)
   variables = ["var1", "var2"]  # Easy to forget variables
   ```

2. **Validate Variables Before Rendering**: Check for missing variables to provide clear errors

   ```python
   # ✅ GOOD: Validate first
   missing = renderer.validate_variables(template.content, variables)
   if missing:
       raise ValueError(f"Missing variables: {missing}")
   rendered = renderer.render(template.content, variables)

   # ❌ BAD: No validation (cryptic errors)
   rendered = renderer.render(template.content, variables)
   # Raises UndefinedError at runtime
   ```

3. **Use Namespaces for Multi-Tenancy**: Leverage user_id for template isolation

   ```python
   # ✅ GOOD: User-owned templates
   template = CreateTemplateRequest(
       name="my_prompt",
       content="...",
       user_id=current_user_id,  # User owns this
       is_public=False  # Private to user
   )

   # System admins can create public templates
   system_template = CreateTemplateRequest(
       name="default_prompt",
       content="...",
       user_id=None,  # System template
       is_public=True  # Everyone can use
   )

   # ❌ BAD: No ownership (can't filter by user)
   template = CreateTemplateRequest(name="template", content="...")
   ```

4. **Version Risky Changes**: Use versioning before updating critical templates

   ```python
   # ✅ GOOD: Version history preserved
   # Each update creates a new version
   await template_registry.update(template_id, UpdateTemplateRequest(content=new_content))

   # Can rollback if needed
   await template_registry.restore_version(template_id, version_number=1)

   # ❌ BAD: Direct content modification
   # (Not possible with immutable models, but conceptually wrong)
   ```

5. **Use Type Constants**: Use `TemplateTypes` constants instead of magic strings

   ```python
   # ✅ GOOD: Type-safe constants
   from portico.ports.template import TemplateTypes

   template = CreateTemplateRequest(
       template_type=TemplateTypes.LLM_PROMPT,  # IDE autocomplete
       ...
   )

   # ❌ BAD: Magic strings (typo-prone)
   template = CreateTemplateRequest(
       template_type="llm_promt",  # Typo!
       ...
   )
   ```

## FAQs

### What template syntax is supported?

Portico uses **Jinja2** template syntax, supporting:

- Variable substitution: `{{ variable }}`
- Conditionals: `{% if condition %}...{% endif %}`
- Loops: `{% for item in items %}...{% endfor %}`
- Filters: `{{ variable|upper }}`
- Comments: `{# comment #}`

See [Jinja2 documentation](https://jinja.palletsprojects.com/) for full syntax.

### How does template versioning work?

Every time you update a template, a new `TemplateVersion` is created automatically. Versions are numbered sequentially (1, 2, 3...) and preserve the complete template state.

**Version operations:**
- `list_versions()` - View version history
- `get_version()` - Load a specific version
- `restore_version()` - Rollback to an old version (creates new version with old content)

**Important**: Restoring a version doesn't delete newer versions—it creates a new version with the old content, preserving full history.

### How do I handle missing template variables?

Use `validate_variables()` before rendering:

```python
missing = renderer.validate_variables(template.content, variables)
if missing:
    # Prompt user for missing values or use defaults
    for var in missing:
        variables[var] = get_default_value(var)

rendered = renderer.render(template.content, variables)
```

Alternatively, catch `TemplateRenderError` when rendering.

### Can I use templates without a database?

Yes! You can use the `TemplateRenderer` directly without storing templates:

```python
from portico.adapters.template import Jinja2TemplateRenderer

renderer = Jinja2TemplateRenderer()

# Render inline template
template_string = "Hello {{ name }}!"
rendered = renderer.render(template_string, {"name": "World"})
```

The `TemplateRegistry` is only needed if you want to store, version, and manage templates in a database.

### How do I implement a custom template storage backend?

Implement the `TemplateRegistry` interface:

```python
from portico.ports.template import (
    TemplateRegistry,
    Template,
    CreateTemplateRequest,
    UpdateTemplateRequest
)

class CustomTemplateRegistry(TemplateRegistry):
    async def create(self, template_data: CreateTemplateRequest) -> Template:
        # Store template in your backend
        template_id = await your_db.insert_template(template_data)
        return await self.get_by_id(template_id)

    async def get_by_id(
        self,
        template_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[Template]:
        # Retrieve from your backend
        data = await your_db.get_template(template_id)

        if not data:
            return None

        template = Template(**data)

        # Check access control
        if user_id and not template.is_accessible_by(user_id):
            return None

        return template

    # Implement all other abstract methods...
```

Then use it directly:

```python
registry = CustomTemplateRegistry(connection_string="...")
template = await registry.create(CreateTemplateRequest(...))
```
