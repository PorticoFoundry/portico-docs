# {PORT_NAME} Port

## Overview

The {PORT_NAME} Port defines the contract for {brief description of what this port does}.

**Purpose**: {One sentence explaining the main purpose}

**Domain**: {Business domain - e.g., "User management", "Caching", "Security"}

**Key Capabilities**:

- {Capability 1}
- {Capability 2}
- {Capability 3}

**Port Type**: {Repository | Provider | Adapter | Registry | Storage | Processor}

**When to Use**:

- {Use case 1}
- {Use case 2}
- {Use case 3}

## Domain Models

### {ModelName}

{Brief description of what this model represents}

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `{field_name}` | `{Type}` | {Yes/No} | `{default}` | {Description} |
| `{field_name_2}` | `{Type}` | {Yes/No} | `{default}` | {Description} |

**Example**:

```python
from portico.ports.{port_name} import {ModelName}

model = {ModelName}(
    {field_name}="value",
    {field_name_2}="value"
)
```

### {RequestModelName}

{Brief description - e.g., "Request model for creating/updating entities"}

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `{field_name}` | `{Type}` | {Yes/No} | `{default}` | {Description} |

**Example**:

```python
request = {RequestModelName}(
    {field_name}="value"
)
```

## Enumerations

### {EnumName}

{Brief description of what this enum represents}

| Value | Description |
|-------|-------------|
| `{VALUE_1}` | {Description} |
| `{VALUE_2}` | {Description} |
| `{VALUE_3}` | {Description} |

**Example**:

```python
from portico.ports.{port_name} import {EnumName}

action = {EnumName}.{VALUE_1}
```

## Port Interfaces

### {InterfaceName}

The `{InterfaceName}` {abstract base class/protocol} defines the contract for {what this interface does}.

**Location**: `portico.ports.{port_name}.{InterfaceName}`

#### Key Methods

##### {primary_method_name}

```python
async def {primary_method_name}(param: Type) -> ReturnType
```

{Detailed description of the most important method}

**Parameters**:

- `param`: {Description}

**Returns**: {Description of return value}

**Example**:

```python
result = await adapter.{primary_method_name}(value)
```

##### {secondary_method_name}

```python
async def {secondary_method_name}(param: Type) -> ReturnType
```

{Detailed description of second most important method}

**Parameters**:

- `param`: {Description}

**Returns**: {Description}

**Example**:

```python
result = await adapter.{secondary_method_name}(value)
```

#### Other Methods

##### {method_name_3}

```python
async def {method_name_3}(param: Type) -> ReturnType
```

{One-line description}

##### {method_name_4}

```python
async def {method_name_4}(param: Type) -> ReturnType
```

{One-line description}

##### {method_name_5}

```python
async def {method_name_5}(param: Type) -> ReturnType
```

{One-line description}

## Common Patterns

### {Most Important Pattern}

```python
from portico.ports.{port_name} import {InterfaceName}, {ModelName}

async def example_usage(adapter: {InterfaceName}):
    # {Brief description of what this shows}
    model = {ModelName}(field="value")
    result = await adapter.{method_name}(model)
    return result
```

### {Second Most Important Pattern}

```python
# {Brief description}
async def another_pattern(adapter: {InterfaceName}):
    # {Show key pattern only}
    pass
```

## Integration with Kits

The {PORT_NAME} Port is used by the **{KitName}** to provide {brief description}.

```python
from portico import compose

# Configure
app = compose.webapp(
    kits=[
        compose.{kit_name}(
            config_param="value"
        )
    ]
)

# Access
{kit_name}_service = app.kits["{kit_name}"].service
result = await {kit_name}_service.{method_name}()
```

See [{KitName} documentation](../kits/{kit_name}.md) for complete usage details.

## Best Practices

1. **{Practice Category 1}**: {Brief description with code snippet if needed}

   ```python
   # ✅ GOOD
   {example_good}

   # ❌ BAD
   {example_bad}
   ```

2. **{Practice Category 2}**: {Brief description}

3. **{Practice Category 3}**: {Brief description}

4. **{Practice Category 4}**: {Brief description}

5. **{Practice Category 5}**: {Brief description}

## FAQs

### {Most common question}

{Answer with brief code example if helpful}

```python
# Example if needed
```

### {Second most common question}

{Answer}

### {Third most common question}

{Answer}

### {Fourth question - optional}

{Answer}

### {Fifth question - optional}

{Answer}
