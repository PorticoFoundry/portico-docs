# {KIT_NAME} Kit

## Overview

**Purpose**: {Brief description of what this kit does and why it exists}

**Key Features**:
- {Feature 1}
- {Feature 2}
- {Feature 3}

**Dependencies**:
- **Injected services**: {List services from other kits received via dependency injection, e.g., "UserService, EventBus"}
- **Port dependencies**: {List port interfaces this kit uses, e.g., "CacheAdapter, LLMProvider"}
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose

# Basic configuration
app = compose.webapp(
    {KIT_NAME_LOWER}={
        # Minimal required configuration
        "option1": "value1",
    }
)

# Access the kit
{kit_name}_kit = app.{kit_name}
```

## Core Concepts

### {Concept 1}

{Explanation of the first major concept}

```python
# Example demonstrating concept
```

### {Concept 2}

{Explanation of the second major concept}

```python
# Example demonstrating concept
```

## Configuration

### Required Settings

| Setting | Type | Description | Example |
|---------|------|-------------|---------|
| `option1` | `str` | {Description} | `"value"` |
| `option2` | `int` | {Description} | `100` |

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `optional1` | `bool` | `False` | {Description} |
| `optional2` | `str` | `None` | {Description} |

## Usage Examples

### Example 1: {Common Use Case}

```python
# Setup
{kit_name}_kit = app_services.{kit_name}

# Usage
result = await {kit_name}_kit.{method_name}(...)

# Result handling
if result:
    # Success path
    pass
else:
    # Error handling
    pass
```

### Example 2: {Advanced Use Case}

```python
# More complex example showing advanced usage
```

## Database Models

### {ModelName}

**Table**: `{table_name}`

**Columns**:
- `id`: Primary key
- `{column1}`: {Description}
- `{column2}`: {Description}
- `created_at`: Timestamp
- `updated_at`: Timestamp

**Relationships**:
- Belongs to: {Related model}
- Has many: {Related model}

## Events

This kit publishes the following events:

### `{EventName}`

**Triggered when**: {Description of when this event fires}

**Payload**:
```python
{
    "event_type": "{event_name}",
    "data": {
        "field1": "value",
        "field2": "value",
    },
    "timestamp": "2025-01-01T00:00:00Z"
}
```

**Listeners**: {Other kits that typically listen to this event}

## Best Practices

### 1. {Practice Name}

{Description of the best practice and why it matters}

```python
# Good example
# ...

# Avoid
# ...
```

### 2. {Practice Name}

{Description}

### 3. Error Handling

{Guidance on proper error handling for this kit}

```python
from portico.exceptions import {RelevantException}

try:
    result = await {kit_name}_kit.{method_name}(...)
except {RelevantException} as e:
    # Handle specific exception
    pass
```

## Security Considerations

{Security guidance specific to this kit}

- {Security consideration 1}
- {Security consideration 2}

## FAQs

### Q: {Common question}

A: {Answer}

### Q: {Common question}

A: {Answer}
