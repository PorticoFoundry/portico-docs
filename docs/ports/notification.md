# Notification Port

## Overview

The Notification Port defines the contract for sending notifications across multiple channels (email, SMS) with support for template-based and direct message delivery.

**Purpose**: Enable multi-channel notification delivery with template management, recipient validation, and comprehensive delivery tracking.

**Domain**: Communication, notifications, messaging

**Key Capabilities**:

- **Multi-Channel Delivery**: Send notifications via email or SMS
- **Template-Based Messages**: Use Jinja2 templates with variable substitution, conditionals, and loops
- **Direct Messages**: Send notifications without templates for ad-hoc messaging
- **Recipient Validation**: Ensure recipients have required information for each channel
- **Batch Operations**: Send multiple notifications efficiently in a single operation
- **Delivery Tracking**: Query notification history with comprehensive filtering and pagination
- **Status Management**: Track pending, sent, failed, and queued notifications

**Port Type**: Adapter

**When to Use**:

- Transactional notifications (welcome emails, password resets, order confirmations)
- Alert systems (security alerts, system notifications, status updates)
- Marketing campaigns with templated messages
- SMS notifications for time-sensitive alerts (2FA codes, verification)
- Multi-channel notification workflows (send both email and SMS)
- Applications requiring notification audit trails and delivery tracking

**When NOT to Use**:

- Real-time chat or instant messaging (use WebSocket or messaging queue)
- Push notifications to mobile apps (use dedicated push notification service)
- In-app notifications/toasts (use frontend notification system)
- High-volume transactional email (>1M/day) requiring advanced deliverability features

## Domain Models

### NotificationRecipient

Represents a notification recipient with contact information and validation capabilities.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `email` | `Optional[EmailStr]` | No | `None` | Recipient email address (validated format) |
| `phone` | `Optional[str]` | No | `None` | Recipient phone number (E.164 format recommended, e.g., +1234567890) |
| `name` | `Optional[str]` | No | `None` | Recipient display name |
| `metadata` | `Dict[str, Any]` | No | `{}` | Additional recipient metadata |

**Methods**:

- `validate_for_channel(channel: NotificationChannel) -> bool` - Check if recipient has required info for channel

**Example**:

```python
from portico.ports.notification import NotificationRecipient, NotificationChannel

# Email recipient
email_recipient = NotificationRecipient(
    email="user@example.com",
    name="Alice Johnson",
    metadata={"user_id": "123", "subscription": "premium"},
)

# SMS recipient
sms_recipient = NotificationRecipient(
    phone="+1234567890",
    name="Bob Smith",
)

# Both channels
multi_recipient = NotificationRecipient(
    email="user@example.com",
    phone="+1234567890",
    name="Carol White",
)

# Validate for channel
assert email_recipient.validate_for_channel(NotificationChannel.EMAIL) == True
assert email_recipient.validate_for_channel(NotificationChannel.SMS) == False
```

### NotificationRequest

Request model for sending a notification, supporting both template-based and direct body messages.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `template_id` | `Optional[UUID]` | No | `None` | Template ID for template-based notifications |
| `template_name` | `Optional[str]` | No | `None` | Template name (alternative to template_id) |
| `channel` | `NotificationChannel` | Yes | - | Delivery channel (EMAIL or SMS) |
| `recipient` | `NotificationRecipient` | Yes | - | Recipient information |
| `variables` | `Dict[str, Any]` | No | `{}` | Template variables for substitution |
| `subject` | `Optional[str]` | No | `None` | Email subject (overrides template subject for EMAIL) |
| `body` | `Optional[str]` | No | `None` | Direct message body (if not using template) |
| `metadata` | `Dict[str, Any]` | No | `{}` | Additional request metadata |

**Example**:

```python
from portico.ports.notification import NotificationRequest, NotificationChannel, NotificationRecipient
from uuid import UUID

# Template-based notification
request = NotificationRequest(
    template_id=UUID("..."),
    channel=NotificationChannel.EMAIL,
    recipient=NotificationRecipient(email="user@example.com"),
    variables={"user_name": "Alice", "order_id": "12345"},
)

# Template by name
request = NotificationRequest(
    template_name="welcome_email",
    channel=NotificationChannel.EMAIL,
    recipient=NotificationRecipient(email="user@example.com"),
    variables={"app_name": "MyApp"},
)

# Direct body (no template)
request = NotificationRequest(
    channel=NotificationChannel.EMAIL,
    recipient=NotificationRecipient(email="user@example.com"),
    subject="Order Confirmation",
    body="Your order #12345 has been confirmed.",
)

# SMS notification
request = NotificationRequest(
    template_name="verification_code",
    channel=NotificationChannel.SMS,
    recipient=NotificationRecipient(phone="+1234567890"),
    variables={"code": "123456", "minutes": "5"},
)
```

### NotificationResult

Result of a notification send operation with status, timestamps, and error information.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | No | Auto-generated | Unique identifier for this notification result |
| `request` | `NotificationRequest` | Yes | - | Original notification request |
| `status` | `NotificationStatus` | Yes | - | Delivery status (PENDING, SENT, FAILED, QUEUED) |
| `channel` | `NotificationChannel` | Yes | - | Delivery channel used |
| `recipient` | `NotificationRecipient` | Yes | - | Recipient information |
| `template_id` | `Optional[UUID]` | No | `None` | Template ID if template was used |
| `sent_at` | `Optional[datetime]` | No | `None` | Timestamp when notification was sent |
| `error_message` | `Optional[str]` | No | `None` | Error message if status is FAILED |
| `external_id` | `Optional[str]` | No | `None` | External service tracking ID (e.g., SendGrid message ID) |
| `metadata` | `Dict[str, Any]` | No | `{}` | Additional result metadata |

**Example**:

```python
from portico.ports.notification import NotificationResult, NotificationStatus
from datetime import datetime

# Successful send
result = NotificationResult(
    request=request,
    status=NotificationStatus.SENT,
    channel=NotificationChannel.EMAIL,
    recipient=recipient,
    sent_at=datetime.now(),
    external_id="sg-msg-123456",
    metadata={"provider": "sendgrid"},
)

# Failed send
result = NotificationResult(
    request=request,
    status=NotificationStatus.FAILED,
    channel=NotificationChannel.EMAIL,
    recipient=recipient,
    error_message="SMTP server unavailable",
)
```

### NotificationQuery

Query parameters for searching and filtering notification history with pagination support.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `channel` | `Optional[NotificationChannel]` | No | `None` | Filter by delivery channel |
| `status` | `Optional[NotificationStatus]` | No | `None` | Filter by delivery status |
| `template_id` | `Optional[UUID]` | No | `None` | Filter by template ID |
| `recipient_email` | `Optional[str]` | No | `None` | Filter by recipient email address |
| `recipient_phone` | `Optional[str]` | No | `None` | Filter by recipient phone number |
| `start_date` | `Optional[datetime]` | No | `None` | Filter notifications sent after this date |
| `end_date` | `Optional[datetime]` | No | `None` | Filter notifications sent before this date |
| `limit` | `int` | No | `100` | Maximum results to return (1-1000) |
| `offset` | `int` | No | `0` | Pagination offset (0+) |

**Example**:

```python
from portico.ports.notification import NotificationQuery, NotificationChannel, NotificationStatus
from datetime import datetime, timedelta

# Query all sent emails in last 7 days
query = NotificationQuery(
    channel=NotificationChannel.EMAIL,
    status=NotificationStatus.SENT,
    start_date=datetime.now() - timedelta(days=7),
    limit=100,
)

# Query failed notifications for specific recipient
query = NotificationQuery(
    status=NotificationStatus.FAILED,
    recipient_email="user@example.com",
)

# Paginated query
query = NotificationQuery(
    limit=50,
    offset=100,  # Get results 100-150
)
```

## Enumerations

### NotificationChannel

Supported notification delivery channels.

| Value | Description |
|-------|-------------|
| `EMAIL` | Email notification delivery |
| `SMS` | SMS/text message delivery |

**Example**:

```python
from portico.ports.notification import NotificationChannel

# Use in requests
channel = NotificationChannel.EMAIL  # "email"
channel = NotificationChannel.SMS    # "sms"
```

### NotificationStatus

Status of a notification delivery attempt.

| Value | Description |
|-------|-------------|
| `PENDING` | Notification queued but not yet processed |
| `SENT` | Notification successfully delivered |
| `FAILED` | Notification delivery failed |
| `QUEUED` | Notification in delivery queue (provider-specific) |

**Example**:

```python
from portico.ports.notification import NotificationStatus

# Check result status
if result.status == NotificationStatus.SENT:
    print("Notification delivered successfully")
elif result.status == NotificationStatus.FAILED:
    print(f"Delivery failed: {result.error_message}")
```

## Port Interface

### NotificationAdapter

The `NotificationAdapter` abstract base class defines the contract for multi-channel notification delivery. Adapters handle sending notifications, template rendering, and tracking delivery history.

**Location**: `portico.ports.notification.NotificationAdapter`

**Note**: Template management is handled by `TemplateRegistry`. Notification templates use `template_type = TemplateTypes.NOTIFICATION_EMAIL` or `TemplateTypes.NOTIFICATION_SMS`.

#### Core Methods

##### send

```python
async def send(request: NotificationRequest) -> NotificationResult
```

Send a single notification to a recipient via the specified channel.

**Parameters**:

- `request`: Notification request with channel, recipient, and content (template or body)

**Returns**: Notification result with delivery status, timestamp, and metadata

**Raises**:
- `ValidationError`: Recipient missing required information for channel
- `NotificationChannelError`: Channel not available or misconfigured
- `NotificationTemplateError`: Template not found or invalid
- `NotificationDeliveryError`: Failed to deliver notification

**Example**:

```python
# Template-based email
result = await adapter.send(
    NotificationRequest(
        template_name="welcome_email",
        channel=NotificationChannel.EMAIL,
        recipient=NotificationRecipient(email="user@example.com"),
        variables={"user_name": "Alice", "app_name": "MyApp"},
    )
)

# Direct SMS
result = await adapter.send(
    NotificationRequest(
        channel=NotificationChannel.SMS,
        recipient=NotificationRecipient(phone="+1234567890"),
        body="Your verification code is 123456. Valid for 5 minutes.",
    )
)

# Check result
if result.status == NotificationStatus.SENT:
    print(f"Sent at: {result.sent_at}")
else:
    print(f"Failed: {result.error_message}")
```

##### send_batch

```python
async def send_batch(requests: List[NotificationRequest]) -> List[NotificationResult]
```

Send multiple notifications in a single batch operation for efficiency.

**Parameters**:

- `requests`: List of notification requests to send

**Returns**: List of notification results in same order as requests

**Raises**:
- `NotificationError`: If batch processing fails

**Note**: Individual notification failures do not stop batch processing. Each result contains its own status.

**Example**:

```python
recipients = [
    NotificationRecipient(email="user1@example.com"),
    NotificationRecipient(email="user2@example.com"),
    NotificationRecipient(email="user3@example.com"),
]

requests = [
    NotificationRequest(
        template_name="newsletter",
        channel=NotificationChannel.EMAIL,
        recipient=r,
        variables={"edition": "March 2025"},
    )
    for r in recipients
]

results = await adapter.send_batch(requests)

# Check results
sent_count = sum(1 for r in results if r.status == NotificationStatus.SENT)
failed_count = sum(1 for r in results if r.status == NotificationStatus.FAILED)
print(f"Sent: {sent_count}, Failed: {failed_count}")
```

##### get_notification_history

```python
async def get_notification_history(query: NotificationQuery) -> List[NotificationResult]
```

Query notification delivery history with filtering and pagination.

**Parameters**:

- `query`: Query parameters for filtering and pagination

**Returns**: List of notification results matching the query

**Example**:

```python
# Get recent failed notifications
query = NotificationQuery(
    status=NotificationStatus.FAILED,
    start_date=datetime.now() - timedelta(hours=24),
    limit=50,
)
failed_notifications = await adapter.get_notification_history(query)

# Get all notifications for a recipient
query = NotificationQuery(
    recipient_email="user@example.com",
    limit=100,
)
user_notifications = await adapter.get_notification_history(query)

# Paginated query
query = NotificationQuery(limit=20, offset=40)  # Page 3 (20 per page)
page_3 = await adapter.get_notification_history(query)
```

## Common Patterns

### Template-Based Email Notifications

```python
from portico.ports.notification import (
    NotificationAdapter,
    NotificationRequest,
    NotificationChannel,
    NotificationRecipient,
)
from portico.ports.template import CreateTemplateRequest, TemplateTypes

async def send_welcome_email(
    adapter: NotificationAdapter,
    template_registry: TemplateRegistry,
    user_email: str,
    user_name: str,
):
    # Create template (once, typically during app initialization)
    template = await template_registry.create(
        CreateTemplateRequest(
            name="welcome_email",
            template_type=TemplateTypes.NOTIFICATION_EMAIL,
            content="""{% raw %}Hello {{ user_name }}!

Welcome to {{ app_name }}. We're excited to have you on board!

{% if is_premium %}
Thank you for choosing our premium plan.
{% endif %}

Best regards,
The {{ app_name }} Team{% endraw %}""",
            variables=["user_name", "app_name", "is_premium"],
            metadata={"subject": "{% raw %}Welcome to {{ app_name }}!{% endraw %}"},
        )
    )

    # Send notification using template
    result = await adapter.send(
        NotificationRequest(
            template_name="welcome_email",
            channel=NotificationChannel.EMAIL,
            recipient=NotificationRecipient(
                email=user_email,
                name=user_name,
            ),
            variables={
                "user_name": user_name,
                "app_name": "MyApp",
                "is_premium": False,
            },
        )
    )

    return result
```

### SMS Verification Codes

```python
async def send_verification_code(
    adapter: NotificationAdapter,
    phone: str,
    code: str,
):
    # Direct SMS (no template)
    result = await adapter.send(
        NotificationRequest(
            channel=NotificationChannel.SMS,
            recipient=NotificationRecipient(phone=phone),
            body=f"Your verification code is {code}. Valid for 5 minutes.",
        )
    )

    if result.status != NotificationStatus.SENT:
        raise Exception(f"Failed to send SMS: {result.error_message}")

    return result
```

### Batch Notifications with Error Handling

```python
async def send_newsletter_batch(
    adapter: NotificationAdapter,
    subscriber_emails: List[str],
    newsletter_content: Dict[str, Any],
):
    # Prepare batch requests
    requests = [
        NotificationRequest(
            template_name="newsletter",
            channel=NotificationChannel.EMAIL,
            recipient=NotificationRecipient(email=email),
            variables=newsletter_content,
        )
        for email in subscriber_emails
    ]

    # Send batch
    results = await adapter.send_batch(requests)

    # Analyze results
    successful = [r for r in results if r.status == NotificationStatus.SENT]
    failed = [r for r in results if r.status == NotificationStatus.FAILED]

    # Log failures for retry
    for result in failed:
        logger.error(
            "newsletter_send_failed",
            recipient=result.recipient.email,
            error=result.error_message,
        )

    return {
        "total": len(results),
        "sent": len(successful),
        "failed": len(failed),
        "failed_recipients": [r.recipient.email for r in failed],
    }
```

### Notification History and Auditing

```python
async def get_user_notification_history(
    adapter: NotificationAdapter,
    user_email: str,
    days: int = 30,
):
    # Query user's notification history
    query = NotificationQuery(
        recipient_email=user_email,
        start_date=datetime.now() - timedelta(days=days),
        limit=100,
    )

    history = await adapter.get_notification_history(query)

    # Group by status
    sent = [n for n in history if n.status == NotificationStatus.SENT]
    failed = [n for n in history if n.status == NotificationStatus.FAILED]

    return {
        "user_email": user_email,
        "period_days": days,
        "total_notifications": len(history),
        "sent_count": len(sent),
        "failed_count": len(failed),
        "notifications": history,
    }
```

### Jinja2 Template Features

```python
# Conditionals
template_content = """{% raw %}
Hello {{ name }}!

{% if is_premium %}
Thank you for being a premium member! Your account expires on {{ expiry_date }}.
{% else %}
Upgrade to premium to unlock exclusive features!
{% endif %}
{% endraw %}"""

# Loops
template_content = """{% raw %}
Your recent orders:

{% for order in orders %}
- Order #{{ order.id }}: ${{ order.total }} ({{ order.status }})
{% endfor %}

Total: ${{ total }}
{% endraw %}"""

# Filters
template_content = """{% raw %}
Name: {{ name|upper }}
Price: ${{ price|round(2) }}
Date: {{ date|date_format('%Y-%m-%d') }}
{% endraw %}"""

# Use in notification
result = await adapter.send(
    NotificationRequest(
        template_name="order_summary",
        channel=NotificationChannel.EMAIL,
        recipient=NotificationRecipient(email=user.email),
        variables={
            "name": user.name,
            "orders": user.recent_orders,
            "total": sum(o.total for o in user.recent_orders),
        },
    )
)
```

## Integration with Kits

The Notification Port integrates with template management for reusable notification templates.

```python
from portico import compose

# Note: No dedicated notification kit yet - use adapters directly
# Templates are managed by TemplateRegistry

# Setup example (conceptual)
from portico.adapters.notification import MemoryNotificationAdapter
from portico.adapters.template import MemoryTemplateRegistry

template_registry = MemoryTemplateRegistry()
notification_adapter = MemoryNotificationAdapter(
    template_registry=template_registry
)

# Create templates
await template_registry.create(
    CreateTemplateRequest(
        name="welcome_email",
        template_type=TemplateTypes.NOTIFICATION_EMAIL,
        content="{% raw %}Welcome {{ user_name }}!{% endraw %}",
        variables=["user_name"],
        metadata={"subject": "Welcome!"},
    )
)

# Send notifications
result = await notification_adapter.send(
    NotificationRequest(
        template_name="welcome_email",
        channel=NotificationChannel.EMAIL,
        recipient=NotificationRecipient(email="user@example.com"),
        variables={"user_name": "Alice"},
    )
)
```

## Best Practices

1. **Validate Recipients Before Sending**: Use `validate_for_channel()` to check recipients have required info

   ```python
   # ✅ GOOD - Validate before sending
   recipient = NotificationRecipient(email="user@example.com")
   if not recipient.validate_for_channel(NotificationChannel.EMAIL):
       raise ValueError("Recipient missing email address")

   result = await adapter.send(request)

   # ❌ BAD - No validation, will fail at send time
   recipient = NotificationRecipient(phone="+1234567890")  # No email
   result = await adapter.send(
       NotificationRequest(
           channel=NotificationChannel.EMAIL,  # Requires email!
           recipient=recipient,
           body="Test",
       )
   )
   ```

2. **Use Templates for Consistent Messaging**: Create reusable templates instead of hardcoding message content

   ```python
   # ✅ GOOD - Template-based
   result = await adapter.send(
       NotificationRequest(
           template_name="order_confirmation",
           channel=NotificationChannel.EMAIL,
           recipient=recipient,
           variables={"order_id": order.id, "total": order.total},
       )
   )

   # ❌ BAD - Hardcoded content duplicated everywhere
   result = await adapter.send(
       NotificationRequest(
           channel=NotificationChannel.EMAIL,
           recipient=recipient,
           subject="Order Confirmation",
           body=f"Your order #{order.id} for ${order.total} has been confirmed.",
       )
   )
   ```

3. **Handle Failures Gracefully**: Check status and handle errors appropriately

   ```python
   # ✅ GOOD - Check status and handle failures
   result = await adapter.send(request)

   if result.status == NotificationStatus.SENT:
       logger.info("notification_sent", recipient=result.recipient.email)
   else:
       logger.error(
           "notification_failed",
           recipient=result.recipient.email,
           error=result.error_message,
       )
       # Implement retry logic or fallback channel
       await retry_notification(request)

   # ❌ BAD - Assume success
   result = await adapter.send(request)
   # No status check - might have failed silently
   ```

4. **Use Batch Operations for Efficiency**: Send multiple notifications in a single call

   ```python
   # ✅ GOOD - Batch send
   results = await adapter.send_batch(requests)

   # ❌ BAD - Individual sends in loop
   results = []
   for request in requests:
       result = await adapter.send(request)
       results.append(result)
   ```

5. **Include Metadata for Tracking**: Add metadata to requests and recipients for debugging and analytics

   ```python
   # ✅ GOOD - Rich metadata
   result = await adapter.send(
       NotificationRequest(
           template_name="alert",
           channel=NotificationChannel.EMAIL,
           recipient=NotificationRecipient(
               email=user.email,
               metadata={"user_id": str(user.id), "account_type": "premium"},
           ),
           variables={"alert_message": alert.message},
           metadata={
               "alert_id": str(alert.id),
               "priority": "high",
               "category": "security",
           },
       )
   )

   # ❌ BAD - No metadata
   result = await adapter.send(
       NotificationRequest(
           template_name="alert",
           channel=NotificationChannel.EMAIL,
           recipient=NotificationRecipient(email=user.email),
           variables={"alert_message": alert.message},
       )
   )
   ```

6. **Query History for Auditing**: Regularly check notification history for delivery issues

   ```python
   # ✅ GOOD - Monitor failed notifications
   query = NotificationQuery(
       status=NotificationStatus.FAILED,
       start_date=datetime.now() - timedelta(hours=24),
   )
   failed_notifications = await adapter.get_notification_history(query)

   if len(failed_notifications) > threshold:
       alert_ops_team(f"High failure rate: {len(failed_notifications)} failures")
   ```

7. **Use Phone Number Validation**: Validate phone numbers are in E.164 format for SMS

   ```python
   # ✅ GOOD - E.164 format validation
   import phonenumbers

   def validate_phone(phone: str) -> str:
       try:
           parsed = phonenumbers.parse(phone, "US")
           if not phonenumbers.is_valid_number(parsed):
               raise ValueError("Invalid phone number")
           return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
       except Exception as e:
           raise ValueError(f"Invalid phone format: {e}")

   validated_phone = validate_phone("+1-234-567-8900")  # Returns "+12345678900"
   recipient = NotificationRecipient(phone=validated_phone)

   # ❌ BAD - Raw phone format
   recipient = NotificationRecipient(phone="(234) 567-8900")
   ```

## FAQs

### What's the difference between template_id and template_name?

Both identify templates, but `template_id` uses the UUID for direct lookup, while `template_name` searches by name. Use `template_name` for readability and `template_id` when you have the UUID cached:

```python
# By name (more readable)
request = NotificationRequest(
    template_name="welcome_email",
    channel=NotificationChannel.EMAIL,
    recipient=recipient,
)

# By ID (faster lookup if you have it)
request = NotificationRequest(
    template_id=uuid.UUID("..."),
    channel=NotificationChannel.EMAIL,
    recipient=recipient,
)
```

### How do I send notifications without templates?

Use the `subject` and `body` fields for direct messages:

```python
result = await adapter.send(
    NotificationRequest(
        channel=NotificationChannel.EMAIL,
        recipient=NotificationRecipient(email="user@example.com"),
        subject="Your order has shipped",
        body="Order #12345 has shipped via FedEx. Tracking: 1234567890.",
    )
)
```

### Can I override the template subject?

Yes, use the `subject` field in `NotificationRequest` to override the template's default subject:

```python
result = await adapter.send(
    NotificationRequest(
        template_name="alert",  # Has default subject "Alert"
        channel=NotificationChannel.EMAIL,
        recipient=recipient,
        subject="URGENT: Security Alert",  # Overrides template subject
        variables={"message": "Unusual login detected"},
    )
)
```

### How do I implement retry logic for failed notifications?

Check the result status and implement exponential backoff:

```python
import asyncio

async def send_with_retry(adapter, request, max_retries=3):
    for attempt in range(max_retries):
        result = await adapter.send(request)

        if result.status == NotificationStatus.SENT:
            return result

        # Exponential backoff
        wait_time = 2 ** attempt
        logger.warning(
            f"Notification failed, retrying in {wait_time}s",
            attempt=attempt + 1,
            error=result.error_message,
        )
        await asyncio.sleep(wait_time)

    raise Exception(f"Failed to send notification after {max_retries} attempts")
```

### How do I send multi-channel notifications (email + SMS)?

Send separate notifications for each channel:

```python
recipient = NotificationRecipient(
    email="user@example.com",
    phone="+1234567890",
    name="Alice",
)

# Send email
email_result = await adapter.send(
    NotificationRequest(
        template_name="alert_email",
        channel=NotificationChannel.EMAIL,
        recipient=recipient,
        variables={"message": "Your account was accessed from a new device"},
    )
)

# Send SMS
sms_result = await adapter.send(
    NotificationRequest(
        channel=NotificationChannel.SMS,
        recipient=recipient,
        body="Alert: Your account was accessed from a new device. Check your email for details.",
    )
)
```

### How do I track notification delivery rates?

Query notification history and calculate metrics:

```python
from datetime import datetime, timedelta

async def get_delivery_metrics(adapter, days=7):
    query = NotificationQuery(
        start_date=datetime.now() - timedelta(days=days),
        limit=1000,
    )
    history = await adapter.get_notification_history(query)

    total = len(history)
    sent = sum(1 for n in history if n.status == NotificationStatus.SENT)
    failed = sum(1 for n in history if n.status == NotificationStatus.FAILED)

    return {
        "period_days": days,
        "total": total,
        "sent": sent,
        "failed": failed,
        "delivery_rate": (sent / total * 100) if total > 0 else 0,
        "by_channel": {
            "email": sum(1 for n in history if n.channel == NotificationChannel.EMAIL),
            "sms": sum(1 for n in history if n.channel == NotificationChannel.SMS),
        },
    }

metrics = await get_delivery_metrics(adapter)
print(f"Delivery rate: {metrics['delivery_rate']:.1f}%")
```

### What Jinja2 features are supported in templates?

Full Jinja2 templating is supported including:

- **Variables**: `{% raw %}{{ variable_name }}{% endraw %}`
- **Conditionals**: `{% raw %}{% if condition %} ... {% else %} ... {% endif %}{% endraw %}`
- **Loops**: `{% raw %}{% for item in items %} ... {% endfor %}{% endraw %}`
- **Filters**: `{% raw %}{{ value|upper }}{% endraw %}`, `{% raw %}{{ price|round(2) }}{% endraw %}`, `{% raw %}{{ date|date_format('%Y-%m-%d') }}{% endraw %}`
- **Comments**: `{% raw %}{# This is a comment #}{% endraw %}`

Example with all features:

```python
template_content = """{% raw %}
Hello {{ name|title }}!  {# Use title case filter #}

{% if is_premium %}
Premium Account Status: Active
Expires: {{ expiry_date|date_format('%B %d, %Y') }}
{% else %}
Upgrade to Premium today!
{% endif %}

Your Recent Orders:
{% for order in orders %}
  - Order #{{ order.id }}: ${{ order.total|round(2) }}
{% endfor %}

Total: ${{ orders|sum(attribute='total')|round(2) }}
{% endraw %}"""
```
