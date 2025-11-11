# File Storage Kit

## Overview

**Purpose**: Provide secure file upload, storage, and retrieval with metadata tracking, access control, and multiple storage backend support (local filesystem, Google Cloud Storage, or memory).

**Key Features**:

- Multiple storage backends: local, Google Cloud Storage (GCS), or in-memory
- Database metadata tracking for all files
- Owner-based access control (user or group ownership)
- File size limits and content type restrictions
- Public and private file support
- Search and filtering capabilities
- Automatic MIME type detection
- Tagging and categorization

**Dependencies**:

- **Injected services**: None
- **Port dependencies**: FileRepository (metadata storage), FileStorageAdapter (file content storage)
- **Note**: Kits cannot directly import from other kits (enforced by import-linter contract #6). Dependencies are injected via constructor in `compose.py`.

## Quick Start

```python
from portico import compose
from portico.ports.file_storage import FileUploadRequest, FileOwnerType

# Basic configuration with local storage
app = compose.webapp(
    database_url="postgresql://localhost/myapp",
    kits=[
        compose.user(),
        compose.file(
            storage_backend="local",
            storage_path="./uploads",
            max_file_size_mb=50,
        ),
    ]
)

# Access the file service
file_service = app.kits["file"].service

# Upload a file
with open("document.pdf", "rb") as f:
    metadata = await file_service.upload_file(
        file_content=f,
        upload_request=FileUploadRequest(
            filename="document.pdf",
            content_type="application/pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            is_public=False,
        )
    )

# Retrieve the file
file_content = await file_service.get_file(
    file_id=metadata.id,
    requesting_user_id=user_id
)
```

## Core Concepts

### Storage Backends

The File Storage Kit supports three storage backends:

```python
# Local filesystem - files stored in directory
compose.file(
    storage_backend="local",
    storage_path="./uploads",  # Required for local
    max_file_size_mb=100,
)

# Google Cloud Storage - files stored in GCS bucket
compose.file(
    storage_backend="gcs",
    gcs_bucket="my-app-files",  # Required for GCS
    gcs_project="my-gcp-project",  # Required for GCS
    gcs_credentials_path="/path/to/credentials.json",  # Optional
    max_file_size_mb=500,
)

# In-memory - files stored in RAM (development/testing only)
compose.file(
    storage_backend="memory",
    max_file_size_mb=10,
)
```

**Local Storage**:

- Files stored in configured directory
- Fast access, no external dependencies
- Not suitable for multi-server deployments
- Best for: Single-server apps, development

**GCS Storage**:

- Files stored in Google Cloud Storage bucket
- Scalable, durable, distributed
- Requires GCP credentials
- Best for: Production multi-server apps, high availability

**Memory Storage**:

- Files stored in RAM
- Very fast but volatile (lost on restart)
- Limited by available memory
- Best for: Testing, temporary files

### File Ownership and Access Control

Files are owned by either a user or a group:

```python
from portico.ports.file_storage import FileOwnerType

# User-owned file (private by default)
upload_request = FileUploadRequest(
    filename="resume.pdf",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    is_public=False,  # Only owner can access
)

# Group-owned file
upload_request = FileUploadRequest(
    filename="team-doc.pdf",
    owner_type=FileOwnerType.GROUP,
    owner_id=group_id,
    is_public=False,  # Only group members can access (TODO)
)

# Public file (anyone can access)
upload_request = FileUploadRequest(
    filename="public-doc.pdf",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    is_public=True,  # Anyone can read
)
```

**Access rules:**

- **Owner** can always read, update, and delete their files
- **Public files** can be read by anyone (but only owner can modify/delete)
- **Private files** can only be accessed by owner
- **Group files** require group membership check (TODO: requires Group Kit integration)

### File Metadata

All files have metadata stored in the database:

```python
# Upload returns metadata
metadata = await file_service.upload_file(file_content, upload_request)

# Metadata fields
print(metadata.id)  # UUID
print(metadata.filename)  # "document.pdf"
print(metadata.original_filename)  # "document.pdf"
print(metadata.content_type)  # "application/pdf"
print(metadata.size_bytes)  # 1024000
print(metadata.owner_type)  # FileOwnerType.USER
print(metadata.owner_id)  # UUID
print(metadata.storage_path)  # Internal path (don't use directly)
print(metadata.is_public)  # False
print(metadata.description)  # Optional description
print(metadata.tags)  # ["invoice", "2024"]
print(metadata.created_at)  # datetime
print(metadata.updated_at)  # datetime
```

### File Operations

Upload, retrieve, update, delete files with built-in access control:

```python
# Upload file
metadata = await file_service.upload_file(
    file_content=file_bytes,
    upload_request=FileUploadRequest(...)
)

# Get file with content
file_content = await file_service.get_file(
    file_id=metadata.id,
    requesting_user_id=user_id
)
# file_content.metadata, file_content.content

# Get metadata only (no file content download)
metadata = await file_service.get_file_metadata(
    file_id=file_id,
    requesting_user_id=user_id
)

# Update metadata
updated = await file_service.update_file_metadata(
    file_id=file_id,
    update_request=FileUpdateRequest(
        description="Updated description",
        tags=["updated", "2024"]
    ),
    requesting_user_id=user_id
)

# Delete file
success = await file_service.delete_file(
    file_id=file_id,
    requesting_user_id=user_id
)
```

### File Search and Listing

Find files by owner, search query, or public access:

```python
# List user's files
user_files = await file_service.list_files_by_owner(
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    requesting_user_id=user_id,
    limit=50,
    offset=0
)

# List group's files
group_files = await file_service.list_files_by_owner(
    owner_type=FileOwnerType.GROUP,
    owner_id=group_id,
    requesting_user_id=user_id,
    limit=50
)

# List all public files
public_files = await file_service.list_public_files(
    limit=100,
    offset=0
)

# Search files (by filename, description, tags)
results = await file_service.search_files(
    query="invoice",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    requesting_user_id=user_id,
    limit=20
)
```

## Configuration

### Required Settings

Storage backend must be configured with backend-specific settings.

**For local storage:**

| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `storage_backend` | `"local"` | Yes | Use local filesystem |
| `storage_path` | `str` | Yes | Directory path for file storage |

**For GCS storage:**

| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `storage_backend` | `"gcs"` | Yes | Use Google Cloud Storage |
| `gcs_bucket` | `str` | Yes | GCS bucket name |
| `gcs_project` | `str` | Yes | GCP project ID |
| `gcs_credentials_path` | `str` | No | Path to service account JSON |

**For memory storage:**

| Setting | Type | Required | Description |
|---------|------|----------|-------------|
| `storage_backend` | `"memory"` | Yes | Use in-memory storage |

### Optional Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_file_size_mb` | `int` | `100` | Maximum file size in megabytes |
| `allowed_content_types` | `set[str] \| None` | `None` | Set of allowed MIME types (None = all allowed) |

**Example Configurations:**

```python
from portico import compose

# Local storage for development
compose.file(
    storage_backend="local",
    storage_path="./uploads",
    max_file_size_mb=50,
)

# GCS for production
compose.file(
    storage_backend="gcs",
    gcs_bucket="myapp-production-files",
    gcs_project="myapp-prod",
    gcs_credentials_path="/secrets/gcs-key.json",
    max_file_size_mb=500,
)

# Restrict content types (images only)
compose.file(
    storage_backend="local",
    storage_path="./images",
    max_file_size_mb=10,
    allowed_content_types={"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Memory for testing
compose.file(
    storage_backend="memory",
    max_file_size_mb=5,
)
```

## Usage Examples

### Example 1: File Upload Endpoint

```python
from fastapi import FastAPI, UploadFile, File
from portico.ports.file_storage import FileUploadRequest, FileOwnerType
from portico.kits.fastapi import Dependencies

deps = Dependencies(app)

@app.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    user = deps.current_user
):
    file_service = deps.webapp.kits["file"].service

    # Read file content
    content = await file.read()

    # Upload with metadata
    metadata = await file_service.upload_file(
        file_content=content,
        upload_request=FileUploadRequest(
            filename=file.filename,
            content_type=file.content_type,
            owner_type=FileOwnerType.USER,
            owner_id=user.id,
            is_public=False,
        )
    )

    return {
        "file_id": str(metadata.id),
        "filename": metadata.filename,
        "size_bytes": metadata.size_bytes,
        "url": f"/files/{metadata.id}"
    }
```

### Example 2: File Download Endpoint

```python
from fastapi import Response
from portico.exceptions import FileNotFoundError, FileAccessError

@app.get("/files/{file_id}")
async def download_file(
    file_id: UUID,
    user = deps.optional_user
):
    file_service = deps.webapp.kits["file"].service

    try:
        # Get file with access control
        file_content = await file_service.get_file(
            file_id=file_id,
            requesting_user_id=user.id if user else None
        )

        # Return file as response
        return Response(
            content=file_content.content,
            media_type=file_content.metadata.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file_content.metadata.filename}"'
            }
        )
    except FileNotFoundError:
        raise HTTPException(404, "File not found")
    except FileAccessError:
        raise HTTPException(403, "Access denied")
```

### Example 3: User File Gallery

```python
@app.get("/users/{user_id}/files")
async def list_user_files(
    user_id: UUID,
    current_user = deps.current_user,
    limit: int = 50,
    offset: int = 0
):
    file_service = deps.webapp.kits["file"].service

    # List user's files (with access control)
    files = await file_service.list_files_by_owner(
        owner_type=FileOwnerType.USER,
        owner_id=user_id,
        requesting_user_id=current_user.id,
        limit=limit,
        offset=offset
    )

    return {
        "files": [
            {
                "id": str(f.id),
                "filename": f.filename,
                "size_bytes": f.size_bytes,
                "content_type": f.content_type,
                "created_at": f.created_at.isoformat(),
                "is_public": f.is_public,
            }
            for f in files
        ],
        "count": len(files)
    }
```

### Example 4: File Search

```python
@app.get("/files/search")
async def search_files(
    q: str,
    current_user = deps.current_user,
    limit: int = 20
):
    file_service = deps.webapp.kits["file"].service

    # Search across user's files
    results = await file_service.search_files(
        query=q,
        owner_type=FileOwnerType.USER,
        owner_id=current_user.id,
        requesting_user_id=current_user.id,
        limit=limit
    )

    return {
        "query": q,
        "results": [
            {
                "id": str(f.id),
                "filename": f.filename,
                "description": f.description,
                "tags": f.tags,
            }
            for f in results
        ]
    }
```

### Example 5: File Metadata Update

```python
from pydantic import BaseModel

class UpdateFileRequest(BaseModel):
    description: str | None = None
    tags: list[str] | None = None
    is_public: bool | None = None

@app.patch("/files/{file_id}")
async def update_file(
    file_id: UUID,
    update_data: UpdateFileRequest,
    user = deps.current_user
):
    file_service = deps.webapp.kits["file"].service

    # Update metadata (with access control)
    updated = await file_service.update_file_metadata(
        file_id=file_id,
        update_request=FileUpdateRequest(
            description=update_data.description,
            tags=update_data.tags,
            is_public=update_data.is_public
        ),
        requesting_user_id=user.id
    )

    return {
        "file_id": str(updated.id),
        "description": updated.description,
        "tags": updated.tags,
        "is_public": updated.is_public
    }
```

## Domain Models

### FileMetadata

Represents file metadata stored in database.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `UUID` | Auto | Unique file identifier |
| `filename` | `str` | - | Current filename |
| `original_filename` | `str` | - | Original filename at upload |
| `content_type` | `str` | - | MIME type (e.g., "image/png") |
| `size_bytes` | `int` | - | File size in bytes |
| `owner_type` | `FileOwnerType` | - | "user" or "group" |
| `owner_id` | `UUID` | - | UUID of owner |
| `storage_path` | `str \| None` | - | Internal storage path (don't use directly) |
| `is_public` | `bool` | `False` | Whether file is publicly accessible |
| `description` | `str \| None` | `None` | Optional file description |
| `tags` | `List[str]` | `[]` | Tags for categorization |
| `created_at` | `datetime` | Auto | When file was uploaded (UTC) |
| `updated_at` | `datetime` | Auto | When metadata was last updated (UTC) |

### FileUploadRequest

Request model for uploading a file.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `filename` | `str` | - | Filename to store |
| `content_type` | `str \| None` | Auto-detect | MIME type |
| `owner_type` | `FileOwnerType` | - | "user" or "group" |
| `owner_id` | `UUID` | - | UUID of owner |
| `is_public` | `bool` | `False` | Make file publicly accessible |
| `description` | `str \| None` | `None` | File description |
| `tags` | `List[str]` | `[]` | Tags for categorization |

### FileUpdateRequest

Request model for updating file metadata.

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str \| None` | New filename |
| `is_public` | `bool \| None` | Change public access |
| `description` | `str \| None` | New description |
| `tags` | `List[str] \| None` | New tags (replaces existing) |

### FileContent

File content with metadata (returned by get_file).

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | `FileMetadata` | File metadata |
| `content` | `bytes` | File binary content |

### FileOwnerType

Enum for file owner types.

| Value | Description |
|-------|-------------|
| `USER` | File owned by individual user |
| `GROUP` | File owned by group |

## Database Models

### FileMetadataModel

**Table**: `file_metadata`

**Columns**:

- `id`: UUID, primary key
- `filename`: String(255)
- `original_filename`: String(255)
- `content_type`: String(100)
- `size_bytes`: BigInteger
- `owner_type`: String(20) - "user" or "group"
- `owner_id`: UUID
- `storage_path`: String(500), nullable
- `is_public`: Boolean, default False
- `description`: Text, nullable
- `tags`: JSON (array of strings)
- `created_at`: DateTime with timezone
- `updated_at`: DateTime with timezone

**Indexes**:

- `idx_file_metadata_owner`: On `owner_type`, `owner_id` columns
- `idx_file_metadata_public`: On `is_public` column

## Events

This kit does not currently publish events, but could be extended to publish:

- `FileUploadedEvent` - When file is uploaded
- `FileDeletedEvent` - When file is deleted
- `FileAccessedEvent` - When file is accessed (for analytics)

## Best Practices

### 1. Always Use Access Control

Pass requesting_user_id to enforce access control:

```python
# ✅ GOOD - Access control enforced
file_content = await file_service.get_file(
    file_id=file_id,
    requesting_user_id=current_user.id
)
# Raises FileAccessError if user doesn't have permission

# ❌ BAD - No access control
file_content = await file_service.get_file(
    file_id=file_id,
    requesting_user_id=None  # Bypasses access control!
)
# Only works for public files
```

### 2. Set Appropriate File Size Limits

Configure limits based on your application needs:

```python
# ✅ GOOD - Reasonable limits per use case
# Image uploads
compose.file(storage_backend="local", max_file_size_mb=10)

# Document uploads
compose.file(storage_backend="local", max_file_size_mb=50)

# Video uploads
compose.file(storage_backend="gcs", max_file_size_mb=500)

# ❌ BAD - No limit or too large
compose.file(storage_backend="local", max_file_size_mb=10000)
# Can exhaust disk space
```

### 3. Restrict Content Types for Security

Limit allowed file types to prevent malicious uploads:

```python
# ✅ GOOD - Whitelist allowed types
compose.file(
    storage_backend="local",
    storage_path="./uploads",
    allowed_content_types={
        "image/jpeg", "image/png", "image/gif",
        "application/pdf",
        "text/plain",
    }
)
# Rejects executable files, scripts, etc.

# ❌ BAD - No restrictions
compose.file(storage_backend="local", storage_path="./uploads")
# Allows any file type, including malware
```

### 4. Use GCS for Production

Local storage doesn't scale across multiple servers:

```python
# ✅ GOOD - GCS for production
if os.getenv("ENVIRONMENT") == "production":
    compose.file(
        storage_backend="gcs",
        gcs_bucket="myapp-prod-files",
        gcs_project="myapp-prod"
    )
else:
    compose.file(
        storage_backend="local",
        storage_path="./uploads"
    )

# ❌ BAD - Local storage in multi-server production
compose.file(storage_backend="local", storage_path="./uploads")
# Files not shared across servers
```

### 5. Use Tags for Organization

Tag files for easier search and filtering:

```python
# ✅ GOOD - Descriptive tags
upload_request = FileUploadRequest(
    filename="invoice.pdf",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    tags=["invoice", "2024", "Q1", "client-ABC"]
)
# Easy to find: "Find all invoices from Q1 2024"

# ❌ BAD - No tags
upload_request = FileUploadRequest(
    filename="invoice.pdf",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    tags=[]  # Hard to organize and search
)
```

### 6. Handle Upload Errors Gracefully

Provide clear error messages to users:

```python
# ✅ GOOD - Specific error handling
from portico.exceptions import FileSizeExceededError, FileUploadError

try:
    metadata = await file_service.upload_file(content, upload_request)
    return {"success": True, "file_id": str(metadata.id)}
except FileSizeExceededError as e:
    return {"error": f"File too large (max {e.max_size_bytes / 1024 / 1024}MB)"}
except FileUploadError as e:
    return {"error": f"Upload failed: {e.message}"}

# ❌ BAD - Generic error
try:
    metadata = await file_service.upload_file(content, upload_request)
except Exception:
    return {"error": "Upload failed"}  # No details for user
```

### 7. Clean Up Deleted Files

Ensure both metadata and content are deleted:

```python
# ✅ GOOD - Service handles both
success = await file_service.delete_file(
    file_id=file_id,
    requesting_user_id=user_id
)
# Deletes both database record and storage file

# ❌ BAD - Manual deletion (error-prone)
await file_repository.delete_metadata(file_id)
# Forgot to delete from storage! Orphaned file
```

## Security Considerations

### File Upload Security

Validate and sanitize uploaded files:

- **Limit file sizes** to prevent DoS attacks
- **Restrict content types** to prevent malicious file uploads
- **Scan files** for malware (consider integrating antivirus)
- **Generate unique filenames** to prevent path traversal attacks

### Access Control

Always enforce ownership checks:

```python
# Check ownership before sensitive operations
metadata = await file_service.get_file_metadata(file_id, user.id)
if metadata.owner_type == FileOwnerType.USER and metadata.owner_id != user.id:
    raise HTTPException(403, "Not your file")
```

### Storage Path Security

Never expose internal storage paths to users:

```python
# Use file IDs, not paths
url = f"/files/{metadata.id}"  # ✅ Safe

# Never do this:
url = f"/files/{metadata.storage_path}"  # ❌ Exposes internal structure
```

### Public File Considerations

Be careful with public files:

- Don't store sensitive data in public files
- Consider adding watermarks or DRM for copyrighted content
- Monitor public file access for abuse
- Implement rate limiting on public file downloads

## FAQs

### Q: How do I migrate from local to GCS storage?

A: Files must be manually copied to GCS. Steps:

1. Upload all local files to GCS bucket
2. Update `storage_path` in database to match GCS paths
3. Change configuration to `storage_backend="gcs"`
4. Test file retrieval
5. Remove local files after verification

### Q: Can I use S3 instead of GCS?

A: Not currently. The kit supports local, GCS, and memory backends. To add S3:

1. Implement `S3FileStorageAdapter` conforming to `FileStorageAdapter` port
2. Add S3 backend option to `compose.file()`
3. Contribute back to Portico!

### Q: How are file permissions checked for group files?

A: Currently, group file permissions require Group Kit integration (marked as TODO in code). Basic check: owner can access. Full implementation requires checking group membership.

### Q: What happens if I upload a file with the same name?

A: Each upload creates a new file with unique ID. Filename is just metadata. Multiple files can have the same filename but different IDs.

### Q: How do I serve files through a CDN?

A: Extend `get_file_url()` to generate CDN URLs:

```python
# Override in custom service
async def get_file_url(self, file_id: UUID) -> str:
    metadata = await self.file_repository.get_metadata_by_id(file_id)
    return f"https://cdn.example.com/files/{metadata.storage_path}"
```

### Q: Can I store file thumbnails or previews?

A: Not directly. Store thumbnails as separate files with tags linking them:

```python
# Upload original
original = await file_service.upload_file(...)

# Upload thumbnail
thumbnail = await file_service.upload_file(
    thumbnail_bytes,
    FileUploadRequest(
        filename=f"thumb_{original.filename}",
        tags=["thumbnail", f"original:{original.id}"]
    )
)
```

### Q: How do I handle large file uploads?

A: Consider streaming uploads and chunking:

1. Use FastAPI's streaming request body
2. Stream directly to storage adapter
3. Validate size incrementally
4. For very large files, consider presigned upload URLs (GCS/S3)

### Q: What's the performance impact of metadata tracking?

A: Minimal. Each upload requires one database INSERT. Retrieval requires one SELECT. Database lookups are fast compared to file I/O.

### Q: Can I use the File Kit without a database?

A: No, the kit requires database for metadata tracking. For pure storage without metadata, use the storage adapters directly (not recommended - loses access control and search).

### Q: How do I implement file versioning?

A: Not built-in. Implement by:

1. Add `version` field to metadata
2. Don't delete old versions, mark as archived
3. Store multiple files with same `original_filename` but different versions
4. Add `parent_file_id` to link versions
