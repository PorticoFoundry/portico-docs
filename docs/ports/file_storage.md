# File Storage Port

## Overview

The File Storage Port defines the contract for file upload, storage, and retrieval in Portico applications.

**Purpose**: Abstract file storage operations to enable pluggable storage backends while maintaining metadata in a database.

**Domain**: File management, binary storage, access control

**Key Capabilities**:

- Binary file upload and download with metadata tracking
- User and group file ownership
- Public and private file access control
- File search and listing with pagination
- Multiple storage backends (local filesystem, cloud storage, database blobs)
- Content type detection and validation
- File size limits and validation
- Tag-based file organization

**Port Type**: Repository + Adapter (dual interface)

**When to Use**:

- Applications requiring user file uploads (documents, images, etc.)
- Multi-tenant systems with per-user or per-group file storage
- Systems requiring cloud storage integration (GCS, S3, etc.)
- Applications with public file sharing requirements
- Document management and file organization features

## Domain Models

### FileMetadata

Represents metadata for a stored file. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `UUID` | Yes | `uuid4()` | Unique file identifier |
| `filename` | `str` | Yes | - | Current filename |
| `original_filename` | `str` | Yes | - | Original filename at upload |
| `content_type` | `str` | Yes | - | MIME type (e.g., "image/png") |
| `size_bytes` | `int` | Yes | - | File size in bytes |
| `owner_type` | `FileOwnerType` | Yes | - | Owner type (USER or GROUP) |
| `owner_id` | `UUID` | Yes | - | Owner identifier |
| `storage_path` | `Optional[str]` | No | `None` | Internal storage path (backend-specific) |
| `is_public` | `bool` | No | `False` | Whether file is publicly accessible |
| `description` | `Optional[str]` | No | `None` | Optional file description |
| `tags` | `list[str]` | No | `[]` | Tags for organization and search |
| `created_at` | `datetime` | Yes | Current UTC time | Upload timestamp |
| `updated_at` | `datetime` | Yes | Current UTC time | Last metadata update timestamp |

**Class Methods**:

- `@classmethod detect_content_type(filename: str) -> str` - Detects MIME type from filename extension

**Example**:

```python
from portico.ports.file_storage import FileMetadata, FileOwnerType

metadata = FileMetadata(
    id=uuid4(),
    filename="report.pdf",
    original_filename="Q4-Report-2024.pdf",
    content_type="application/pdf",
    size_bytes=1024000,
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    is_public=False,
    tags=["report", "q4", "2024"]
)

# Detect content type
content_type = FileMetadata.detect_content_type("image.png")  # "image/png"
```

### FileUploadRequest

Request model for uploading a new file.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filename` | `str` | Yes | - | Filename for the uploaded file |
| `content_type` | `Optional[str]` | No | `None` | MIME type (auto-detected if not provided) |
| `owner_type` | `FileOwnerType` | Yes | - | Owner type (USER or GROUP) |
| `owner_id` | `UUID` | Yes | - | Owner identifier |
| `is_public` | `bool` | No | `False` | Whether file is publicly accessible |
| `description` | `Optional[str]` | No | `None` | Optional file description |
| `tags` | `list[str]` | No | `[]` | Tags for organization |

**Methods**:

- `get_content_type() -> str` - Returns content type, detecting from filename if not provided

**Example**:

```python
from portico.ports.file_storage import FileUploadRequest, FileOwnerType

request = FileUploadRequest(
    filename="document.pdf",
    owner_type=FileOwnerType.USER,
    owner_id=user_id,
    is_public=False,
    description="Important document",
    tags=["work", "important"]
)

# Content type auto-detected as "application/pdf"
print(request.get_content_type())
```

### FileUpdateRequest

Request model for updating file metadata. All fields optional for partial updates.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filename` | `Optional[str]` | No | `None` | New filename |
| `is_public` | `Optional[bool]` | No | `None` | New public status |
| `description` | `Optional[str]` | No | `None` | New description |
| `tags` | `Optional[list[str]]` | No | `None` | New tags (replaces existing) |

**Example**:

```python
from portico.ports.file_storage import FileUpdateRequest

# Update filename and make public
request = FileUpdateRequest(
    filename="renamed-document.pdf",
    is_public=True
)
```

### FileContent

Represents file content with metadata for download operations.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `metadata` | `FileMetadata` | Yes | - | File metadata |
| `content` | `bytes` | Yes | - | Binary file content |

**Example**:

```python
from portico.ports.file_storage import FileContent

file_content = await storage_service.get_file(file_id)
print(f"Filename: {file_content.metadata.filename}")
print(f"Size: {file_content.metadata.size_bytes} bytes")

# Save to disk
with open(file_content.metadata.filename, "wb") as f:
    f.write(file_content.content)
```

## Enumerations

### FileOwnerType

Valid owner types for files.

| Value | Description |
|-------|-------------|
| `USER` | File owned by a user |
| `GROUP` | File owned by a group |

**Example**:

```python
from portico.ports.file_storage import FileOwnerType

# User-owned file
owner_type = FileOwnerType.USER

# Group-owned file
owner_type = FileOwnerType.GROUP
```

## Port Interfaces

### FileRepository

The `FileRepository` abstract base class defines the contract for file metadata storage and retrieval.

**Location**: `portico.ports.file_storage.FileRepository`

#### Key Methods

##### create_metadata

```python
async def create_metadata(
    file_id: UUID,
    file_data: FileUploadRequest,
    storage_path: str,
    size_bytes: int
) -> FileMetadata
```

Create file metadata record after file has been stored.

**Parameters**:

- `file_id`: Unique file identifier
- `file_data`: File upload request data
- `storage_path`: Path where file is stored in the storage backend
- `size_bytes`: File size in bytes

**Returns**: Created file metadata.

**Example**:

```python
from uuid import uuid4
from portico.ports.file_storage import FileUploadRequest, FileOwnerType

file_id = uuid4()
storage_path = f"uploads/{file_id}/document.pdf"

metadata = await file_repository.create_metadata(
    file_id=file_id,
    file_data=FileUploadRequest(
        filename="document.pdf",
        owner_type=FileOwnerType.USER,
        owner_id=user_id,
        tags=["document"]
    ),
    storage_path=storage_path,
    size_bytes=102400
)
```

##### get_metadata_by_id

```python
async def get_metadata_by_id(file_id: UUID) -> Optional[FileMetadata]
```

Retrieve file metadata by ID.

**Parameters**:

- `file_id`: File identifier

**Returns**: File metadata if found, None otherwise.

**Example**:

```python
metadata = await file_repository.get_metadata_by_id(file_id)

if metadata:
    print(f"File: {metadata.filename}")
    print(f"Size: {metadata.size_bytes} bytes")
    print(f"Owner: {metadata.owner_type}:{metadata.owner_id}")
else:
    print("File not found")
```

#### Other Methods

##### update_metadata

```python
async def update_metadata(file_id: UUID, update_data: FileUpdateRequest) -> Optional[FileMetadata]
```

Update file metadata. Returns updated metadata if found, None otherwise.

##### delete_metadata

```python
async def delete_metadata(file_id: UUID) -> bool
```

Delete file metadata. Returns True if deleted, False if not found.

##### list_files_by_owner

```python
async def list_files_by_owner(
    owner_type: FileOwnerType,
    owner_id: UUID,
    limit: int = 100,
    offset: int = 0
) -> list[FileMetadata]
```

List files owned by a specific user or group with pagination.

##### list_public_files

```python
async def list_public_files(limit: int = 100, offset: int = 0) -> list[FileMetadata]
```

List all public files with pagination.

##### search_files

```python
async def search_files(
    query: str,
    owner_type: Optional[FileOwnerType] = None,
    owner_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0
) -> list[FileMetadata]
```

Search files by filename or description. Optionally filter by owner.

### FileStorageAdapter

The `FileStorageAdapter` abstract base class defines the contract for physical file storage operations.

**Location**: `portico.ports.file_storage.FileStorageAdapter`

#### Key Methods

##### store_file

```python
async def store_file(file_content: BinaryIO, storage_path: str) -> bool
```

Store file content at the specified path. Primary method for file uploads.

**Parameters**:

- `file_content`: Binary file content to store (file-like object)
- `storage_path`: Path where file should be stored (format depends on backend)

**Returns**: True if stored successfully, False otherwise.

**Example**:

```python
import io
from uuid import uuid4

# Store file from bytes
file_content = b"Hello, World!"
file_stream = io.BytesIO(file_content)

file_id = uuid4()
storage_path = storage_adapter.generate_storage_path(file_id, "hello.txt")

success = await storage_adapter.store_file(file_stream, storage_path)
if success:
    print(f"File stored at: {storage_path}")
```

##### retrieve_file

```python
async def retrieve_file(storage_path: str) -> Optional[bytes]
```

Retrieve file content from storage path. Primary method for file downloads.

**Parameters**:

- `storage_path`: Path to retrieve file from

**Returns**: File content as bytes if found, None otherwise.

**Example**:

```python
content = await storage_adapter.retrieve_file(storage_path)

if content:
    # Save to disk
    with open("downloaded_file.pdf", "wb") as f:
        f.write(content)
else:
    print("File not found in storage")
```

#### Other Methods

##### delete_file

```python
async def delete_file(storage_path: str) -> bool
```

Delete file from storage. Returns True if deleted successfully, False otherwise.

##### file_exists

```python
async def file_exists(storage_path: str) -> bool
```

Check if file exists at storage path. Returns True if file exists, False otherwise.

##### get_file_size

```python
async def get_file_size(storage_path: str) -> Optional[int]
```

Get file size in bytes. Returns file size if found, None otherwise.

##### generate_storage_path

```python
def generate_storage_path(file_id: UUID, filename: str) -> str
```

Generate a storage path for a file. Implementation depends on backend (local path, cloud storage key, etc.).

## Common Patterns

### File Upload with Access Control

```python
from portico.ports.file_storage import FileUploadRequest, FileOwnerType
from portico.exceptions import FileSizeExceededError, FileUploadError

async def upload_user_document(
    user_id: UUID,
    file_content: bytes,
    filename: str,
    file_service: FileStorageService
) -> FileMetadata:
    """Upload a file owned by a user."""

    try:
        # Create upload request
        upload_request = FileUploadRequest(
            filename=filename,
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            is_public=False,  # Private by default
            tags=["user-upload"]
        )

        # Upload file
        metadata = await file_service.upload_file(file_content, upload_request)

        logger.info("file_uploaded", file_id=str(metadata.id), user_id=str(user_id))
        return metadata

    except FileSizeExceededError as e:
        logger.warning("file_too_large", size=e.actual_size, limit=e.max_size)
        raise
    except FileUploadError as e:
        logger.error("upload_failed", error=str(e))
        raise
```

### Secure File Download

```python
from portico.exceptions import FileNotFoundError, FileAccessError

async def download_file_with_access_check(
    file_id: UUID,
    requesting_user_id: UUID,
    file_service: FileStorageService
) -> FileContent:
    """Download file with access control."""

    try:
        # Service checks access permissions automatically
        file_content = await file_service.get_file(
            file_id=file_id,
            requesting_user_id=requesting_user_id
        )

        logger.info("file_downloaded", file_id=str(file_id), user_id=str(requesting_user_id))
        return file_content

    except FileNotFoundError:
        logger.warning("file_not_found", file_id=str(file_id))
        raise
    except FileAccessError:
        logger.warning("access_denied", file_id=str(file_id), user_id=str(requesting_user_id))
        raise
```

## Integration with Kits

The File Storage Port is used by the **File Storage Kit** to provide high-level file management services.

```python
from portico import compose

# Configure with local storage (development)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.file(
            storage_backend="local",
            storage_path="./uploads",
            max_file_size_mb=100
        ),
    ],
)

# Configure with Google Cloud Storage (production)
app = compose.webapp(
    database_url="postgresql+asyncpg://user:pass@localhost/db",
    kits=[
        compose.file(
            storage_backend="gcs",
            gcs_bucket="my-app-files",
            gcs_project="my-gcp-project",
            gcs_credentials_path="/path/to/service-account.json",
            max_file_size_mb=500
        ),
    ],
)

# Access file service
file_service = app.kits["file_storage"].service

# Upload file
with open("document.pdf", "rb") as f:
    metadata = await file_service.upload_file(
        file_content=f,
        upload_request=FileUploadRequest(
            filename="document.pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            is_public=False
        )
    )

# Download file
file_content = await file_service.get_file(
    file_id=metadata.id,
    requesting_user_id=user_id
)

# Save to disk
with open("downloaded.pdf", "wb") as f:
    f.write(file_content.content)
```

The File Storage Kit provides:

- Local filesystem storage adapter
- Google Cloud Storage (GCS) adapter
- Database blob storage adapter
- Automatic content type detection
- File size validation
- User and group ownership with access control
- Public/private file sharing

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Always Validate File Size**: Set appropriate limits to prevent storage abuse and DoS attacks

   ```python
   # ✅ GOOD: Set reasonable limits
   compose.file(storage_backend="local", max_file_size_mb=100)

   # ❌ BAD: No size limits (vulnerable to abuse)
   # Missing max_file_size configuration
   ```

2. **Use Access Control**: Always pass `requesting_user_id` to enforce ownership checks

   ```python
   # ✅ GOOD: Enforce access control
   file_content = await file_service.get_file(
       file_id=file_id,
       requesting_user_id=current_user.id
   )

   # ❌ BAD: No access control
   file_content = await file_service.get_file(file_id)
   ```

3. **Choose Appropriate Storage Backend**: Use local for development, cloud for production

   ```python
   # ✅ GOOD: Local for dev, GCS for prod
   if os.getenv("ENV") == "production":
       backend = "gcs"
       config = {"gcs_bucket": "prod-files", ...}
   else:
       backend = "local"
       config = {"storage_path": "./dev-uploads"}

   # ❌ BAD: Local storage in production (doesn't scale)
   backend = "local"  # Won't work with multiple servers!
   ```

4. **Tag Files for Organization**: Use tags to enable bulk operations and categorization

   ```python
   # ✅ GOOD: Tags enable organization
   FileUploadRequest(
       filename="report.pdf",
       owner_type=FileOwnerType.USER,
       owner_id=user_id,
       tags=["report", "q4-2024", "finance"]
   )

   # Later: search by tag
   results = await file_service.search_files(query="q4-2024")

   # ❌ BAD: No tags (harder to organize)
   FileUploadRequest(filename="report.pdf", owner_type=..., owner_id=...)
   ```

5. **Clean Up Storage on Metadata Deletion**: Ensure both metadata and file content are deleted together

   ```python
   # ✅ GOOD: Service handles both (use FileStorageService.delete_file)
   success = await file_service.delete_file(file_id, requesting_user_id)
   # Deletes both storage and metadata

   # ❌ BAD: Only deleting metadata (orphans file in storage)
   await file_repository.delete_metadata(file_id)
   # File still exists in storage!
   ```

## FAQs

### What storage backends are available?

Portico includes three built-in adapters:

- **LocalFileStorageAdapter**: Stores files on local filesystem (development)
- **GCSFileStorageAdapter**: Google Cloud Storage (production)
- **DatabaseBlobStorageAdapter**: Stores files as BLOBs in database (simple deployments)

Each has trade-offs:
- **Local**: Fast, simple, but doesn't scale across servers
- **GCS**: Scalable, reliable, but requires cloud setup
- **Database**: Simple deployment, but not ideal for large files

### How does file ownership and access control work?

Files have an `owner_type` (USER or GROUP) and `owner_id`. Access rules:

1. **Public files** (`is_public=True`): Anyone can read
2. **User-owned files**: Only owner can read/write
3. **Group-owned files**: Group members can read (requires group membership check)

```python
# Upload as user-owned
FileUploadRequest(owner_type=FileOwnerType.USER, owner_id=user_id)

# Upload as group-owned
FileUploadRequest(owner_type=FileOwnerType.GROUP, owner_id=group_id)
```

### Can I restrict file types?

Yes! Configure `allowed_content_types` when creating the file service:

```python
file_service = FileStorageService(
    file_repository=repo,
    storage_adapter=adapter,
    allowed_content_types={"image/png", "image/jpeg", "application/pdf"}
)

# Now only images and PDFs can be uploaded
```

### What happens if storage fails but metadata succeeds?

The `FileStorageService` stores the file first, then creates metadata. If storage fails, no metadata is created (no orphan records). If metadata creation fails after successful storage, you may have orphaned files in storage.

**Best Practice**: Use the `FileStorageService` which handles the two-phase commit correctly:

```python
# ✅ Service handles errors correctly
metadata = await file_service.upload_file(file_content, upload_request)

# ❌ Don't manually coordinate repository + adapter
# Risk of inconsistency!
```

### How do I generate download URLs for files?

The `FileStorageService` provides a `get_file_url()` method that returns a URL path. You'll need to implement a FastAPI endpoint to serve files:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/files/{file_id}")
async def download_file(
    file_id: UUID,
    current_user: User = deps.current_user
):
    # Get file with access control
    file_content = await file_service.get_file(
        file_id=file_id,
        requesting_user_id=current_user.id
    )

    # Stream response
    return StreamingResponse(
        io.BytesIO(file_content.content),
        media_type=file_content.metadata.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{file_content.metadata.filename}"'
        }
    )
```
