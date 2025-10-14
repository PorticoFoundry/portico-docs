"""Test examples for File Storage port documentation."""

from datetime import UTC, datetime
from io import BytesIO
from typing import BinaryIO, Optional
from uuid import UUID, uuid4

import pytest

from portico.ports.file_storage import (
    FileMetadata,
    FileOwnerType,
    FileRepository,
    FileStorageAdapter,
    FileUpdateRequest,
    FileUploadRequest,
)


class MockFileRepository(FileRepository):
    """Mock file repository for testing."""

    def __init__(self):
        self.files: dict[UUID, FileMetadata] = {}

    async def create_metadata(
        self,
        file_id: UUID,
        file_data: FileUploadRequest,
        storage_path: str,
        size_bytes: int,
    ) -> FileMetadata:
        """Create file metadata record."""
        metadata = FileMetadata(
            id=file_id,
            filename=file_data.filename,
            original_filename=file_data.filename,
            content_type=file_data.get_content_type(),
            size_bytes=size_bytes,
            owner_type=file_data.owner_type,
            owner_id=file_data.owner_id,
            storage_path=storage_path,
            is_public=file_data.is_public,
            description=file_data.description,
            tags=file_data.tags,
        )

        self.files[file_id] = metadata
        return metadata

    async def get_metadata_by_id(self, file_id: UUID) -> Optional[FileMetadata]:
        """Retrieve file metadata by ID."""
        return self.files.get(file_id)

    async def update_metadata(
        self, file_id: UUID, update_data: FileUpdateRequest
    ) -> Optional[FileMetadata]:
        """Update file metadata."""
        metadata = self.files.get(file_id)
        if not metadata:
            return None

        # Create updated metadata (FileMetadata is frozen)
        updated = metadata.model_copy(
            update={
                "filename": update_data.filename
                if update_data.filename
                else metadata.filename,
                "is_public": update_data.is_public
                if update_data.is_public is not None
                else metadata.is_public,
                "description": update_data.description
                if update_data.description is not None
                else metadata.description,
                "tags": update_data.tags
                if update_data.tags is not None
                else metadata.tags,
                "updated_at": datetime.now(UTC),
            }
        )

        self.files[file_id] = updated
        return updated

    async def delete_metadata(self, file_id: UUID) -> bool:
        """Delete file metadata."""
        if file_id in self.files:
            del self.files[file_id]
            return True
        return False

    async def list_files_by_owner(
        self,
        owner_type: FileOwnerType,
        owner_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FileMetadata]:
        """List files owned by a specific user or group."""
        files = [
            f
            for f in self.files.values()
            if f.owner_type == owner_type and f.owner_id == owner_id
        ]
        return files[offset : offset + limit]

    async def list_public_files(
        self, limit: int = 100, offset: int = 0
    ) -> list[FileMetadata]:
        """List public files."""
        files = [f for f in self.files.values() if f.is_public]
        return files[offset : offset + limit]

    async def search_files(
        self,
        query: str,
        owner_type: Optional[FileOwnerType] = None,
        owner_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FileMetadata]:
        """Search files by filename or description."""
        query_lower = query.lower()
        files = []

        for file in self.files.values():
            # Apply owner filters if specified
            if owner_type and file.owner_type != owner_type:
                continue
            if owner_id and file.owner_id != owner_id:
                continue

            # Search in filename and description
            if query_lower in file.filename.lower() or (
                file.description and query_lower in file.description.lower()
            ):
                files.append(file)

        return files[offset : offset + limit]


class MockFileStorageAdapter(FileStorageAdapter):
    """Mock file storage adapter for testing."""

    def __init__(self):
        self.storage: dict[str, bytes] = {}

    async def store_file(self, file_content: BinaryIO, storage_path: str) -> bool:
        """Store file content at the specified path."""
        self.storage[storage_path] = file_content.read()
        return True

    async def retrieve_file(self, storage_path: str) -> Optional[bytes]:
        """Retrieve file content from storage path."""
        return self.storage.get(storage_path)

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage."""
        if storage_path in self.storage:
            del self.storage[storage_path]
            return True
        return False

    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists at storage path."""
        return storage_path in self.storage

    async def get_file_size(self, storage_path: str) -> Optional[int]:
        """Get file size in bytes."""
        content = self.storage.get(storage_path)
        return len(content) if content else None

    def generate_storage_path(self, file_id: UUID, filename: str) -> str:
        """Generate a storage path for a file."""
        return f"uploads/{file_id}/{filename}"


# --8<-- [start:basic-upload]
@pytest.mark.asyncio
async def test_basic_file_upload():
    """Upload a basic file."""
    repo = MockFileRepository()
    storage = MockFileStorageAdapter()

    # Prepare file
    file_id = uuid4()
    user_id = uuid4()
    file_content = BytesIO(b"Hello, World!")

    # Create upload request
    upload_request = FileUploadRequest(
        filename="hello.txt",
        content_type="text/plain",
        owner_type=FileOwnerType.USER,
        owner_id=user_id,
    )

    # Generate storage path
    storage_path = storage.generate_storage_path(file_id, upload_request.filename)

    # Store file content
    await storage.store_file(file_content, storage_path)

    # Create metadata
    metadata = await repo.create_metadata(
        file_id=file_id,
        file_data=upload_request,
        storage_path=storage_path,
        size_bytes=len(b"Hello, World!"),
    )

    assert metadata is not None
    assert metadata.filename == "hello.txt"
    assert metadata.size_bytes == 13


# --8<-- [end:basic-upload]


# --8<-- [start:detect-content-type]
@pytest.mark.asyncio
async def test_detect_content_type():
    """Automatically detect content type from filename."""
    repo = MockFileRepository()
    user_id = uuid4()

    # Upload without specifying content type
    upload_request = FileUploadRequest(
        filename="document.pdf",
        owner_type=FileOwnerType.USER,
        owner_id=user_id,
    )

    # Content type is auto-detected
    content_type = upload_request.get_content_type()
    assert content_type == "application/pdf"

    # Create metadata
    file_id = uuid4()
    metadata = await repo.create_metadata(
        file_id=file_id,
        file_data=upload_request,
        storage_path=f"uploads/{file_id}/document.pdf",
        size_bytes=1024,
    )

    assert metadata.content_type == "application/pdf"


# --8<-- [end:detect-content-type]


# --8<-- [start:retrieve-file]
@pytest.mark.asyncio
async def test_retrieve_file():
    """Retrieve a file by ID."""
    repo = MockFileRepository()
    storage = MockFileStorageAdapter()

    # Upload file
    file_id = uuid4()
    user_id = uuid4()
    content = b"File contents here"
    storage_path = f"uploads/{file_id}/test.txt"

    await storage.store_file(BytesIO(content), storage_path)
    await repo.create_metadata(
        file_id=file_id,
        file_data=FileUploadRequest(
            filename="test.txt",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
        ),
        storage_path=storage_path,
        size_bytes=len(content),
    )

    # Retrieve metadata
    metadata = await repo.get_metadata_by_id(file_id)
    assert metadata is not None

    # Retrieve file content
    retrieved_content = await storage.retrieve_file(metadata.storage_path)
    assert retrieved_content == content


# --8<-- [end:retrieve-file]


# --8<-- [start:update-metadata]
@pytest.mark.asyncio
async def test_update_file_metadata():
    """Update file metadata."""
    repo = MockFileRepository()
    file_id = uuid4()
    user_id = uuid4()

    # Create file
    await repo.create_metadata(
        file_id=file_id,
        file_data=FileUploadRequest(
            filename="old_name.txt",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            description="Old description",
        ),
        storage_path=f"uploads/{file_id}/old_name.txt",
        size_bytes=100,
    )

    # Update metadata
    updated = await repo.update_metadata(
        file_id,
        FileUpdateRequest(
            filename="new_name.txt",
            description="Updated description",
            tags=["important", "docs"],
        ),
    )

    assert updated is not None
    assert updated.filename == "new_name.txt"
    assert updated.description == "Updated description"
    assert "important" in updated.tags


# --8<-- [end:update-metadata]


# --8<-- [start:delete-file]
@pytest.mark.asyncio
async def test_delete_file():
    """Delete a file completely."""
    repo = MockFileRepository()
    storage = MockFileStorageAdapter()

    # Upload file
    file_id = uuid4()
    user_id = uuid4()
    storage_path = f"uploads/{file_id}/delete_me.txt"

    await storage.store_file(BytesIO(b"Delete this"), storage_path)
    await repo.create_metadata(
        file_id=file_id,
        file_data=FileUploadRequest(
            filename="delete_me.txt",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
        ),
        storage_path=storage_path,
        size_bytes=11,
    )

    # Delete file content
    deleted_storage = await storage.delete_file(storage_path)
    assert deleted_storage is True

    # Delete metadata
    deleted_metadata = await repo.delete_metadata(file_id)
    assert deleted_metadata is True

    # Verify deletion
    assert await storage.file_exists(storage_path) is False
    assert await repo.get_metadata_by_id(file_id) is None


# --8<-- [end:delete-file]


# --8<-- [start:list-by-owner]
@pytest.mark.asyncio
async def test_list_files_by_owner():
    """List all files owned by a user."""
    repo = MockFileRepository()
    user_id = uuid4()

    # Upload multiple files
    for i in range(5):
        await repo.create_metadata(
            file_id=uuid4(),
            file_data=FileUploadRequest(
                filename=f"file_{i}.txt",
                owner_type=FileOwnerType.USER,
                owner_id=user_id,
            ),
            storage_path=f"uploads/file_{i}.txt",
            size_bytes=100 * i,
        )

    # List user's files
    files = await repo.list_files_by_owner(
        owner_type=FileOwnerType.USER, owner_id=user_id
    )

    assert len(files) == 5


# --8<-- [end:list-by-owner]


# --8<-- [start:public-files]
@pytest.mark.asyncio
async def test_public_files():
    """Create and list public files."""
    repo = MockFileRepository()
    user_id = uuid4()

    # Upload public file
    public_file = await repo.create_metadata(
        file_id=uuid4(),
        file_data=FileUploadRequest(
            filename="public_doc.pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            is_public=True,
        ),
        storage_path="uploads/public_doc.pdf",
        size_bytes=1024,
    )

    # Upload private file
    await repo.create_metadata(
        file_id=uuid4(),
        file_data=FileUploadRequest(
            filename="private_doc.pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            is_public=False,
        ),
        storage_path="uploads/private_doc.pdf",
        size_bytes=2048,
    )

    # List public files
    public_files = await repo.list_public_files()
    assert len(public_files) == 1
    assert public_files[0].filename == "public_doc.pdf"


# --8<-- [end:public-files]


# --8<-- [start:search-files]
@pytest.mark.asyncio
async def test_search_files():
    """Search files by filename or description."""
    repo = MockFileRepository()
    user_id = uuid4()

    # Upload files with different names and descriptions
    await repo.create_metadata(
        file_id=uuid4(),
        file_data=FileUploadRequest(
            filename="report_2024.pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            description="Annual financial report",
        ),
        storage_path="uploads/report_2024.pdf",
        size_bytes=1024,
    )

    await repo.create_metadata(
        file_id=uuid4(),
        file_data=FileUploadRequest(
            filename="invoice.pdf",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            description="Report for Q1",
        ),
        storage_path="uploads/invoice.pdf",
        size_bytes=512,
    )

    # Search for "report"
    results = await repo.search_files(query="report")
    assert len(results) == 2

    # Search for "financial"
    results = await repo.search_files(query="financial")
    assert len(results) == 1
    assert results[0].filename == "report_2024.pdf"


# --8<-- [end:search-files]


# --8<-- [start:group-files]
@pytest.mark.asyncio
async def test_group_file_storage():
    """Upload files owned by a group."""
    repo = MockFileRepository()
    group_id = uuid4()

    # Upload file owned by group
    group_file = await repo.create_metadata(
        file_id=uuid4(),
        file_data=FileUploadRequest(
            filename="team_document.docx",
            owner_type=FileOwnerType.GROUP,
            owner_id=group_id,
            description="Shared team document",
        ),
        storage_path="uploads/team_document.docx",
        size_bytes=2048,
    )

    assert group_file.owner_type == FileOwnerType.GROUP
    assert group_file.owner_id == group_id

    # List group files
    group_files = await repo.list_files_by_owner(
        owner_type=FileOwnerType.GROUP, owner_id=group_id
    )

    assert len(group_files) == 1


# --8<-- [end:group-files]


# --8<-- [start:file-tags]
@pytest.mark.asyncio
async def test_file_tags():
    """Organize files with tags."""
    repo = MockFileRepository()
    user_id = uuid4()

    # Upload file with tags
    file_id = uuid4()
    await repo.create_metadata(
        file_id=file_id,
        file_data=FileUploadRequest(
            filename="presentation.pptx",
            owner_type=FileOwnerType.USER,
            owner_id=user_id,
            tags=["work", "important", "q4-2024"],
        ),
        storage_path="uploads/presentation.pptx",
        size_bytes=5120,
    )

    # Retrieve and check tags
    metadata = await repo.get_metadata_by_id(file_id)
    assert metadata is not None
    assert "work" in metadata.tags
    assert "important" in metadata.tags
    assert len(metadata.tags) == 3

    # Update tags
    updated = await repo.update_metadata(
        file_id, FileUpdateRequest(tags=["archived", "2024"])
    )

    assert len(updated.tags) == 2
    assert "archived" in updated.tags


# --8<-- [end:file-tags]
