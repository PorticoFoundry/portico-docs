"""Test examples for Vector Store documentation.

This module tests code examples from vector-store-related documentation to ensure
they remain accurate and working.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pytest

# --8<-- [start:imports]
from portico.ports.vector_store import (
    Document,
    DocumentChunk,
    SearchQuery,
    SearchResult,
    SimilarityRequest,
    SimilarityResponse,
    VectorStore,
    VectorStoreConfig,
    VectorStoreStats,
)

# --8<-- [end:imports]


# --8<-- [start:document-basic]
def test_document_basic():
    """Basic document creation."""
    doc = Document(
        id="doc_123",
        content="Machine learning is a subset of artificial intelligence.",
        metadata={"source": "wikipedia", "category": "AI"},
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.now(UTC),
    )

    assert doc.id == "doc_123"
    assert doc.content == "Machine learning is a subset of artificial intelligence."
    assert doc.metadata["category"] == "AI"
    assert len(doc.embedding) == 3


# --8<-- [end:document-basic]


# --8<-- [start:document-chunk-basic]
def test_document_chunk_basic():
    """Basic document chunk creation."""
    chunk = DocumentChunk(
        id="chunk_456",
        content="Machine learning is a subset of AI",
        metadata={"page": 1},
        embedding=[0.1, 0.2, 0.3],
        document_id="doc_123",
        chunk_index=0,
        start_char=0,
        end_char=38,
        created_at=datetime.now(UTC),
    )

    assert chunk.id == "chunk_456"
    assert chunk.document_id == "doc_123"
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 38


# --8<-- [end:document-chunk-basic]


# --8<-- [start:search-query-vector]
def test_search_query_vector():
    """Search query with pre-computed vector."""
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=5,
        threshold=0.7,
        filters={"category": "AI"},
    )

    assert query.vector == [0.1, 0.2, 0.3]
    assert query.k == 5
    assert query.threshold == 0.7
    assert query.filters["category"] == "AI"


# --8<-- [end:search-query-vector]


# --8<-- [start:search-query-text]
def test_search_query_text():
    """Search query with text (to be embedded)."""
    query = SearchQuery(
        text="What is machine learning?",
        k=5,
        namespace="wikipedia",
    )

    assert query.text == "What is machine learning?"
    assert query.vector is None
    assert query.k == 5
    assert query.namespace == "wikipedia"


# --8<-- [end:search-query-text]


# --8<-- [start:search-query-validation]
def test_search_query_validation():
    """Search query validation - requires vector or text."""
    # Must provide either vector or text
    with pytest.raises(ValueError, match="Either 'vector' or 'text' must be provided"):
        SearchQuery(k=5)

    # Cannot provide both
    with pytest.raises(ValueError, match="Cannot provide both 'vector' and 'text'"):
        SearchQuery(vector=[0.1, 0.2], text="query text", k=5)


# --8<-- [end:search-query-validation]


# --8<-- [start:search-result-basic]
def test_search_result_basic():
    """Search result with document and score."""
    doc = Document(
        id="doc_1",
        content="Machine learning is AI",
        embedding=[0.1, 0.2, 0.3],
    )

    result = SearchResult(
        document=doc,
        score=0.92,
    )

    assert result.score == 0.92
    assert result.document.id == "doc_1"
    assert result.document.content == "Machine learning is AI"


# --8<-- [end:search-result-basic]


# --8<-- [start:search-result-chunk]
def test_search_result_chunk():
    """Search result with document chunk."""
    chunk = DocumentChunk(
        id="chunk_1",
        content="ML is a subset of AI",
        embedding=[0.1, 0.2, 0.3],
        document_id="doc_123",
        chunk_index=0,
        start_char=0,
        end_char=20,
    )

    result = SearchResult(
        document=chunk,
        score=0.85,
    )

    assert result.score == 0.85
    assert isinstance(result.document, DocumentChunk)
    assert result.document.document_id == "doc_123"


# --8<-- [end:search-result-chunk]


# --8<-- [start:similarity-request-vectors]
def test_similarity_request_vectors():
    """Similarity request with vectors."""
    request = SimilarityRequest(
        query_vector=[0.1, 0.2, 0.3],
        target_vectors=[
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
        ],
        method="cosine",
    )

    assert request.query_vector == [0.1, 0.2, 0.3]
    assert len(request.target_vectors) == 2
    assert request.method == "cosine"


# --8<-- [end:similarity-request-vectors]


# --8<-- [start:similarity-request-text]
def test_similarity_request_text():
    """Similarity request with text."""
    request = SimilarityRequest(
        query_text="What is AI?",
        target_texts=[
            "Machine learning is AI",
            "Deep learning is neural networks",
        ],
        method="cosine",
    )

    assert request.query_text == "What is AI?"
    assert len(request.target_texts) == 2
    assert request.method == "cosine"


# --8<-- [end:similarity-request-text]


# --8<-- [start:similarity-request-validation]
def test_similarity_request_validation():
    """Similarity request validation."""
    # Must provide query
    with pytest.raises(
        ValueError, match="Either 'query_vector' or 'query_text' must be provided"
    ):
        SimilarityRequest(target_vectors=[[0.1, 0.2]])

    # Must provide targets
    with pytest.raises(
        ValueError, match="Either 'target_vectors' or 'target_texts' must be provided"
    ):
        SimilarityRequest(query_vector=[0.1, 0.2])


# --8<-- [end:similarity-request-validation]


# --8<-- [start:similarity-response]
def test_similarity_response():
    """Similarity response with scores."""
    response = SimilarityResponse(
        scores=[0.92, 0.85, 0.78],
        method="cosine",
    )

    assert len(response.scores) == 3
    assert response.scores[0] == 0.92
    assert response.method == "cosine"


# --8<-- [end:similarity-response]


# --8<-- [start:vector-store-stats]
def test_vector_store_stats():
    """Vector store statistics."""
    stats = VectorStoreStats(
        total_documents=100,
        total_chunks=500,
        namespaces=["default", "wikipedia", "arxiv"],
        average_embedding_dimension=1536,
        storage_size_bytes=1024000,
    )

    assert stats.total_documents == 100
    assert stats.total_chunks == 500
    assert len(stats.namespaces) == 3
    assert stats.average_embedding_dimension == 1536


# --8<-- [end:vector-store-stats]


# --8<-- [start:vector-store-config]
def test_vector_store_config():
    """Vector store configuration."""
    config = VectorStoreConfig(
        default_namespace="production",
        default_search_k=20,
        default_similarity_threshold=0.75,
        similarity_method="cosine",
        batch_size=200,
        enable_metadata_indexing=True,
    )

    assert config.default_namespace == "production"
    assert config.default_search_k == 20
    assert config.default_similarity_threshold == 0.75
    assert config.similarity_method == "cosine"


# --8<-- [end:vector-store-config]


# --8<-- [start:vector-store-config-defaults]
def test_vector_store_config_defaults():
    """Vector store configuration default values."""
    config = VectorStoreConfig()

    assert config.default_namespace == "default"
    assert config.default_search_k == 10
    assert config.default_similarity_threshold == 0.0
    assert config.similarity_method == "cosine"
    assert config.batch_size == 100
    assert config.enable_metadata_indexing is True


# --8<-- [end:vector-store-config-defaults]


# --8<-- [start:vector-store-interface]
@pytest.mark.asyncio
async def test_vector_store_interface():
    """Vector store interface implementation."""

    class MockVectorStore(VectorStore):
        """Mock vector store for testing."""

        def __init__(self):
            self.documents: Dict[str, Document] = {}
            self.chunks: Dict[str, DocumentChunk] = {}

        async def index_document(
            self, document: Document, namespace: Optional[str] = None
        ) -> str:
            """Index a document."""
            self.documents[document.id] = document
            return document.id

        async def index_documents(
            self, documents: List[Document], namespace: Optional[str] = None
        ) -> List[str]:
            """Index multiple documents."""
            ids = []
            for doc in documents:
                doc_id = await self.index_document(doc, namespace)
                ids.append(doc_id)
            return ids

        async def index_chunk(
            self, chunk: DocumentChunk, namespace: Optional[str] = None
        ) -> str:
            """Index a chunk."""
            self.chunks[chunk.id] = chunk
            return chunk.id

        async def index_chunks(
            self, chunks: List[DocumentChunk], namespace: Optional[str] = None
        ) -> List[str]:
            """Index multiple chunks."""
            ids = []
            for chunk in chunks:
                chunk_id = await self.index_chunk(chunk, namespace)
                ids.append(chunk_id)
            return ids

        async def search(self, query: SearchQuery) -> List[SearchResult]:
            """Search for similar documents."""
            # Mock implementation returns first document
            if self.documents:
                doc = list(self.documents.values())[0]
                return [SearchResult(document=doc, score=0.9)]
            return []

        async def get_document(
            self, document_id: str, namespace: Optional[str] = None
        ) -> Optional[Document]:
            """Get document by ID."""
            return self.documents.get(document_id)

        async def get_chunk(
            self, chunk_id: str, namespace: Optional[str] = None
        ) -> Optional[DocumentChunk]:
            """Get chunk by ID."""
            return self.chunks.get(chunk_id)

        async def delete_document(
            self, document_id: str, namespace: Optional[str] = None
        ) -> bool:
            """Delete document."""
            if document_id in self.documents:
                del self.documents[document_id]
                return True
            return False

        async def delete_chunk(
            self, chunk_id: str, namespace: Optional[str] = None
        ) -> bool:
            """Delete chunk."""
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                return True
            return False

        async def delete_by_metadata(
            self, filters: Dict[str, Any], namespace: Optional[str] = None
        ) -> int:
            """Delete by metadata."""
            return 0

        async def clear_namespace(self, namespace: str) -> int:
            """Clear namespace."""
            return 0

        async def clear_all(self) -> int:
            """Clear all."""
            count = len(self.documents) + len(self.chunks)
            self.documents.clear()
            self.chunks.clear()
            return count

        async def compute_similarity(
            self, request: SimilarityRequest
        ) -> SimilarityResponse:
            """Compute similarity."""
            return SimilarityResponse(scores=[0.9, 0.8], method=request.method)

        async def get_stats(self, namespace: Optional[str] = None) -> VectorStoreStats:
            """Get statistics."""
            return VectorStoreStats(
                total_documents=len(self.documents),
                total_chunks=len(self.chunks),
                namespaces=["default"],
            )

        async def list_namespaces(self) -> List[str]:
            """List namespaces."""
            return ["default", "test"]

    # Test the interface
    store = MockVectorStore()

    # Test index_document
    doc = Document(id="doc1", content="Test", embedding=[0.1, 0.2])
    doc_id = await store.index_document(doc)
    assert doc_id == "doc1"

    # Test get_document
    retrieved = await store.get_document("doc1")
    assert retrieved is not None
    assert retrieved.content == "Test"

    # Test search
    query = SearchQuery(vector=[0.1, 0.2], k=5)
    results = await store.search(query)
    assert len(results) == 1
    assert results[0].score == 0.9

    # Test stats
    stats = await store.get_stats()
    assert stats.total_documents == 1
    assert stats.total_chunks == 0

    # Test delete
    deleted = await store.delete_document("doc1")
    assert deleted is True

    # Test list_namespaces
    namespaces = await store.list_namespaces()
    assert "default" in namespaces


# --8<-- [end:vector-store-interface]


# --8<-- [start:document-metadata-rich]
def test_document_metadata_rich():
    """Document with rich metadata for filtering."""
    doc = Document(
        id="doc_1",
        content="Content about machine learning",
        embedding=[0.1, 0.2, 0.3],
        metadata={
            "category": "AI",
            "language": "en",
            "author": "Alice",
            "year": 2024,
            "tags": ["machine-learning", "deep-learning"],
        },
    )

    assert doc.metadata["category"] == "AI"
    assert doc.metadata["year"] == 2024
    assert "machine-learning" in doc.metadata["tags"]


# --8<-- [end:document-metadata-rich]


# --8<-- [start:search-query-metadata-filtering]
def test_search_query_metadata_filtering():
    """Search query with metadata filters."""
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=10,
        filters={
            "category": "AI",
            "language": "en",
            "year": 2024,
        },
    )

    assert query.filters["category"] == "AI"
    assert query.filters["language"] == "en"
    assert query.filters["year"] == 2024


# --8<-- [end:search-query-metadata-filtering]


# --8<-- [start:search-query-threshold]
def test_search_query_threshold():
    """Search query with similarity threshold."""
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=10,
        threshold=0.7,
    )

    assert query.threshold == 0.7
    assert query.k == 10


# --8<-- [end:search-query-threshold]


# --8<-- [start:search-query-namespace]
def test_search_query_namespace():
    """Search query with namespace."""
    query = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=5,
        namespace="wikipedia",
    )

    assert query.namespace == "wikipedia"


# --8<-- [end:search-query-namespace]


# --8<-- [start:document-auto-fields]
def test_document_auto_fields():
    """Document with auto-generated fields."""
    doc = Document(
        content="Test content",
        embedding=[0.1, 0.2, 0.3],
    )

    # Auto-generated ID
    assert doc.id is not None
    assert len(doc.id) > 0

    # Auto-generated timestamp
    assert doc.created_at is not None
    assert isinstance(doc.created_at, datetime)

    # Default empty metadata
    assert doc.metadata == {}


# --8<-- [end:document-auto-fields]


# --8<-- [start:chunk-position-metadata]
def test_chunk_position_metadata():
    """Document chunk with position metadata."""
    chunk = DocumentChunk(
        id="chunk_1",
        content="First paragraph of the document.",
        embedding=[0.1, 0.2, 0.3],
        document_id="doc_123",
        chunk_index=0,
        start_char=0,
        end_char=33,
        metadata={"section": "introduction"},
    )

    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 33
    assert chunk.metadata["section"] == "introduction"


# --8<-- [end:chunk-position-metadata]


# --8<-- [start:similarity-methods]
def test_similarity_methods():
    """Different similarity computation methods."""
    # Cosine similarity
    cosine_req = SimilarityRequest(
        query_vector=[0.1, 0.2, 0.3],
        target_vectors=[[0.2, 0.3, 0.4]],
        method="cosine",
    )
    assert cosine_req.method == "cosine"

    # Euclidean distance
    euclidean_req = SimilarityRequest(
        query_vector=[0.1, 0.2, 0.3],
        target_vectors=[[0.2, 0.3, 0.4]],
        method="euclidean",
    )
    assert euclidean_req.method == "euclidean"

    # Dot product
    dot_req = SimilarityRequest(
        query_vector=[0.1, 0.2, 0.3],
        target_vectors=[[0.2, 0.3, 0.4]],
        method="dot_product",
    )
    assert dot_req.method == "dot_product"


# --8<-- [end:similarity-methods]


# --8<-- [start:search-query-metadata-control]
def test_search_query_metadata_control():
    """Search query metadata inclusion control."""
    # Include metadata, exclude embeddings
    query1 = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=5,
        include_metadata=True,
        include_embeddings=False,
    )
    assert query1.include_metadata is True
    assert query1.include_embeddings is False

    # Include both
    query2 = SearchQuery(
        vector=[0.1, 0.2, 0.3],
        k=5,
        include_metadata=True,
        include_embeddings=True,
    )
    assert query2.include_metadata is True
    assert query2.include_embeddings is True


# --8<-- [end:search-query-metadata-control]


# --8<-- [start:vector-store-stats-minimal]
def test_vector_store_stats_minimal():
    """Vector store statistics with minimal fields."""
    stats = VectorStoreStats(
        total_documents=50,
        total_chunks=200,
        namespaces=["default"],
    )

    assert stats.total_documents == 50
    assert stats.total_chunks == 200
    assert stats.average_embedding_dimension is None
    assert stats.storage_size_bytes is None


# --8<-- [end:vector-store-stats-minimal]


# --8<-- [start:document-immutability]
def test_document_immutability():
    """Documents are immutable (frozen)."""
    doc = Document(
        id="doc_1",
        content="Original content",
        embedding=[0.1, 0.2, 0.3],
    )

    # Cannot modify fields
    with pytest.raises(Exception):  # Pydantic ValidationError
        doc.content = "Modified content"


# --8<-- [end:document-immutability]


# --8<-- [start:search-result-high-score]
def test_search_result_high_score():
    """Search result with high similarity score."""
    doc = Document(
        id="doc_1",
        content="Exact match content",
        embedding=[0.1, 0.2, 0.3],
    )

    result = SearchResult(
        document=doc,
        score=0.98,  # Very high similarity
    )

    assert result.score > 0.95
    assert result.score <= 1.0


# --8<-- [end:search-result-high-score]


# --8<-- [start:config-performance-tuning]
def test_config_performance_tuning():
    """Vector store config for performance tuning."""
    config = VectorStoreConfig(
        batch_size=500,  # Larger batches
        enable_metadata_indexing=False,  # Disable if not filtering
        default_search_k=50,  # More results
    )

    assert config.batch_size == 500
    assert config.enable_metadata_indexing is False
    assert config.default_search_k == 50


# --8<-- [end:config-performance-tuning]
