"""Test examples for Document Processor documentation.

This module tests code examples from document-processor-related documentation
to ensure they remain accurate and working.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pytest

# --8<-- [start:imports]
from portico.ports.document_processor import (
    ChunkingConfig,
    ChunkingStrategy,
    ContentType,
    DocumentContent,
    DocumentProcessor,
    DocumentProcessorConfig,
    ProcessedChunk,
    ProcessedDocument,
)

# --8<-- [end:imports]


# --8<-- [start:content-type-enum]
def test_content_type_enum():
    """Content type enumeration."""
    assert ContentType.TEXT == "text/plain"
    assert ContentType.MARKDOWN == "text/markdown"
    assert ContentType.HTML == "text/html"
    assert ContentType.PDF == "application/pdf"
    assert ContentType.JSON == "application/json"
    assert ContentType.CSV == "text/csv"


# --8<-- [end:content-type-enum]


# --8<-- [start:document-content-basic]
def test_document_content_basic():
    """Basic document content creation."""
    doc = DocumentContent(
        id="doc_123",
        content="Machine learning is a subset of AI...",
        content_type=ContentType.TEXT,
        title="Introduction to Machine Learning",
        source_url="https://example.com/ml-intro",
        metadata={"author": "Alice", "category": "AI"},
        created_at=datetime.now(UTC),
    )

    assert doc.id == "doc_123"
    assert doc.content_type == ContentType.TEXT
    assert doc.title == "Introduction to Machine Learning"
    assert doc.metadata["author"] == "Alice"


# --8<-- [end:document-content-basic]


# --8<-- [start:document-content-auto-fields]
def test_document_content_auto_fields():
    """Document content with auto-generated fields."""
    doc = DocumentContent(
        content="Sample document content",
    )

    # Auto-generated ID
    assert doc.id is not None
    assert len(doc.id) > 0

    # Auto-generated timestamp
    assert doc.created_at is not None
    assert isinstance(doc.created_at, datetime)

    # Default values
    assert doc.content_type == ContentType.TEXT
    assert doc.title is None
    assert doc.metadata == {}


# --8<-- [end:document-content-auto-fields]


# --8<-- [start:chunking-config-basic]
def test_chunking_config_basic():
    """Basic chunking configuration."""
    config = ChunkingConfig(
        chunk_size=800,
        chunk_overlap=100,
        respect_sentence_boundaries=True,
        min_chunk_size=200,
        max_chunk_size=1200,
    )

    assert config.chunk_size == 800
    assert config.chunk_overlap == 100
    assert config.min_chunk_size == 200
    assert config.max_chunk_size == 1200


# --8<-- [end:chunking-config-basic]


# --8<-- [start:chunking-config-defaults]
def test_chunking_config_defaults():
    """Chunking configuration default values."""
    config = ChunkingConfig()

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.respect_sentence_boundaries is True
    assert config.respect_paragraph_boundaries is True
    assert config.min_chunk_size == 100
    assert config.max_chunk_size == 2000
    assert config.preserve_code_blocks is True
    assert config.preserve_headers is True


# --8<-- [end:chunking-config-defaults]


# --8<-- [start:processed-chunk-basic]
def test_processed_chunk_basic():
    """Basic processed chunk creation."""
    chunk = ProcessedChunk(
        id="chunk_1",
        content="Machine learning is a subset of artificial intelligence.",
        metadata={"section": "Introduction"},
        document_id="doc_123",
        chunk_index=0,
        start_char=0,
        end_char=58,
        token_count=12,
        language="en",
        content_type=ContentType.TEXT,
        created_at=datetime.now(UTC),
    )

    assert chunk.id == "chunk_1"
    assert chunk.document_id == "doc_123"
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 58
    assert chunk.token_count == 12
    assert chunk.language == "en"


# --8<-- [end:processed-chunk-basic]


# --8<-- [start:processed-chunk-auto-fields]
def test_processed_chunk_auto_fields():
    """Processed chunk with auto-generated fields."""
    chunk = ProcessedChunk(
        content="Sample chunk content",
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=20,
    )

    # Auto-generated ID
    assert chunk.id is not None
    assert len(chunk.id) > 0

    # Auto-generated timestamp
    assert chunk.created_at is not None

    # Default values
    assert chunk.metadata == {}
    assert chunk.token_count is None
    assert chunk.language is None
    assert chunk.content_type == ContentType.TEXT


# --8<-- [end:processed-chunk-auto-fields]


# --8<-- [start:processed-document-basic]
def test_processed_document_basic():
    """Basic processed document creation."""
    doc = DocumentContent(
        id="doc_1",
        content="Sample content",
    )

    chunk1 = ProcessedChunk(
        id="chunk_1",
        content="Sample",
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=6,
    )

    chunk2 = ProcessedChunk(
        id="chunk_2",
        content="content",
        document_id="doc_1",
        chunk_index=1,
        start_char=7,
        end_char=14,
    )

    config = ChunkingConfig()

    processed = ProcessedDocument(
        id="proc_123",
        original_document=doc,
        chunks=[chunk1, chunk2],
        chunking_strategy="semantic",
        chunking_config=config,
        total_chunks=2,
        total_characters=14,
        total_tokens=2,
        average_chunk_size=7.0,
        chunk_size_variance=0.5,
        processed_at=datetime.now(UTC),
    )

    assert processed.id == "proc_123"
    assert processed.total_chunks == 2
    assert len(processed.chunks) == 2
    assert processed.chunking_strategy == "semantic"


# --8<-- [end:processed-document-basic]


# --8<-- [start:processed-document-validation]
def test_processed_document_validation():
    """Processed document validation."""
    doc = DocumentContent(id="doc_1", content="Test")

    chunk1 = ProcessedChunk(
        id="chunk_1",
        content="Test",
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=4,
    )

    config = ChunkingConfig()

    # Chunk count mismatch
    with pytest.raises(ValueError, match="Chunk count mismatch"):
        ProcessedDocument(
            original_document=doc,
            chunks=[chunk1],
            chunking_strategy="test",
            chunking_config=config,
            total_chunks=2,  # Mismatch!
            total_characters=4,
            average_chunk_size=4.0,
            chunk_size_variance=0.0,
        )

    # Non-sequential indices
    chunk2_bad = ProcessedChunk(
        id="chunk_2",
        content="Test",
        document_id="doc_1",
        chunk_index=2,  # Should be 1!
        start_char=0,
        end_char=4,
    )

    with pytest.raises(ValueError, match="Non-sequential chunk indices"):
        ProcessedDocument(
            original_document=doc,
            chunks=[chunk1, chunk2_bad],
            chunking_strategy="test",
            chunking_config=config,
            total_chunks=2,
            total_characters=8,
            average_chunk_size=4.0,
            chunk_size_variance=0.0,
        )


# --8<-- [end:processed-document-validation]


# --8<-- [start:chunking-strategy-interface]
def test_chunking_strategy_interface():
    """Chunking strategy interface implementation."""

    class SimpleChunker(ChunkingStrategy):
        """Simple chunking strategy for testing."""

        @property
        def strategy_name(self) -> str:
            return "simple"

        def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
            """Split text into fixed-size chunks."""
            chunks = []
            chunk_size = config.chunk_size

            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                chunks.append(chunk)

            return chunks

        def get_chunk_boundaries(
            self, text: str, config: ChunkingConfig
        ) -> List[tuple[int, int]]:
            """Get chunk boundaries."""
            boundaries = []
            chunk_size = config.chunk_size

            for i in range(0, len(text), chunk_size):
                start = i
                end = min(i + chunk_size, len(text))
                boundaries.append((start, end))

            return boundaries

    # Test the strategy
    strategy = SimpleChunker()
    config = ChunkingConfig(chunk_size=10)

    text = "This is a test document with some content"
    chunks = strategy.chunk_text(text, config)

    assert strategy.strategy_name == "simple"
    assert len(chunks) == 5  # 42 chars / 10 = 5 chunks
    assert chunks[0] == "This is a "

    boundaries = strategy.get_chunk_boundaries(text, config)
    assert len(boundaries) == 5
    assert boundaries[0] == (0, 10)
    assert boundaries[4] == (40, 41)  # 41 chars total (0-40 inclusive)


# --8<-- [end:chunking-strategy-interface]


# --8<-- [start:document-processor-interface]
@pytest.mark.asyncio
async def test_document_processor_interface():
    """Document processor interface implementation."""

    class MockDocumentProcessor(DocumentProcessor):
        """Mock document processor for testing."""

        async def process_document(
            self,
            document: DocumentContent,
            chunking_config: Optional[ChunkingConfig] = None,
        ) -> ProcessedDocument:
            """Process a document."""
            config = chunking_config or ChunkingConfig()

            # Simple chunking
            chunk_size = config.chunk_size
            chunks = []

            for i in range(0, len(document.content), chunk_size):
                chunk_content = document.content[i : i + chunk_size]
                chunks.append(
                    ProcessedChunk(
                        id=f"chunk_{i}",
                        content=chunk_content,
                        document_id=document.id,
                        chunk_index=i // chunk_size,
                        start_char=i,
                        end_char=min(i + chunk_size, len(document.content)),
                    )
                )

            return ProcessedDocument(
                original_document=document,
                chunks=chunks,
                chunking_strategy="simple",
                chunking_config=config,
                total_chunks=len(chunks),
                total_characters=len(document.content),
                average_chunk_size=len(document.content) / len(chunks),
                chunk_size_variance=0.0,
            )

        async def process_text(
            self,
            text: str,
            content_type: ContentType = ContentType.TEXT,
            chunking_config: Optional[ChunkingConfig] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> ProcessedDocument:
            """Process text."""
            doc = DocumentContent(
                content=text,
                content_type=content_type,
                metadata=metadata or {},
            )
            return await self.process_document(doc, chunking_config)

        async def chunk_document(
            self,
            document: DocumentContent,
            strategy: ChunkingStrategy,
            config: Optional[ChunkingConfig] = None,
        ) -> List[ProcessedChunk]:
            """Chunk a document."""
            cfg = config or ChunkingConfig()
            boundaries = strategy.get_chunk_boundaries(document.content, cfg)

            chunks = []
            for i, (start, end) in enumerate(boundaries):
                chunks.append(
                    ProcessedChunk(
                        id=f"chunk_{i}",
                        content=document.content[start:end],
                        document_id=document.id,
                        chunk_index=i,
                        start_char=start,
                        end_char=end,
                    )
                )

            return chunks

        def get_supported_content_types(self) -> List[ContentType]:
            """Get supported types."""
            return [ContentType.TEXT, ContentType.MARKDOWN]

        def estimate_token_count(self, text: str, model: Optional[str] = None) -> int:
            """Estimate tokens."""
            # Simple estimation: ~4 chars per token
            return len(text) // 4

    # Test the processor
    processor = MockDocumentProcessor()

    # Test process_text
    text = "This is a test document with multiple sentences. It should be chunked."
    processed = await processor.process_text(text, content_type=ContentType.TEXT)

    assert processed.total_chunks > 0
    assert processed.chunking_strategy == "simple"
    assert processed.total_characters == len(text)

    # Test process_document
    doc = DocumentContent(id="doc_1", content="Sample content")
    processed = await processor.process_document(doc)
    assert processed.original_document.id == "doc_1"

    # Test get_supported_content_types
    types = processor.get_supported_content_types()
    assert ContentType.TEXT in types
    assert ContentType.MARKDOWN in types

    # Test estimate_token_count
    tokens = processor.estimate_token_count("This is a test")
    assert tokens > 0


# --8<-- [end:document-processor-interface]


# --8<-- [start:document-processor-config]
def test_document_processor_config():
    """Document processor configuration."""
    config = DocumentProcessorConfig(
        enable_token_counting=True,
        enable_language_detection=True,
        enable_content_analysis=True,
        auto_detect_content_type=True,
        max_document_size=5_000_000,
        max_chunks_per_document=500,
        processing_timeout_seconds=60.0,
        default_tokenizer_model="gpt-4",
        tokens_per_chunk_target=500,
    )

    assert config.enable_token_counting is True
    assert config.max_document_size == 5_000_000
    assert config.default_tokenizer_model == "gpt-4"
    assert config.tokens_per_chunk_target == 500


# --8<-- [end:document-processor-config]


# --8<-- [start:document-processor-config-defaults]
def test_document_processor_config_defaults():
    """Document processor configuration defaults."""
    config = DocumentProcessorConfig()

    assert isinstance(config.default_chunking_config, ChunkingConfig)
    assert config.enable_token_counting is True
    assert config.enable_language_detection is True
    assert config.enable_content_analysis is True
    assert config.auto_detect_content_type is True
    assert config.fallback_content_type == ContentType.TEXT
    assert config.max_document_size == 10_000_000
    assert config.max_chunks_per_document == 1000
    assert config.processing_timeout_seconds == 30.0
    assert config.default_tokenizer_model == "gpt-3.5-turbo"
    assert config.tokens_per_chunk_target == 300


# --8<-- [end:document-processor-config-defaults]


# --8<-- [start:chunking-config-qa]
def test_chunking_config_qa():
    """Chunking config for Q&A use case."""
    config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
    )

    assert config.chunk_size == 500
    assert config.chunk_overlap == 50


# --8<-- [end:chunking-config-qa]


# --8<-- [start:chunking-config-context]
def test_chunking_config_context():
    """Chunking config for broader context."""
    config = ChunkingConfig(
        chunk_size=1500,
        chunk_overlap=200,
    )

    assert config.chunk_size == 1500
    assert config.chunk_overlap == 200


# --8<-- [end:chunking-config-context]


# --8<-- [start:chunking-config-overlap]
def test_chunking_config_overlap():
    """Chunking config with overlap for context."""
    config = ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,  # 20% overlap
    )

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200


# --8<-- [end:chunking-config-overlap]


# --8<-- [start:chunking-config-boundaries]
def test_chunking_config_boundaries():
    """Chunking config respecting content boundaries."""
    config = ChunkingConfig(
        chunk_size=800,
        respect_sentence_boundaries=True,
        respect_paragraph_boundaries=True,
    )

    assert config.chunk_size == 800
    assert config.respect_sentence_boundaries is True
    assert config.respect_paragraph_boundaries is True


# --8<-- [end:chunking-config-boundaries]


# --8<-- [start:chunk-metadata-rich]
def test_chunk_metadata_rich():
    """Chunk with rich metadata."""
    chunk = ProcessedChunk(
        id="chunk_1",
        content="Sample content",
        metadata={
            "section": "Introduction",
            "page": 1,
            "heading": "Getting Started",
            "author": "Alice",
        },
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=14,
    )

    assert chunk.metadata["section"] == "Introduction"
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["heading"] == "Getting Started"
    assert chunk.metadata["author"] == "Alice"


# --8<-- [end:chunk-metadata-rich]


# --8<-- [start:document-content-markdown]
def test_document_content_markdown():
    """Document content with markdown type."""
    doc = DocumentContent(
        id="doc_1",
        content="# Title\n\nThis is markdown content",
        content_type=ContentType.MARKDOWN,
        title="Guide to Machine Learning",
    )

    assert doc.content_type == ContentType.MARKDOWN
    assert doc.title == "Guide to Machine Learning"


# --8<-- [end:document-content-markdown]


# --8<-- [start:chunking-config-content-preservation]
def test_chunking_config_content_preservation():
    """Chunking config with content preservation."""
    config = ChunkingConfig(
        chunk_size=1000,
        preserve_code_blocks=True,
        preserve_headers=True,
    )

    assert config.preserve_code_blocks is True
    assert config.preserve_headers is True


# --8<-- [end:chunking-config-content-preservation]


# --8<-- [start:processed-document-metrics]
def test_processed_document_metrics():
    """Processed document with quality metrics."""
    doc = DocumentContent(id="doc_1", content="Test content")

    chunk = ProcessedChunk(
        id="chunk_1",
        content="Test content",
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=12,
    )

    config = ChunkingConfig()

    processed = ProcessedDocument(
        original_document=doc,
        chunks=[chunk],
        chunking_strategy="simple",
        chunking_config=config,
        total_chunks=1,
        total_characters=12,
        total_tokens=3,
        average_chunk_size=12.0,
        chunk_size_variance=0.0,
    )

    assert processed.average_chunk_size == 12.0
    assert processed.chunk_size_variance == 0.0
    assert processed.total_tokens == 3


# --8<-- [end:processed-document-metrics]


# --8<-- [start:document-content-immutability]
def test_document_content_immutability():
    """Document content is immutable."""
    doc = DocumentContent(
        id="doc_1",
        content="Original content",
    )

    # Cannot modify fields
    with pytest.raises(Exception):  # Pydantic ValidationError
        doc.content = "Modified content"


# --8<-- [end:document-content-immutability]


# --8<-- [start:chunk-position-tracking]
def test_chunk_position_tracking():
    """Chunk with position tracking."""
    chunk = ProcessedChunk(
        id="chunk_1",
        content="First paragraph of the document.",
        document_id="doc_123",
        chunk_index=0,
        start_char=0,
        end_char=33,
        metadata={"section": "introduction"},
    )

    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 33
    # Content is 32 chars, end_char is exclusive (33)
    assert chunk.end_char - chunk.start_char == 33


# --8<-- [end:chunk-position-tracking]


# --8<-- [start:config-performance-limits]
def test_config_performance_limits():
    """Config with performance limits."""
    config = DocumentProcessorConfig(
        max_document_size=20_000_000,  # 20MB
        max_chunks_per_document=2000,
        processing_timeout_seconds=120.0,
    )

    assert config.max_document_size == 20_000_000
    assert config.max_chunks_per_document == 2000
    assert config.processing_timeout_seconds == 120.0


# --8<-- [end:config-performance-limits]


# --8<-- [start:config-token-targeting]
def test_config_token_targeting():
    """Config with token targeting."""
    config = DocumentProcessorConfig(
        default_tokenizer_model="gpt-4",
        tokens_per_chunk_target=500,
    )

    assert config.default_tokenizer_model == "gpt-4"
    assert config.tokens_per_chunk_target == 500


# --8<-- [end:config-token-targeting]


# --8<-- [start:content-type-detection]
def test_content_type_detection():
    """Content type detection config."""
    config = DocumentProcessorConfig(
        auto_detect_content_type=True,
        fallback_content_type=ContentType.MARKDOWN,
    )

    assert config.auto_detect_content_type is True
    assert config.fallback_content_type == ContentType.MARKDOWN


# --8<-- [end:content-type-detection]
