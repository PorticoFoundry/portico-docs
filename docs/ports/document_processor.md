# Document Processor Port

## Overview

The Document Processor Port defines the contract for processing documents into searchable chunks for RAG (Retrieval-Augmented Generation) systems.

**Purpose**: Provides interfaces and domain models for splitting documents into semantic chunks, estimating token counts, and managing document processing pipelines.

**Domain**: Document processing, text chunking, content analysis, RAG preparation

**Key Capabilities**:

- Document chunking with configurable strategies
- Multiple content type support (text, markdown, HTML, PDF, JSON, CSV)
- Flexible chunking configurations (size, overlap, boundaries)
- Token estimation for LLM context management
- Chunk metadata and position tracking
- Quality metrics (average size, variance)
- Configurable processing limits and timeouts

**Port Type**: Processor

**When to Use**:

- Building RAG (Retrieval-Augmented Generation) systems
- Processing documents for vector storage and semantic search
- Splitting long documents into manageable chunks
- Preparing content for LLM consumption
- Implementing document analysis pipelines
- Creating knowledge bases from unstructured content

## Domain Models

### DocumentContent

Raw document content with metadata. Immutable snapshot of the original document.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | `uuid4()` | Unique document identifier |
| `content` | `str` | Yes | - | Raw document text content |
| `content_type` | `ContentType` | Yes | `ContentType.TEXT` | MIME type of content |
| `title` | `Optional[str]` | No | `None` | Document title |
| `source_url` | `Optional[str]` | No | `None` | Original source URL |
| `metadata` | `Dict[str, Any]` | Yes | `{}` | Custom metadata |
| `created_at` | `datetime` | Yes | `now(UTC)` | Document creation timestamp |

**Example**:

```python
from portico.ports.document_processor import DocumentContent, ContentType

# Plain text document
doc = DocumentContent(
    content="This is a sample document for processing.",
    content_type=ContentType.TEXT,
    title="Sample Document",
    metadata={"author": "John Doe", "category": "tutorial"}
)

# Markdown document from web
markdown_doc = DocumentContent(
    content="# Introduction\n\nThis is markdown content...",
    content_type=ContentType.MARKDOWN,
    source_url="https://example.com/docs/intro.md",
    metadata={"version": "1.0"}
)
```

### ChunkingConfig

Configuration for document chunking strategies.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `chunk_size` | `int` | Yes | `1000` | Target chunk size in characters |
| `chunk_overlap` | `int` | Yes | `200` | Overlap between chunks in characters |
| `respect_sentence_boundaries` | `bool` | Yes | `True` | Avoid splitting sentences |
| `respect_paragraph_boundaries` | `bool` | Yes | `True` | Avoid splitting paragraphs |
| `min_chunk_size` | `int` | Yes | `100` | Minimum chunk size to avoid tiny chunks |
| `max_chunk_size` | `int` | Yes | `2000` | Maximum chunk size as hard limit |
| `preserve_code_blocks` | `bool` | Yes | `True` | Keep code blocks intact (markdown/HTML) |
| `preserve_headers` | `bool` | Yes | `True` | Include headers in chunk metadata |

**Example**:

```python
from portico.ports.document_processor import ChunkingConfig

# Standard configuration
config = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    respect_sentence_boundaries=True
)

# Large chunks for detailed context
large_config = ChunkingConfig(
    chunk_size=2000,
    chunk_overlap=300,
    max_chunk_size=3000
)

# Precise sentence-based chunking
sentence_config = ChunkingConfig(
    chunk_size=500,
    chunk_overlap=50,
    respect_sentence_boundaries=True,
    respect_paragraph_boundaries=False,
    min_chunk_size=200
)

# Code-focused configuration
code_config = ChunkingConfig(
    chunk_size=800,
    preserve_code_blocks=True,
    preserve_headers=True,
    respect_sentence_boundaries=False
)
```

### ProcessedChunk

Individual chunk from document processing with position tracking and metadata.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | `uuid4()` | Unique chunk identifier |
| `content` | `str` | Yes | - | Chunk text content |
| `metadata` | `Dict[str, Any]` | Yes | `{}` | Chunk-specific metadata |
| `document_id` | `str` | Yes | - | Parent document ID |
| `chunk_index` | `int` | Yes | - | Sequential position in document (0-indexed) |
| `start_char` | `int` | Yes | - | Starting character position in original |
| `end_char` | `int` | Yes | - | Ending character position in original |
| `token_count` | `Optional[int]` | No | `None` | Estimated token count |
| `language` | `Optional[str]` | No | `None` | Detected language code |
| `content_type` | `ContentType` | Yes | `ContentType.TEXT` | Content type |
| `created_at` | `datetime` | Yes | `now(UTC)` | Chunk creation timestamp |

**Example**:

```python
from portico.ports.document_processor import ProcessedChunk, ContentType

chunk = ProcessedChunk(
    content="This is the first paragraph of the document.",
    document_id="doc-123",
    chunk_index=0,
    start_char=0,
    end_char=45,
    token_count=12,
    language="en",
    content_type=ContentType.TEXT,
    metadata={"section": "introduction"}
)

# Chunks maintain order and position
print(f"Chunk {chunk.chunk_index}: chars {chunk.start_char}-{chunk.end_char}")
print(f"Estimated tokens: {chunk.token_count}")
```

### ProcessedDocument

Document split into searchable chunks with processing metadata and quality metrics.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | `uuid4()` | Unique processed document ID |
| `original_document` | `DocumentContent` | Yes | - | Original document reference |
| `chunks` | `List[ProcessedChunk]` | Yes | - | List of processed chunks |
| `chunking_strategy` | `str` | Yes | - | Strategy name used for chunking |
| `chunking_config` | `ChunkingConfig` | Yes | - | Configuration used |
| `total_chunks` | `int` | Yes | - | Number of chunks created |
| `total_characters` | `int` | Yes | - | Total characters in original |
| `total_tokens` | `Optional[int]` | No | `None` | Total estimated tokens |
| `average_chunk_size` | `float` | Yes | - | Mean chunk size in characters |
| `chunk_size_variance` | `float` | Yes | - | Statistical variance of chunk sizes |
| `processed_at` | `datetime` | Yes | `now(UTC)` | Processing timestamp |

**Example**:

```python
from portico.ports.document_processor import ProcessedDocument

# After processing a document
processed = ProcessedDocument(
    original_document=doc,
    chunks=[chunk1, chunk2, chunk3],
    chunking_strategy="paragraph",
    chunking_config=config,
    total_chunks=3,
    total_characters=1500,
    total_tokens=350,
    average_chunk_size=500.0,
    chunk_size_variance=25.5
)

print(f"Split into {processed.total_chunks} chunks")
print(f"Average chunk: {processed.average_chunk_size:.0f} chars")
print(f"Total tokens: {processed.total_tokens}")

# Access chunks in order
for chunk in processed.chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.content[:50]}...")
```

### DocumentProcessorConfig

Configuration for document processing operations.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default_chunking_config` | `ChunkingConfig` | Yes | `ChunkingConfig()` | Default chunking settings |
| `enable_token_counting` | `bool` | Yes | `True` | Estimate token counts |
| `enable_language_detection` | `bool` | Yes | `True` | Detect chunk language |
| `enable_content_analysis` | `bool` | Yes | `True` | Analyze content structure |
| `auto_detect_content_type` | `bool` | Yes | `True` | Auto-detect MIME type |
| `fallback_content_type` | `ContentType` | Yes | `ContentType.TEXT` | Fallback if detection fails |
| `max_document_size` | `int` | Yes | `10_000_000` | Maximum document size (10MB) |
| `max_chunks_per_document` | `int` | Yes | `1000` | Maximum chunks to generate |
| `processing_timeout_seconds` | `float` | Yes | `30.0` | Processing timeout |
| `default_tokenizer_model` | `str` | Yes | `"gpt-3.5-turbo"` | Model for token estimation |
| `tokens_per_chunk_target` | `int` | Yes | `300` | Target tokens per chunk |

**Example**:

```python
from portico.ports.document_processor import DocumentProcessorConfig, ChunkingConfig

# Production configuration
config = DocumentProcessorConfig(
    default_chunking_config=ChunkingConfig(
        chunk_size=1200,
        chunk_overlap=200
    ),
    enable_token_counting=True,
    max_document_size=20_000_000,  # 20MB
    max_chunks_per_document=2000,
    processing_timeout_seconds=60.0,
    default_tokenizer_model="gpt-4"
)

# Fast processing (minimal analysis)
fast_config = DocumentProcessorConfig(
    enable_token_counting=False,
    enable_language_detection=False,
    enable_content_analysis=False,
    processing_timeout_seconds=10.0
)
```

## Enumerations

### ContentType

Supported content types for document processing.

| Value | Description |
|-------|-------------|
| `TEXT` | Plain text (`text/plain`) |
| `MARKDOWN` | Markdown formatted text (`text/markdown`) |
| `HTML` | HTML content (`text/html`) |
| `PDF` | PDF documents (`application/pdf`) |
| `JSON` | JSON structured data (`application/json`) |
| `CSV` | Comma-separated values (`text/csv`) |

**Example**:

```python
from portico.ports.document_processor import ContentType

# Type-safe content type specification
doc = DocumentContent(
    content=markdown_text,
    content_type=ContentType.MARKDOWN
)

# Different processing for different types
if doc.content_type == ContentType.MARKDOWN:
    # Use markdown-aware chunking
    strategy = MarkdownChunker()
elif doc.content_type == ContentType.PDF:
    # Use PDF-specific processing
    strategy = PDFChunker()
```

## Port Interfaces

### DocumentProcessor

The `DocumentProcessor` abstract base class defines the contract for document processing pipelines.

**Location**: `portico.ports.document_processor.DocumentProcessor`

#### Key Methods

##### process_document

```python
async def process_document(
    document: DocumentContent,
    chunking_config: Optional[ChunkingConfig] = None
) -> ProcessedDocument
```

Process a document into searchable chunks. Primary method for document processing.

**Parameters**:

- `document: DocumentContent` - Raw document content to process
- `chunking_config: Optional[ChunkingConfig]` - Optional chunking configuration (uses default if None)

**Returns**: ProcessedDocument with chunks and processing metadata

**Raises**:

- `ValueError` - If document size exceeds maximum allowed

**Example**:

```python
from portico.ports.document_processor import DocumentProcessor, DocumentContent, ChunkingConfig

# Process with default configuration
doc = DocumentContent(content="Long document text...")
processed = await processor.process_document(doc)

print(f"Created {processed.total_chunks} chunks")
for chunk in processed.chunks:
    print(f"  Chunk {chunk.chunk_index}: {len(chunk.content)} chars")

# Process with custom configuration
config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
processed = await processor.process_document(doc, config)
```

##### process_text

```python
async def process_text(
    text: str,
    content_type: ContentType = ContentType.TEXT,
    chunking_config: Optional[ChunkingConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessedDocument
```

Process raw text into searchable chunks. Convenience method for text-only content.

**Parameters**:

- `text: str` - Raw text content
- `content_type: ContentType` - Type of content for processing hints
- `chunking_config: Optional[ChunkingConfig]` - Optional chunking configuration
- `metadata: Optional[Dict[str, Any]]` - Optional metadata for the document

**Returns**: ProcessedDocument with chunks

**Example**:

```python
# Simple text processing
text = "This is a long document that needs to be chunked..."
processed = await processor.process_text(text)

# Markdown processing with metadata
markdown_text = "# Title\n\nContent..."
processed = await processor.process_text(
    markdown_text,
    content_type=ContentType.MARKDOWN,
    metadata={"source": "docs", "author": "team"}
)

# Custom chunking for specific use case
config = ChunkingConfig(chunk_size=800, preserve_code_blocks=True)
processed = await processor.process_text(
    code_documentation,
    content_type=ContentType.MARKDOWN,
    chunking_config=config
)
```

#### Other Methods

##### chunk_document

```python
async def chunk_document(
    document: DocumentContent,
    strategy: ChunkingStrategy,
    config: Optional[ChunkingConfig] = None
) -> List[ProcessedChunk]
```

Chunk a document using a specific strategy. Returns list of processed chunks.

##### get_supported_content_types

```python
def get_supported_content_types() -> List[ContentType]
```

Get list of supported content types for processing. Returns list of ContentType enum values.

##### estimate_token_count

```python
def estimate_token_count(
    text: str,
    model: Optional[str] = None
) -> int
```

Estimate token count for text using the specified model. Returns estimated token count.

### ChunkingStrategy

The `ChunkingStrategy` abstract base class defines the contract for chunking algorithms.

**Location**: `portico.ports.document_processor.ChunkingStrategy`

#### Key Methods

##### chunk_text

```python
def chunk_text(
    text: str,
    config: ChunkingConfig
) -> List[str]
```

Split text into chunks according to the strategy.

**Parameters**:

- `text: str` - Text content to chunk
- `config: ChunkingConfig` - Chunking configuration

**Returns**: List of text chunks

**Example**:

```python
from portico.ports.document_processor import ChunkingStrategy, ChunkingConfig

# Use a specific strategy
strategy = ParagraphChunker()
config = ChunkingConfig(chunk_size=1000)

chunks = strategy.chunk_text(long_text, config)
print(f"Split into {len(chunks)} chunks")
```

##### get_chunk_boundaries

```python
def get_chunk_boundaries(
    text: str,
    config: ChunkingConfig
) -> List[tuple[int, int]]
```

Get start and end character positions for each chunk.

**Parameters**:

- `text: str` - Text content to analyze
- `config: ChunkingConfig` - Chunking configuration

**Returns**: List of (start_char, end_char) tuples for each chunk

##### strategy_name

```python
@property
def strategy_name() -> str
```

Name of the chunking strategy. Returns strategy identifier.

## Common Patterns

### Basic Document Processing for RAG

```python
from portico.ports.document_processor import (
    DocumentProcessor,
    DocumentContent,
    ChunkingConfig,
    ContentType
)

async def prepare_document_for_rag(
    processor: DocumentProcessor,
    text: str,
    title: str,
    source_url: str
) -> ProcessedDocument:
    """Prepare a document for RAG system."""

    # Create document
    doc = DocumentContent(
        content=text,
        content_type=ContentType.TEXT,
        title=title,
        source_url=source_url,
        metadata={"indexed_at": datetime.now(UTC).isoformat()}
    )

    # Configure chunking for optimal RAG performance
    config = ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        respect_sentence_boundaries=True,
        min_chunk_size=100,
        max_chunk_size=1500
    )

    # Process document
    processed = await processor.process_document(doc, config)

    # Log processing results
    print(f"Processed: {processed.original_document.title}")
    print(f"  Chunks: {processed.total_chunks}")
    print(f"  Avg size: {processed.average_chunk_size:.0f} chars")
    print(f"  Tokens: {processed.total_tokens}")

    return processed

# Usage
processed = await prepare_document_for_rag(
    processor,
    article_text,
    "API Documentation",
    "https://docs.example.com/api"
)

# Store chunks in vector database
for chunk in processed.chunks:
    await vector_store.add(
        text=chunk.content,
        metadata={
            "document_id": processed.original_document.id,
            "chunk_index": chunk.chunk_index,
            "source_url": processed.original_document.source_url
        }
    )
```

### Content-Type Specific Processing

```python
from portico.ports.document_processor import ContentType, ChunkingConfig

async def process_by_content_type(
    processor: DocumentProcessor,
    content: str,
    content_type: ContentType
):
    """Process document with type-specific configuration."""

    # Different configs for different content types
    if content_type == ContentType.MARKDOWN:
        config = ChunkingConfig(
            chunk_size=1200,
            preserve_code_blocks=True,
            preserve_headers=True,
            respect_paragraph_boundaries=True
        )
    elif content_type == ContentType.CODE:
        config = ChunkingConfig(
            chunk_size=800,
            preserve_code_blocks=True,
            respect_sentence_boundaries=False
        )
    elif content_type == ContentType.PDF:
        config = ChunkingConfig(
            chunk_size=1500,
            chunk_overlap=300,
            respect_paragraph_boundaries=True
        )
    else:
        config = ChunkingConfig()  # Default

    # Process with appropriate configuration
    processed = await processor.process_text(
        content,
        content_type=content_type,
        chunking_config=config
    )

    return processed

# Usage
markdown_processed = await process_by_content_type(
    processor,
    markdown_content,
    ContentType.MARKDOWN
)
```

### Token-Aware Chunking for LLMs

```python
from portico.ports.document_processor import DocumentProcessor, ChunkingConfig

async def chunk_for_llm_context(
    processor: DocumentProcessor,
    text: str,
    max_tokens_per_chunk: int = 500
):
    """Chunk document to fit LLM context windows."""

    # Estimate characters per token (rough approximation)
    chars_per_token = 4  # Average for English text
    target_chars = max_tokens_per_chunk * chars_per_token

    config = ChunkingConfig(
        chunk_size=target_chars,
        chunk_overlap=max(100, target_chars // 10),
        respect_sentence_boundaries=True,
        max_chunk_size=target_chars + 500
    )

    processed = await processor.process_text(text, chunking_config=config)

    # Verify chunks fit in token budget
    for chunk in processed.chunks:
        if chunk.token_count and chunk.token_count > max_tokens_per_chunk:
            print(f"Warning: Chunk {chunk.chunk_index} exceeds token limit")

    return processed

# Usage for GPT-3.5 context
processed = await chunk_for_llm_context(processor, document, max_tokens_per_chunk=500)

# Use chunks in LLM prompts
for chunk in processed.chunks:
    prompt = f"Analyze this text:\n\n{chunk.content}\n\nProvide summary:"
    response = await llm.complete(prompt)
```

### Batch Document Processing

```python
async def batch_process_documents(
    processor: DocumentProcessor,
    documents: List[DocumentContent],
    config: Optional[ChunkingConfig] = None
) -> List[ProcessedDocument]:
    """Process multiple documents in batch."""

    processed_docs = []
    failed_docs = []

    for doc in documents:
        try:
            processed = await processor.process_document(doc, config)
            processed_docs.append(processed)

            print(f"✓ Processed: {doc.title} ({processed.total_chunks} chunks)")

        except ValueError as e:
            print(f"✗ Failed: {doc.title} - {e}")
            failed_docs.append((doc, str(e)))
        except Exception as e:
            print(f"✗ Error: {doc.title} - {e}")
            failed_docs.append((doc, str(e)))

    # Summary
    print(f"\nProcessed: {len(processed_docs)}/{len(documents)} documents")
    print(f"Failed: {len(failed_docs)} documents")

    # Calculate total statistics
    total_chunks = sum(p.total_chunks for p in processed_docs)
    total_chars = sum(p.total_characters for p in processed_docs)

    print(f"Total chunks: {total_chunks}")
    print(f"Total characters: {total_chars:,}")

    return processed_docs

# Usage
docs = [
    DocumentContent(content=text1, title="Doc 1"),
    DocumentContent(content=text2, title="Doc 2"),
    DocumentContent(content=text3, title="Doc 3"),
]

config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
processed_batch = await batch_process_documents(processor, docs, config)
```

## Integration with Kits

The Document Processor Port is used by the **RAG Kit** to prepare documents for vector storage and semantic search.

```python
from portico import compose
from portico.ports.document_processor import ChunkingConfig

# Configure RAG kit with document processing
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_api_key="sk-...",
            # Document processing configuration
            chunking_config=ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                respect_sentence_boundaries=True
            )
        )
    ]
)

await app.initialize()

# Access RAG service (uses document processor internally)
rag_service = app.kits["rag"].service

# Ingest document (automatically chunks it)
from portico.ports.document_processor import DocumentContent

doc = DocumentContent(
    content=long_article_text,
    title="Machine Learning Guide",
    source_url="https://example.com/ml-guide"
)

await rag_service.ingest_document(doc)
# Document is automatically chunked, embedded, and stored

# Query (searches across chunks)
results = await rag_service.query("What is supervised learning?")
```

The RAG Kit provides:

- Automatic document processing and chunking
- Vector embedding of chunks
- Semantic search across document chunks
- Context retrieval for LLM prompts
- Document and chunk management

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Choose Appropriate Chunk Sizes**: Match chunk size to your use case

   ```python
   # ✅ GOOD: Size matched to LLM context
   ChunkingConfig(
       chunk_size=1000,  # ~250 tokens
       chunk_overlap=200  # 20% overlap for context
   )

   # ❌ BAD: Chunks too large for embedding models
   ChunkingConfig(
       chunk_size=10000,  # Most embedding models max at ~512 tokens
       chunk_overlap=0
   )
   ```

2. **Use Overlap for Context Preservation**: Include overlap to avoid breaking semantic meaning

   ```python
   # ✅ GOOD: Overlap preserves context across chunks
   ChunkingConfig(
       chunk_size=1000,
       chunk_overlap=200  # 20% overlap
   )

   # ❌ BAD: No overlap loses context at boundaries
   ChunkingConfig(
       chunk_size=1000,
       chunk_overlap=0  # Sentences may be split
   )
   ```

3. **Respect Content Boundaries**: Keep semantic units together

   ```python
   # ✅ GOOD: Respect natural boundaries
   ChunkingConfig(
       respect_sentence_boundaries=True,
       respect_paragraph_boundaries=True,
       preserve_code_blocks=True
   )

   # ❌ BAD: May split mid-sentence
   ChunkingConfig(
       respect_sentence_boundaries=False,
       respect_paragraph_boundaries=False
   )
   ```

4. **Set Reasonable Limits**: Protect against resource exhaustion

   ```python
   # ✅ GOOD: Reasonable limits
   DocumentProcessorConfig(
       max_document_size=10_000_000,  # 10MB
       max_chunks_per_document=1000,
       processing_timeout_seconds=30.0
   )

   # ❌ BAD: No limits
   DocumentProcessorConfig(
       max_document_size=999_999_999,  # Too large
       max_chunks_per_document=99999
   )
   ```

5. **Handle Content Types Appropriately**: Use type-specific processing

   ```python
   # ✅ GOOD: Different configs for different types
   if content_type == ContentType.MARKDOWN:
       config = ChunkingConfig(preserve_code_blocks=True, preserve_headers=True)
   elif content_type == ContentType.TEXT:
       config = ChunkingConfig(respect_paragraph_boundaries=True)

   # ❌ BAD: One config for all types
   config = ChunkingConfig()  # Same for markdown, code, text, etc.
   ```

## FAQs

### What chunk size should I use?

Depends on your use case:

- **RAG with embedding models**: 500-1500 characters (~125-375 tokens) - fits most embedding model limits
- **LLM context augmentation**: 1000-2000 characters (~250-500 tokens) - provides good context without overwhelming
- **Detailed analysis**: 2000-4000 characters (~500-1000 tokens) - preserves more context per chunk

```python
# For RAG/embeddings (most common)
ChunkingConfig(chunk_size=1000, chunk_overlap=200)

# For LLM context windows
ChunkingConfig(chunk_size=2000, chunk_overlap=400)
```

### How much overlap should I use?

Typically 10-20% of chunk size:

```python
# Standard overlap (20%)
ChunkingConfig(chunk_size=1000, chunk_overlap=200)

# More overlap for critical context preservation (30%)
ChunkingConfig(chunk_size=1000, chunk_overlap=300)

# Minimal overlap for storage efficiency (10%)
ChunkingConfig(chunk_size=1000, chunk_overlap=100)
```

### How do I estimate token counts accurately?

Use the `estimate_token_count()` method with your target model:

```python
# Estimate for specific model
token_count = processor.estimate_token_count(
    text,
    model="gpt-4"  # Use your target model
)

# Configure processor with your model
config = DocumentProcessorConfig(
    default_tokenizer_model="gpt-4",
    tokens_per_chunk_target=500
)
```

### Can I implement custom chunking strategies?

Yes! Implement the `ChunkingStrategy` interface:

```python
from portico.ports.document_processor import ChunkingStrategy, ChunkingConfig

class CustomChunker(ChunkingStrategy):
    @property
    def strategy_name(self) -> str:
        return "custom"

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        # Your custom chunking logic
        chunks = []
        # ... implement your strategy
        return chunks

    def get_chunk_boundaries(
        self,
        text: str,
        config: ChunkingConfig
    ) -> List[tuple[int, int]]:
        # Return (start, end) positions for each chunk
        boundaries = []
        # ... calculate boundaries
        return boundaries

# Use custom strategy
custom_chunker = CustomChunker()
chunks = await processor.chunk_document(doc, custom_chunker, config)
```

### How do I handle very large documents?

Use the configuration limits:

```python
config = DocumentProcessorConfig(
    max_document_size=50_000_000,  # 50MB limit
    max_chunks_per_document=5000,
    processing_timeout_seconds=120.0
)

try:
    processed = await processor.process_document(huge_doc)
except ValueError as e:
    print(f"Document too large: {e}")
    # Handle: split into smaller docs, summarize first, etc.
```

### How do I preserve code blocks in markdown?

Enable code block preservation:

```python
config = ChunkingConfig(
    chunk_size=1200,
    preserve_code_blocks=True,  # Don't split code blocks
    preserve_headers=True,      # Include headers in metadata
    respect_paragraph_boundaries=True
)

processed = await processor.process_text(
    markdown_with_code,
    content_type=ContentType.MARKDOWN,
    chunking_config=config
)
```

### How do I access chunk metadata?

Each `ProcessedChunk` includes metadata and position information:

```python
for chunk in processed.chunks:
    print(f"Chunk {chunk.chunk_index}:")
    print(f"  Position: {chunk.start_char}-{chunk.end_char}")
    print(f"  Tokens: {chunk.token_count}")
    print(f"  Language: {chunk.language}")
    print(f"  Metadata: {chunk.metadata}")

    # Use in vector storage
    await vector_store.add(
        text=chunk.content,
        metadata={
            "doc_id": chunk.document_id,
            "chunk_idx": chunk.chunk_index,
            "start": chunk.start_char,
            "end": chunk.end_char,
            **chunk.metadata
        }
    )
```
