# Managed RAG Port

## Overview

The Managed RAG Port defines the contract for fully-managed RAG (Retrieval-Augmented Generation) platforms that bundle ingestion, embeddings, vector storage, and retrieval into a single managed service.

**Purpose**: Enable integration with end-to-end RAG platforms like Graphlit, Vectara, and Carbon that handle the complete RAG pipeline internally.

**Domain**: Knowledge management, document search, conversational AI

**Key Capabilities**:

- **Document Ingestion**: Text, URLs, files (PDF, DOCX, audio, video), and automated feeds
- **Semantic Retrieval**: Platform-managed vector search with internal embeddings
- **End-to-End RAG**: Complete retrieval + generation pipeline with source citations
- **Multi-turn Conversations**: Stateful conversation management with context
- **Document Management**: CRUD operations, metadata updates, and listing
- **Feed Management**: Automated ingestion from Google Drive, Slack, RSS, GitHub, etc.
- **Multi-modal Support**: Process text, audio, video, images depending on platform

**Port Type**: Provider

**When to Use**:

- Rapid prototyping requiring production-ready RAG in hours
- Multi-modal RAG applications (audio/video transcription, image analysis)
- Knowledge graph-based retrieval (GraphRAG)
- Applications needing built-in connectors (Google Drive, Slack, etc.)
- Teams preferring managed infrastructure over DIY RAG components
- Use cases requiring automated feed monitoring and continuous ingestion

**When NOT to Use**:

- Maximum control over embedding models, chunking strategies, or vector indexes
- Cost-sensitive applications with very high query volume
- On-premise/air-gapped deployments requiring full data sovereignty
- Custom RAG architectures with specialized retrieval pipelines

## Domain Models

### RetrievalResult

Represents a single result from semantic search retrieval.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `document_id` | `str` | Yes | - | Platform-specific document identifier |
| `content` | `str` | Yes | - | Retrieved content (may be full document or chunk) |
| `score` | `float` | Yes | - | Relevance score (platform-specific, typically 0-1) |
| `metadata` | `Dict[str, Any]` | Yes | - | Document metadata (title, author, date, etc.) |
| `source_url` | `Optional[str]` | No | `None` | Original source URL if available |
| `chunk_index` | `Optional[int]` | No | `None` | Chunk index if content is chunked |

**Example**:

```python
from portico.ports.managed_rag import RetrievalResult

result = RetrievalResult(
    document_id="doc-12345",
    content="Portico is a Python framework...",
    score=0.92,
    metadata={"title": "Portico Documentation", "author": "Portico Team"},
    source_url="https://docs.portico.dev",
    chunk_index=0,
)
```

### ManagedRAGResponse

Complete response from an end-to-end RAG query including generated text and sources.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `response` | `str` | Yes | - | Generated text response from LLM |
| `sources` | `List[RetrievalResult]` | Yes | - | Source documents used for generation |
| `conversation_id` | `Optional[str]` | No | `None` | Conversation ID if using conversation mode |
| `usage` | `Optional[Dict[str, Any]]` | No | `None` | Platform-specific usage metrics (tokens, API calls) |
| `metadata` | `Dict[str, Any]` | No | `{}` | Additional response metadata |

**Example**:

```python
from portico.ports.managed_rag import ManagedRAGResponse, RetrievalResult

response = ManagedRAGResponse(
    response="Portico is a Python framework for building GPT-powered applications...",
    sources=[
        RetrievalResult(
            document_id="doc-1",
            content="Portico framework...",
            score=0.95,
            metadata={"title": "Overview"},
        )
    ],
    conversation_id="conv-789",
    usage={"tokens": 250, "api_calls": 1},
)
```

### DocumentMetadata

Metadata for a document stored in the managed platform.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | - | Platform-specific document ID |
| `name` | `str` | Yes | - | Document name or title |
| `created_at` | `datetime` | Yes | - | Document creation timestamp |
| `updated_at` | `Optional[datetime]` | No | `None` | Last update timestamp |
| `size_bytes` | `Optional[int]` | No | `None` | Document size in bytes |
| `content_type` | `Optional[str]` | No | `None` | MIME type (e.g., "application/pdf") |
| `metadata` | `Dict[str, Any]` | No | `{}` | Custom metadata fields |

**Example**:

```python
from datetime import datetime
from portico.ports.managed_rag import DocumentMetadata

doc_meta = DocumentMetadata(
    id="doc-12345",
    name="Q4 Report.pdf",
    created_at=datetime.now(),
    updated_at=datetime.now(),
    size_bytes=524288,
    content_type="application/pdf",
    metadata={"department": "Finance", "year": 2025},
)
```

### FeedMetadata

Metadata for an automated ingestion feed that monitors external sources.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | Yes | - | Platform-specific feed ID |
| `name` | `str` | Yes | - | Feed name or title |
| `feed_type` | `str` | Yes | - | Feed type: "web", "rss", "google_drive", "slack", etc. |
| `status` | `str` | Yes | - | Feed status: "active", "paused", "error" |
| `document_count` | `int` | Yes | - | Number of documents ingested by this feed |
| `last_update` | `Optional[datetime]` | No | `None` | Last successful feed update timestamp |
| `metadata` | `Dict[str, Any]` | No | `{}` | Custom feed metadata |

**Example**:

```python
from datetime import datetime
from portico.ports.managed_rag import FeedMetadata

feed = FeedMetadata(
    id="feed-789",
    name="Engineering Docs",
    feed_type="google_drive",
    status="active",
    document_count=142,
    last_update=datetime.now(),
    metadata={"folder_id": "abc123", "update_frequency": "hourly"},
)
```

## Port Interface

### ManagedRAGPlatform

The `ManagedRAGPlatform` abstract base class defines the contract for fully-managed RAG service providers. Platforms like Graphlit, Vectara, and Carbon handle embedding generation, vector storage, and retrieval internally - you don't control individual components.

**Location**: `portico.ports.managed_rag.ManagedRAGPlatform`

**Key Characteristics**:

- **Ingestion**: Platform processes and indexes documents automatically
- **Embeddings**: Generated internally (model selection may be limited)
- **Vector Storage**: Managed and scaled automatically
- **Retrieval**: Combines vector search with platform-specific enhancements
- **Knowledge Graph**: Some platforms extract entities and relationships (GraphRAG)
- **Connectors**: Built-in integrations for Google Drive, Slack, GitHub, RSS, etc.

#### Ingestion Methods

##### ingest_document

```python
async def ingest_document(
    content: str,
    metadata: Dict[str, Any],
    source_id: Optional[str] = None,
) -> str
```

Ingest text document into the platform for indexing and retrieval.

**Parameters**:

- `content`: Document text content to index
- `metadata`: Document metadata (title, author, category, date, etc.)
- `source_id`: Optional external source identifier for tracking

**Returns**: Platform-specific document ID

**Raises**:
- `ValidationError`: Invalid content or metadata format
- `ExternalServiceError`: Platform API error

**Example**:

```python
doc_id = await platform.ingest_document(
    content="Portico is a Python framework for GPT-powered applications...",
    metadata={
        "title": "Portico Overview",
        "author": "Portico Team",
        "category": "documentation",
        "date": "2025-01-15",
    },
    source_id="portico-docs-001",
)
```

##### ingest_from_url

```python
async def ingest_from_url(
    url: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str
```

Ingest document directly from a URL. Platform fetches, processes, and indexes the content automatically. Supports web pages, PDFs, Word documents, images, videos, and more depending on platform capabilities.

**Parameters**:

- `url`: Document URL to fetch and ingest
- `metadata`: Optional metadata to attach to the document

**Returns**: Platform-specific document ID

**Example**:

```python
doc_id = await platform.ingest_from_url(
    url="https://docs.portico.dev/getting-started",
    metadata={"source": "official_docs", "priority": "high"},
)
```

##### ingest_file

```python
async def ingest_file(
    file_content: Any,  # BinaryIO or bytes
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[tuple[str, str]]] = None,
) -> str
```

Ingest binary file content into the platform. Supports diverse formats including documents (PDF, DOCX, TXT), audio (MP3, WAV), video (MP4, MOV), and images (JPEG, PNG) depending on platform capabilities.

**Parameters**:

- `file_content`: File content as BinaryIO stream or bytes
- `filename`: Original filename (used for MIME type detection)
- `metadata`: Optional metadata to attach (title, author, etc.)
- `tags`: Optional list of (key, value) tag tuples for filtering

**Returns**: Platform-specific document ID

**Example**:

```python
with open("research_paper.pdf", "rb") as f:
    doc_id = await platform.ingest_file(
        file_content=f,
        filename="research_paper.pdf",
        metadata={"author": "John Doe", "department": "Research"},
        tags=[("category", "research"), ("year", "2025")],
    )
```

##### ingest_from_feed

```python
async def ingest_from_feed(
    feed_config: Dict[str, Any],
) -> str
```

Create automated feed for continuous ingestion from external sources. Examples include Google Drive folder monitoring, Slack channel indexing, RSS feed tracking, and GitHub repository watching.

**Parameters**:

- `feed_config`: Platform-specific feed configuration dictionary
  ```python
  {
      "name": "My Feed",
      "type": "google_drive",  # or "slack", "rss", "github", etc.
      "config": {...}  # Platform-specific config
  }
  ```

**Returns**: Feed ID for monitoring and management

**Example**:

```python
feed_id = await platform.ingest_from_feed({
    "name": "Engineering Docs",
    "type": "google_drive",
    "config": {
        "folder_id": "1abc...xyz",
        "update_frequency": "hourly",
        "file_types": ["pdf", "docx", "txt"],
    },
})
```

##### ingest_batch

```python
async def ingest_batch(
    documents: List[Dict[str, Any]],
) -> List[str]
```

Batch ingest multiple documents in a single operation for efficiency.

**Parameters**:

- `documents`: List of documents, each with "content" and "metadata" keys

**Returns**: List of document IDs in same order as input

**Example**:

```python
doc_ids = await platform.ingest_batch([
    {
        "content": "Document 1 content...",
        "metadata": {"title": "Doc 1", "category": "tech"},
    },
    {
        "content": "Document 2 content...",
        "metadata": {"title": "Doc 2", "category": "business"},
    },
])
```

#### Retrieval Methods

##### retrieve

```python
async def retrieve(
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
) -> List[RetrievalResult]
```

Retrieve relevant documents using the platform's internal semantic search pipeline. Platform handles query embedding, vector similarity search, knowledge graph traversal (if supported), and result ranking automatically.

**Parameters**:

- `query`: Natural language search query
- `k`: Number of results to return
- `filters`: Platform-specific metadata filters for scoping search
- `namespace`: Optional namespace/collection for multi-tenancy

**Returns**: List of retrieval results sorted by relevance score

**Example**:

```python
results = await platform.retrieve(
    query="How do I configure authentication in Portico?",
    k=5,
    filters={"category": "documentation", "version": "latest"},
    namespace="prod-docs",
)

for result in results:
    print(f"Score: {result.score:.2f} - {result.metadata['title']}")
    print(f"Content: {result.content[:200]}...")
```

#### RAG Query Methods

##### query

```python
async def query(
    query: str,
    conversation_id: Optional[str] = None,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> ManagedRAGResponse
```

Execute complete end-to-end RAG query combining retrieval and generation. Platform handles:
1. Embedding the query
2. Retrieving relevant sources
3. Constructing context from sources
4. Generating response with LLM
5. Returning response with source citations

**Parameters**:

- `query`: User question or prompt
- `conversation_id`: Optional conversation ID for multi-turn context
- `k`: Number of sources to retrieve
- `filters`: Metadata filters for retrieval scoping
- `llm_config`: LLM configuration overrides (model, temperature, etc.)

**Returns**: Generated response with source citations and metadata

**Example**:

```python
response = await platform.query(
    query="What are the main features of Portico?",
    k=5,
    filters={"category": "documentation"},
    llm_config={
        "model": "gpt-4-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
    },
)

print(f"Response: {response.response}")
print(f"Sources ({len(response.sources)}):")
for source in response.sources:
    print(f"  - {source.metadata.get('title')} (score: {source.score:.2f})")
```

#### Conversation Management

##### create_conversation

```python
async def create_conversation(
    name: str,
    llm_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> str
```

Create conversation context for multi-turn RAG interactions with persistent history.

**Parameters**:

- `name`: Conversation name or title
- `llm_config`: LLM configuration (provider, model, temperature, etc.)
- `system_prompt`: Optional system prompt for conversation behavior

**Returns**: Conversation ID for subsequent queries

**Example**:

```python
conv_id = await platform.create_conversation(
    name="User Support Chat",
    llm_config={
        "provider": "openai",
        "model": "gpt-4-turbo",
        "temperature": 0.7,
    },
    system_prompt="You are a helpful assistant for Portico framework users.",
)
```

##### get_conversation_history

```python
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]
```

Retrieve message history for a conversation.

**Returns**: List of messages with role, content, and citations

**Example**:

```python
history = await platform.get_conversation_history(
    conversation_id=conv_id,
    limit=20,
)

for msg in history:
    print(f"{msg['role']}: {msg['content']}")
```

##### delete_conversation

```python
async def delete_conversation(conversation_id: str) -> bool
```

Delete conversation and its complete message history.

**Returns**: True if deleted successfully

#### Document Management

##### get_document

```python
async def get_document(document_id: str) -> Optional[DocumentMetadata]
```

Retrieve document metadata by ID.

**Returns**: DocumentMetadata object or None if not found

##### delete_document

```python
async def delete_document(document_id: str) -> bool
```

Delete document from the platform, removing it from search results.

**Returns**: True if deleted successfully

##### list_documents

```python
async def list_documents(
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[DocumentMetadata]
```

List documents with optional filtering and pagination.

**Parameters**:

- `filters`: Platform-specific filters (e.g., metadata queries)
- `limit`: Maximum documents to return
- `offset`: Pagination offset

**Returns**: List of document metadata

**Example**:

```python
docs = await platform.list_documents(
    filters={"category": "engineering", "year": 2025},
    limit=50,
    offset=0,
)
```

##### update_document_metadata

```python
async def update_document_metadata(
    document_id: str,
    metadata: Dict[str, Any],
) -> bool
```

Update document metadata without re-indexing content.

**Returns**: True if updated successfully

#### Feed Management

##### list_feeds

```python
async def list_feeds() -> List[FeedMetadata]
```

List all configured feeds.

**Returns**: List of feed metadata

##### get_feed

```python
async def get_feed(feed_id: str) -> Optional[FeedMetadata]
```

Get feed metadata by ID.

**Returns**: FeedMetadata object or None if not found

##### pause_feed

```python
async def pause_feed(feed_id: str) -> bool
```

Pause feed processing temporarily.

**Returns**: True if paused successfully

##### resume_feed

```python
async def resume_feed(feed_id: str) -> bool
```

Resume paused feed processing.

**Returns**: True if resumed successfully

##### delete_feed

```python
async def delete_feed(feed_id: str) -> bool
```

Delete feed and optionally its ingested documents.

**Returns**: True if deleted successfully

#### Health & Statistics

##### get_stats

```python
async def get_stats() -> Dict[str, Any]
```

Get platform statistics and usage metrics.

**Returns**: Dictionary with platform-specific metrics such as total documents, conversations, storage used, etc.

**Example**:

```python
stats = await platform.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Storage used: {stats['storage_used_bytes'] / 1024 / 1024:.2f} MB")
```

##### health_check

```python
async def health_check() -> bool
```

Check platform health and connectivity.

**Returns**: True if platform is healthy and accessible

## Common Patterns

### Basic Document Ingestion and Query

```python
from portico.ports.managed_rag import ManagedRAGPlatform

async def ingest_and_query(platform: ManagedRAGPlatform):
    # Ingest document
    doc_id = await platform.ingest_document(
        content="Portico uses hexagonal architecture with ports and adapters...",
        metadata={"title": "Architecture Guide", "version": "2.0"},
    )

    # Query with RAG
    response = await platform.query(
        query="What architecture does Portico use?",
        k=3,
    )

    print(f"Answer: {response.response}")
    print(f"Based on {len(response.sources)} sources")
```

### Multi-turn Conversation

```python
async def conversational_rag(platform: ManagedRAGPlatform):
    # Create conversation
    conv_id = await platform.create_conversation(
        name="Technical Discussion",
        llm_config={"model": "gpt-4-turbo", "temperature": 0.7},
        system_prompt="You are an expert on the Portico framework.",
    )

    # First question
    response1 = await platform.query(
        query="What is Portico?",
        conversation_id=conv_id,
    )
    print(f"Q1: {response1.response}")

    # Follow-up question (context maintained)
    response2 = await platform.query(
        query="How do I install it?",
        conversation_id=conv_id,
    )
    print(f"Q2: {response2.response}")

    # Get conversation history
    history = await platform.get_conversation_history(conv_id)
    print(f"Conversation has {len(history)} messages")
```

### File Upload and Retrieval

```python
async def upload_and_search(platform: ManagedRAGPlatform):
    # Upload PDF
    with open("documentation.pdf", "rb") as f:
        doc_id = await platform.ingest_file(
            file_content=f,
            filename="documentation.pdf",
            metadata={"type": "technical_doc", "version": "1.0"},
            tags=[("department", "engineering")],
        )

    # Wait for processing (platform-dependent)
    await asyncio.sleep(5)

    # Search uploaded document
    results = await platform.retrieve(
        query="How to configure the database?",
        k=3,
        filters={"type": "technical_doc"},
    )

    for result in results:
        print(f"Found in: {result.metadata['title']}")
        print(f"Score: {result.score:.2f}")
```

### Automated Feed Setup

```python
async def setup_continuous_ingestion(platform: ManagedRAGPlatform):
    # Create Google Drive feed
    feed_id = await platform.ingest_from_feed({
        "name": "Team Documentation",
        "type": "google_drive",
        "config": {
            "folder_id": "1234abcd",
            "update_frequency": "hourly",
            "file_types": ["pdf", "docx", "txt"],
        },
    })

    # Monitor feed status
    feed = await platform.get_feed(feed_id)
    print(f"Feed: {feed.name}")
    print(f"Status: {feed.status}")
    print(f"Documents: {feed.document_count}")

    # Pause if needed
    if feed.document_count > 1000:
        await platform.pause_feed(feed_id)
```

## Integration with Kits

The Managed RAG Port is used by the **RAG Kit** to provide managed platform capabilities.

```python
from portico import compose

# Configure with Graphlit adapter
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            rag_provider="graphlit",
            graphlit_organization_id="org-123",
            graphlit_environment_id="env-456",
            graphlit_jwt_secret="your-jwt-secret",
            llm_provider="openai",
            llm_api_key="sk-...",
        ),
    ],
)

# Access managed RAG service
rag_service = app.kits["rag"].service

# Ingest document
doc_id = await rag_service.ingest_document(
    content="Your document content...",
    metadata={"title": "Document Title"},
)

# Query with RAG
response = await rag_service.query(
    query="What is this document about?",
    k=5,
)
```

See the RAG Kit documentation for complete usage details.

## Best Practices

1. **Always Include Rich Metadata**: Add comprehensive metadata during ingestion for better filtering and retrieval

   ```python
   # ✅ GOOD - Rich metadata
   await platform.ingest_document(
       content=content,
       metadata={
           "title": "User Guide",
           "author": "Engineering Team",
           "category": "documentation",
           "version": "2.0",
           "date": "2025-01-15",
           "keywords": ["auth", "security", "users"],
       },
   )

   # ❌ BAD - Minimal metadata
   await platform.ingest_document(
       content=content,
       metadata={"title": "doc"},
   )
   ```

2. **Use Filters to Scope Retrieval**: Leverage metadata filters to improve relevance and reduce noise

   ```python
   # ✅ GOOD - Filtered retrieval
   results = await platform.retrieve(
       query="authentication setup",
       k=5,
       filters={"category": "documentation", "version": "latest"},
   )

   # ❌ BAD - Unfiltered retrieval
   results = await platform.retrieve(query="authentication setup", k=5)
   ```

3. **Handle Platform-Specific Errors**: Wrap platform calls in try-except blocks and handle specific exceptions

   ```python
   # ✅ GOOD - Explicit error handling
   from portico.exceptions import ExternalServiceError, ValidationError

   try:
       doc_id = await platform.ingest_document(content, metadata)
   except ValidationError as e:
       logger.error(f"Invalid document: {e}")
       return None
   except ExternalServiceError as e:
       logger.error(f"Platform error: {e}")
       # Retry with backoff
       return await retry_with_backoff(platform.ingest_document, content, metadata)
   ```

4. **Use Conversations for Multi-turn Interactions**: Create conversations instead of sending isolated queries when context matters

   ```python
   # ✅ GOOD - Conversation context
   conv_id = await platform.create_conversation("Support Chat", llm_config={})

   response1 = await platform.query("What is Portico?", conversation_id=conv_id)
   response2 = await platform.query("How do I install it?", conversation_id=conv_id)
   # Second query benefits from first query's context

   # ❌ BAD - Isolated queries lose context
   response1 = await platform.query("What is Portico?")
   response2 = await platform.query("How do I install it?")
   # Second query doesn't know what "it" refers to
   ```

5. **Monitor Platform Health**: Regularly check platform health before critical operations

   ```python
   # ✅ GOOD - Health check before batch operation
   healthy = await platform.health_check()
   if not healthy:
       logger.warning("Platform unhealthy, deferring batch ingestion")
       return

   doc_ids = await platform.ingest_batch(documents)
   ```

6. **Batch Operations for Efficiency**: Use batch methods when ingesting multiple documents

   ```python
   # ✅ GOOD - Batch ingestion
   documents = [{"content": doc.content, "metadata": doc.metadata} for doc in docs]
   doc_ids = await platform.ingest_batch(documents)

   # ❌ BAD - Individual ingestion in loop
   doc_ids = []
   for doc in docs:
       doc_id = await platform.ingest_document(doc.content, doc.metadata)
       doc_ids.append(doc_id)
   ```

7. **Clean Up Resources**: Delete conversations and documents when no longer needed

   ```python
   # ✅ GOOD - Cleanup after use
   try:
       response = await platform.query(query, conversation_id=conv_id)
       # ... use response ...
   finally:
       await platform.delete_conversation(conv_id)
   ```

## FAQs

### What's the difference between retrieve() and query()?

`retrieve()` returns raw relevant documents with scores, while `query()` performs full RAG by retrieving documents AND generating a natural language response using an LLM.

Use `retrieve()` when:
- You want to display source documents to users
- You need custom post-processing of retrieved content
- You're building your own generation pipeline

Use `query()` when:
- You want a complete answer with citations
- You're building a conversational interface
- You want the platform to handle the entire RAG pipeline

### How do I handle platform-specific features?

Different platforms have different capabilities (e.g., Graphlit supports audio/video, knowledge graphs). Check the adapter documentation and use the `metadata` dict to pass platform-specific options:

```python
# Graphlit-specific: audio transcription settings
doc_id = await platform.ingest_file(
    file_content=audio_file,
    filename="podcast.mp3",
    metadata={
        "transcription": {
            "model": "whisper",
            "language": "en",
        },
    },
)
```

### Should I use Managed RAG or DIY RAG (embedding + vector store)?

**Use Managed RAG when**:
- You want to prototype quickly (hours vs weeks)
- You need multi-modal support (audio, video, images)
- You want automated feed ingestion (Google Drive, Slack, etc.)
- Your team prefers managed services over infrastructure
- You're building knowledge graph applications

**Use DIY RAG when**:
- You need maximum control over chunking, embeddings, and retrieval
- You have very high query volume (cost optimization)
- You require on-premise/air-gapped deployment
- You want to use specific embedding models or vector databases
- You need custom retrieval pipelines (hybrid search, re-ranking, etc.)

### How do I implement multi-tenancy?

Use the `namespace` parameter in `retrieve()` and `query()` to isolate data by tenant, and include tenant IDs in document metadata:

```python
# Ingestion - tag with tenant
await platform.ingest_document(
    content=content,
    metadata={"tenant_id": "acme-corp", "title": "Document"},
)

# Retrieval - scope to tenant
results = await platform.retrieve(
    query="search query",
    namespace="acme-corp",
    filters={"tenant_id": "acme-corp"},
)
```

### What file formats are supported?

Support varies by platform, but common formats include:
- **Documents**: PDF, DOCX, TXT, MD, HTML
- **Audio**: MP3, WAV, M4A
- **Video**: MP4, MOV, AVI
- **Images**: JPEG, PNG, GIF
- **Data**: JSON, CSV, XML

Check your platform adapter documentation for specific supported formats.

### How do I monitor platform usage and costs?

Use `get_stats()` to monitor usage metrics:

```python
stats = await platform.get_stats()

# Platform-specific metrics may include:
print(f"Documents: {stats.get('total_documents', 0)}")
print(f"Conversations: {stats.get('total_conversations', 0)}")
print(f"Storage: {stats.get('storage_used_bytes', 0) / 1024 / 1024:.2f} MB")
print(f"API Calls (month): {stats.get('api_calls_this_month', 0)}")
```

Additionally, check response metadata for per-query costs:

```python
response = await platform.query(query)
if response.usage:
    print(f"Tokens used: {response.usage.get('tokens', 0)}")
```

### What happens if the platform is temporarily unavailable?

Implement retry logic with exponential backoff:

```python
import asyncio
from portico.exceptions import ExternalServiceError

async def ingest_with_retry(platform, content, metadata, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await platform.ingest_document(content, metadata)
        except ExternalServiceError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Platform error, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise
```

Also monitor platform health before critical operations:

```python
healthy = await platform.health_check()
if not healthy:
    # Defer non-critical operations or use fallback
    return await fallback_search(query)
```
