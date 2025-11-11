# RAG Kit

## Overview

The RAG (Retrieval-Augmented Generation) Kit provides document retrieval and context-aware response generation capabilities. It enables applications to answer questions based on indexed documents by combining semantic search with LLM generation, grounding responses in factual content.

**Purpose**: Ground LLM responses in factual document content through semantic retrieval and context injection.

**Domain**: Document retrieval, semantic search, context-aware generation, vector embeddings

**Capabilities**:

- Document ingestion and chunking
- Embedding generation for semantic search
- Vector similarity search across document collections
- Context construction from retrieved documents
- LLM response generation with source citations
- Two operational modes: DIY (component-based) and Managed (platform-based)
- Conversation-aware RAG with multi-turn context
- Automated feeds for continuous document ingestion
- Document metadata management and filtering
- Performance metrics and health monitoring

**Architecture Type**: Stateless kit (no database models, uses vector stores and external platforms)

**When to Use**:

- Question answering over document collections
- Customer support chatbots with knowledge base
- Research assistants and document analysis
- Content recommendation systems
- Technical documentation search
- Fact-checking and source verification

## Quick Start

### DIY RAG (Component-Based)

```python
from portico import compose

# Development configuration with in-memory vector store
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_api_key="sk-...",
            vector_store_type="memory",
        ),
    ],
)

await app.initialize()

# Get RAG service
rag_service = app.kits["rag"].service

# Index a document
from portico.ports.vector_store import Document

document = Document(
    id="doc_1",
    content="Portico is a Python framework for building GPT-powered applications using hexagonal architecture.",
    metadata={"title": "What is Portico", "source": "docs"}
)

await rag_service.vector_store.add_documents([document])

# Query with RAG
from portico.kits.rag import RAGQuery

query = RAGQuery(
    query="What is Portico?",
    k=3,
    temperature=0.3
)

response = await rag_service.query(query)

print(f"Answer: {response.response}")
print(f"Sources: {len(response.sources)}")
for i, source in enumerate(response.sources, 1):
    print(f"  {i}. {source.title} (score: {source.score:.2f})")
```

### Managed RAG (Platform-Based)

```python
# Using Graphlit managed platform
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            use_managed_rag=True,
            managed_rag_provider="graphlit",
            managed_rag_config={
                "api_key": "your-graphlit-api-key",
                "platform_url": "https://api.graphlit.io/api/v1/graphql",
            },
        ),
    ],
)

await app.initialize()

rag_service = app.kits["rag"].service

# Ingest document from URL
document_id = await rag_service.ingest_from_url(
    url="https://docs.portico.dev/overview",
    metadata={"category": "documentation"}
)

# Query (retrieval + generation handled by platform)
response = await rag_service.query(
    query="What is hexagonal architecture?",
    k=5
)

print(f"Answer: {response.response}")
```

## Core Concepts

### RAGService (DIY Mode)

The `RAGService` orchestrates the complete RAG pipeline: query embedding, vector search, context construction, and LLM generation.

**Key Methods**:

- `query()` - Complete RAG pipeline (retrieve + generate)
- `retrieve_sources()` - Retrieve relevant documents without generation
- `generate_with_context()` - Generate response with provided context
- `get_rag_metrics()` - Get performance metrics
- `health_check()` - Check component health

**Pipeline Steps**:

1. **Query Embedding**: Convert user query to vector
2. **Vector Search**: Find similar documents in vector store
3. **Context Construction**: Format retrieved documents into context
4. **LLM Generation**: Generate response using context
5. **Citation Extraction**: Return response with source citations

### ManagedRAGService (Managed Mode)

The `ManagedRAGService` wraps managed RAG platforms (Graphlit, Vectara) with validation, logging, and monitoring.

**Key Methods**:

- `ingest_document()` - Ingest text document
- `ingest_from_url()` - Ingest from web URL
- `upload_document()` - Upload binary file (PDF, DOCX, MP3, MP4, etc.)
- `ingest_batch()` - Batch ingest multiple documents
- `query()` - Execute RAG query with conversation support
- `retrieve()` - Semantic search without generation
- `create_conversation()` - Create multi-turn conversation context
- `create_feed()` - Set up automated content ingestion

### Vector Store

Vector stores index document embeddings for semantic similarity search. Supported backends:

- **Memory**: In-memory storage (development/testing)
- **Pinecone**: Managed vector database (production)

### Embedding Provider

Embedding providers generate vector representations of text for semantic search:

- **OpenAI**: `text-embedding-3-small` (1536 dimensions), `text-embedding-3-large` (3072 dimensions)

### Document Processor

Document processors chunk long documents into smaller segments for efficient retrieval:

- **BasicDocumentProcessor**: Simple chunk-based splitting with overlap

### RAG Configuration

RAG behavior is controlled through `RAGConfig` dataclass:

- Retrieval settings (k, similarity threshold, reranking)
- Context construction (max tokens, source separator, metadata inclusion)
- Generation settings (model, temperature, system prompt)
- Quality control (minimum sources, citation requirements)

## Configuration

### RagKitConfig

```python
from dataclasses import dataclass

@dataclass
class RagKitConfig:
    # Mode selection
    use_managed_rag: bool = False
    managed_rag_provider: Optional[str] = None  # "graphlit", "vectara"
    managed_rag_config: Optional[Dict[str, Any]] = None

    # DIY RAG Configuration (when use_managed_rag=False)
    llm_provider: str = "openai"                 # "openai" or "anthropic"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None

    embedding_provider: str = "openai"
    embedding_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"

    vector_store_type: str = "memory"            # "memory" or "pinecone"
    vector_store_config: Optional[VectorStoreConfig] = None
    document_processor_config: Optional[DocumentProcessorConfig] = None
    rag_config: Optional[RAGConfig] = None
```

### Composing DIY RAG

```python
from portico import compose

# Development: Memory vector store
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_api_key="sk-...",  # Defaults to llm_api_key if not provided
            vector_store_type="memory",
        ),
    ],
)

# Production: Pinecone vector store
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_api_key="sk-...",
            vector_store_type="pinecone",
            vector_store_config={
                "api_key": "pinecone-api-key",
                "index_name": "my-docs",
                "dimension": 1536,  # Must match embedding model
                "cloud": "aws",
                "region": "us-east-1",
                "metric": "cosine",
            },
        ),
    ],
)
```

### Composing Managed RAG

```python
# Graphlit managed platform
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            use_managed_rag=True,
            managed_rag_provider="graphlit",
            managed_rag_config={
                "api_key": "your-graphlit-api-key",
                "platform_url": "https://api.graphlit.io/api/v1/graphql",
                "organization_id": "your-org-id",  # Optional
                "environment_id": "your-env-id",   # Optional
            },
        ),
    ],
)
```

## Usage Examples

### 1. Document Ingestion and Query (DIY)

```python
from portico.ports.vector_store import Document
from portico.kits.rag import RAGQuery

rag_service = app.kits["rag"].service

# Prepare documents
documents = [
    Document(
        id="doc_1",
        content="Python 3.13 introduces improved error messages and performance optimizations.",
        metadata={"title": "Python 3.13 Release", "category": "changelog"}
    ),
    Document(
        id="doc_2",
        content="Type hints in Python enable static analysis and improve code documentation.",
        metadata={"title": "Python Type Hints", "category": "tutorial"}
    ),
]

# Index documents (automatically chunks and embeds)
await rag_service.vector_store.add_documents(documents)

# Query
query = RAGQuery(
    query="What's new in Python 3.13?",
    k=3,
    similarity_threshold=0.7,
    temperature=0.3,
    include_sources=True
)

response = await rag_service.query(query)

print(f"Answer: {response.response}")
print(f"Used {response.tokens_used} tokens in {response.total_time_ms:.0f}ms")

# Show sources
for source in response.sources:
    print(f"- {source.title} (relevance: {source.score:.2f})")
```

### 2. Managed RAG with File Upload

```python
rag_service = app.kits["rag"].service

# Upload PDF document
with open("technical_manual.pdf", "rb") as f:
    document_id = await rag_service.upload_document(
        file_content=f.read(),
        filename="technical_manual.pdf",
        metadata={"category": "manuals", "version": "2.0"},
        tags=[("department", "engineering"), ("confidential", "false")]
    )

print(f"Document uploaded: {document_id}")

# Query across all documents
response = await rag_service.query(
    query="What is the recommended maintenance schedule?",
    k=5,
    filters={"category": "manuals"}  # Filter by metadata
)

print(response.response)
```

### 3. Multi-Turn Conversation with RAG

```python
# Create conversation context
conversation_id = await rag_service.create_conversation(
    name="Product Support Chat",
    system_prompt="You are a helpful product support assistant."
)

# First question
response1 = await rag_service.query(
    query="How do I install the software?",
    conversation_id=conversation_id,
    k=3
)

print(f"Assistant: {response1.response}")

# Follow-up question (conversation history maintained)
response2 = await rag_service.query(
    query="What are the system requirements?",
    conversation_id=conversation_id,
    k=3
)

print(f"Assistant: {response2.response}")

# Get conversation history
messages = await rag_service.get_conversation_history(
    conversation_id=conversation_id
)

for msg in messages:
    print(f"{msg['role']}: {msg['content'][:100]}...")
```

### 4. Custom Context Construction

```python
# Retrieve sources without generating response
sources = await rag_service.retrieve_sources(
    query="Python decorators",
    k=5,
    threshold=0.6,
    namespace="python_docs"
)

# Build custom context
context_parts = []
for i, source in enumerate(sources, 1):
    context_parts.append(f"[{i}] {source.title}:\n{source.content}\n")

context = "\n---\n".join(context_parts)

# Generate with custom prompt
response_text = await rag_service.generate_with_context(
    query="Explain decorators with examples from the sources",
    context=context,
    temperature=0.4,
    max_tokens=800,
    system_prompt="You are a Python tutor. Use the provided documentation to explain concepts with examples."
)

print(response_text)
```

### 5. Batch Document Ingestion (Managed)

```python
documents = [
    {
        "content": "Product A specifications...",
        "metadata": {"product": "A", "type": "specs", "version": "1.0"}
    },
    {
        "content": "Product A user guide...",
        "metadata": {"product": "A", "type": "guide", "version": "1.0"}
    },
    {
        "content": "Product B specifications...",
        "metadata": {"product": "B", "type": "specs", "version": "2.0"}
    },
]

# Batch ingest
document_ids = await rag_service.ingest_batch(documents)

print(f"Indexed {len(document_ids)} documents")

# Query with metadata filtering
response = await rag_service.query(
    query="What are the specs for Product B?",
    filters={"product": "B", "type": "specs"},
    k=3
)

print(response.response)
```

## Domain Models

### RAGQuery

User query with RAG configuration.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique query identifier (UUID) |
| `query` | `str` | User question text |
| `k` | `int` | Number of documents to retrieve (default: 5) |
| `similarity_threshold` | `Optional[float]` | Minimum similarity score |
| `namespace` | `Optional[str]` | Vector store namespace |
| `model` | `Optional[str]` | LLM model override |
| `temperature` | `Optional[float]` | Generation temperature |
| `max_tokens` | `Optional[int]` | Maximum tokens to generate |
| `include_sources` | `bool` | Include source citations (default: True) |
| `metadata_filters` | `Dict[str, Any]` | Document metadata filters |
| `context_template` | `Optional[str]` | Custom context prompt template |
| `system_prompt` | `Optional[str]` | Custom system prompt |
| `expand_query` | `bool` | Expand query with synonyms |
| `rerank_results` | `bool` | Rerank retrieved results |
| `created_at` | `datetime` | Query creation timestamp |

### RAGResponse

Generated response with sources and metadata.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Response identifier (UUID) |
| `query` | `str` | Original query text |
| `response` | `str` | Generated answer text |
| `sources` | `List[SourceCitation]` | Source citations |
| `context` | `Optional[RetrievalContext]` | Retrieved context details |
| `model` | `Optional[str]` | Model used for generation |
| `tokens_used` | `Optional[int]` | Total tokens consumed |
| `generation_time_ms` | `Optional[float]` | Generation duration |
| `confidence_score` | `Optional[float]` | Model confidence |
| `relevance_score` | `Optional[float]` | Context relevance |
| `total_time_ms` | `Optional[float]` | Total pipeline duration |
| `created_at` | `datetime` | Response creation timestamp |

### SourceCitation

Citation information for a retrieved document.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Citation identifier |
| `content` | `str` | Document/chunk content |
| `score` | `float` | Similarity score (0.0-1.0) |
| `metadata` | `Dict[str, Any]` | Document metadata |
| `document_id` | `Optional[str]` | Source document ID |
| `chunk_index` | `Optional[int]` | Chunk position in document |
| `start_char` | `Optional[int]` | Start character offset |
| `end_char` | `Optional[int]` | End character offset |
| `title` | `Optional[str]` | Document title |
| `url` | `Optional[str]` | Source URL |
| `author` | `Optional[str]` | Document author |

### RetrievalContext

Retrieved documents with formatted context.

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original query |
| `retrieved_sources` | `List[SourceCitation]` | Retrieved documents |
| `total_sources` | `int` | Sources before filtering |
| `context_text` | `str` | Formatted context for LLM |
| `context_tokens` | `Optional[int]` | Estimated token count |
| `retrieval_time_ms` | `Optional[float]` | Retrieval duration |
| `reranked` | `bool` | Whether results were reranked |
| `namespace` | `Optional[str]` | Vector store namespace used |
| `created_at` | `datetime` | Context creation timestamp |

**Properties**:

- `source_count` - Number of retrieved sources
- `average_score` - Average similarity score
- `max_score` - Maximum similarity score

### RAGConfig

Configuration for RAG operations.

| Field | Type | Description |
|-------|------|-------------|
| `default_k` | `int` | Default documents to retrieve (5) |
| `max_k` | `int` | Maximum documents allowed (20) |
| `default_similarity_threshold` | `float` | Default minimum score (0.0) |
| `max_context_tokens` | `int` | Maximum context tokens (4000) |
| `context_template` | `str` | Context prompt template |
| `source_separator` | `str` | Separator between sources |
| `include_source_metadata` | `bool` | Include metadata in context |
| `default_model` | `Optional[str]` | Default LLM model |
| `default_temperature` | `float` | Default temperature (0.1) |
| `default_max_tokens` | `int` | Default max tokens (500) |
| `system_prompt` | `str` | Default system prompt |
| `enable_query_expansion` | `bool` | Query expansion enabled |
| `enable_reranking` | `bool` | Result reranking enabled |
| `min_sources_for_response` | `int` | Minimum sources required (1) |
| `require_source_citation` | `bool` | Require source citations |
| `max_retrieval_time_ms` | `float` | Retrieval timeout (5000ms) |
| `max_generation_time_ms` | `float` | Generation timeout (10000ms) |
| `enable_caching` | `bool` | Enable response caching |
| `cache_ttl_seconds` | `int` | Cache TTL (3600s) |

### RAGMetrics

Performance and quality metrics.

| Field | Type | Description |
|-------|------|-------------|
| `retrieval_time_ms` | `float` | Retrieval duration |
| `generation_time_ms` | `float` | Generation duration |
| `total_time_ms` | `float` | Total pipeline duration |
| `sources_retrieved` | `int` | Documents retrieved |
| `sources_used` | `int` | Documents used in context |
| `average_similarity_score` | `float` | Average relevance score |
| `max_similarity_score` | `float` | Maximum relevance score |
| `tokens_generated` | `Optional[int]` | Tokens in response |
| `tokens_in_context` | `Optional[int]` | Tokens in context |
| `model_used` | `Optional[str]` | Model used |
| `has_answer` | `bool` | Answer was generated |
| `has_sources` | `bool` | Sources were found |
| `confidence_score` | `Optional[float]` | Model confidence |
| `query_length` | `int` | Query character count |
| `response_length` | `int` | Response character count |
| `timestamp` | `datetime` | Metrics timestamp |

**Properties**:

- `retrieval_success_rate` - Percentage of sources used
- `tokens_per_second` - Generation speed

## Best Practices

### 1. Choose Embedding Model Based on Requirements

Match embedding dimensions to your use case and vector store capacity.

```python
# GOOD - Small embeddings for large-scale deployment
compose.rag(
    llm_provider="openai",
    llm_api_key="sk-...",
    embedding_model="text-embedding-3-small",  # 1536 dimensions, faster
    vector_store_config={"dimension": 1536}
)

# GOOD - Large embeddings for higher quality
compose.rag(
    llm_provider="openai",
    llm_api_key="sk-...",
    embedding_model="text-embedding-3-large",  # 3072 dimensions, more accurate
    vector_store_config={"dimension": 3072}
)

# BAD - Dimension mismatch
compose.rag(
    embedding_model="text-embedding-3-large",  # 3072 dimensions
    vector_store_config={"dimension": 1536}    # BAD: Dimensions don't match
)
```

**Why**: Embedding dimensions must match vector store configuration. Smaller embeddings reduce storage and search costs; larger embeddings improve semantic precision.

### 2. Set Appropriate Similarity Thresholds

Filter low-quality retrievals with similarity thresholds based on your quality requirements.

```python
# GOOD - Threshold for factual Q&A
query = RAGQuery(
    query="What is the capital of France?",
    k=5,
    similarity_threshold=0.7,  # High threshold for factual accuracy
    temperature=0.2
)

# GOOD - Lower threshold for exploratory search
query = RAGQuery(
    query="Tell me about machine learning",
    k=10,
    similarity_threshold=0.5,  # Lower threshold for broader results
    temperature=0.5
)

# BAD - No threshold with broad query
query = RAGQuery(
    query="Tell me everything",
    k=20,
    # BAD: No threshold may include irrelevant results
)
```

**Why**: Similarity thresholds filter out irrelevant documents, improving response quality and reducing hallucination risk.

### 3. Chunk Documents Appropriately

Optimize chunk size for your use case and content type.

```python
from portico.ports.document_processor import DocumentProcessorConfig

# GOOD - Smaller chunks for technical docs
config = DocumentProcessorConfig(
    chunk_size=512,      # Smaller chunks for precise retrieval
    chunk_overlap=50,    # Overlap preserves context
    split_method="sentence"
)

# GOOD - Larger chunks for narrative content
config = DocumentProcessorConfig(
    chunk_size=1500,     # Larger chunks for stories/articles
    chunk_overlap=200,
    split_method="paragraph"
)

# BAD - No overlap, context loss
config = DocumentProcessorConfig(
    chunk_size=1000,
    chunk_overlap=0      # BAD: Information split across chunks may be lost
)
```

**Why**: Chunk size affects retrieval granularity. Smaller chunks enable precise retrieval but may lose context; larger chunks preserve context but reduce precision. Overlap prevents information loss at boundaries.

### 4. Use Metadata Filtering to Narrow Search Scope

Leverage metadata to filter documents before semantic search.

```python
# GOOD - Filtered search with metadata
query = RAGQuery(
    query="Product pricing information",
    k=5,
    metadata_filters={
        "category": "pricing",
        "version": "2024",
        "public": True
    }
)

response = await rag_service.query(query)

# BAD - Unfiltered search across all documents
query = RAGQuery(
    query="Product pricing information",
    k=5
    # BAD: May retrieve outdated or irrelevant versions
)
```

**Why**: Metadata filtering reduces search space, improves relevance, and speeds up retrieval by pre-filtering before semantic search.

### 5. Monitor and Tune RAG Performance

Track metrics to identify bottlenecks and optimize configuration.

```python
# GOOD - Monitor performance
from portico.kits.logging import get_logger

logger = get_logger(__name__)

async def monitored_rag_query(user_query: str) -> RAGResponse:
    query = RAGQuery(query=user_query, k=5)

    response = await rag_service.query(query)

    # Log performance metrics
    logger.info(
        "rag_query_completed",
        retrieval_time_ms=response.context.retrieval_time_ms if response.context else None,
        generation_time_ms=response.generation_time_ms,
        total_time_ms=response.total_time_ms,
        sources_count=response.source_count,
        tokens_used=response.tokens_used,
        relevance_score=response.relevance_score
    )

    # Alert on poor performance
    if response.total_time_ms and response.total_time_ms > 5000:
        logger.warning("slow_rag_query", query=user_query, time_ms=response.total_time_ms)

    # Alert on low relevance
    if response.relevance_score and response.relevance_score < 0.5:
        logger.warning("low_relevance_rag_query", query=user_query, score=response.relevance_score)

    return response

# BAD - No monitoring
response = await rag_service.query(RAGQuery(query=user_query))
return response  # No visibility into performance issues
```

**Why**: Performance monitoring identifies slow retrievals, low relevance scores, and high token usage, enabling continuous optimization.

### 6. Implement Graceful Fallbacks

Handle cases where insufficient relevant sources are found.

```python
# GOOD - Graceful handling of no sources
from portico.exceptions import RetrievalError

async def query_with_fallback(user_query: str) -> str:
    try:
        query = RAGQuery(query=user_query, k=5, similarity_threshold=0.6)
        response = await rag_service.query(query)

        # Check if sources meet quality threshold
        if response.source_count == 0:
            return "I don't have enough information in my knowledge base to answer that question. Could you rephrase or ask something else?"

        if response.relevance_score and response.relevance_score < 0.5:
            return f"{response.response}\n\n(Note: I found limited relevant information. This answer may not be comprehensive.)"

        return response.response

    except RetrievalError:
        return "I'm having trouble accessing my knowledge base. Please try again later."

# BAD - No fallback handling
response = await rag_service.query(RAGQuery(query=user_query))
return response.response  # May return hallucinated answer if no sources found
```

**Why**: Graceful fallbacks prevent hallucination when the knowledge base lacks relevant information.

### 7. Use Managed RAG for Production When Possible

Managed platforms handle infrastructure complexity and provide advanced features.

```python
# GOOD - Managed RAG for production
app = compose.webapp(
    kits=[
        compose.rag(
            use_managed_rag=True,
            managed_rag_provider="graphlit",
            managed_rag_config={
                "api_key": os.getenv("GRAPHLIT_API_KEY"),
                "platform_url": "https://api.graphlit.io/api/v1/graphql",
            },
        ),
    ],
)

# Features available with managed platform:
# - Automatic document processing (PDF, DOCX, audio, video)
# - Built-in conversation management
# - Automated feeds for continuous ingestion
# - Advanced reranking and filtering
# - No infrastructure management

# DIY RAG is better for:
# - Full control over components
# - Custom embeddings or vector stores
# - Cost optimization for high-volume use
# - Specific compliance requirements
```

**Why**: Managed platforms reduce operational overhead, provide production-grade infrastructure, and offer advanced features like multi-modal document processing.

## Security Considerations

### 1. API Key Protection

Never expose API keys in code or logs.

```python
# GOOD - API keys from environment
import os

app = compose.webapp(
    kits=[
        compose.rag(
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_api_key=os.getenv("OPENAI_API_KEY"),
            vector_store_config={
                "api_key": os.getenv("PINECONE_API_KEY")
            }
        ),
    ],
)

# BAD - Hardcoded API keys
app = compose.webapp(
    kits=[
        compose.rag(
            llm_api_key="sk-proj-abc123...",  # Exposed in code
        ),
    ],
)
```

### 2. Input Sanitization

Validate and sanitize user queries before processing.

```python
def sanitize_query(query: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove excessive whitespace
    query = " ".join(query.split())

    # Limit length
    max_length = 1000
    if len(query) > max_length:
        query = query[:max_length]

    # Remove potentially harmful patterns
    query = query.replace("<!--", "").replace("-->", "")
    query = query.replace("<script>", "").replace("</script>", "")

    return query

# GOOD - Sanitized input
user_query = sanitize_query(request.form["question"])
response = await rag_service.query(RAGQuery(query=user_query))

# BAD - Raw user input
response = await rag_service.query(RAGQuery(query=request.form["question"]))
```

### 3. Document Access Control

Implement access control for sensitive documents using metadata filtering.

```python
# GOOD - User-specific document filtering
async def query_for_user(user_id: str, query_text: str) -> RAGResponse:
    # Get user's access groups
    user_groups = await get_user_groups(user_id)

    # Filter documents by access control
    query = RAGQuery(
        query=query_text,
        k=5,
        metadata_filters={
            "access_groups": {"$in": user_groups},  # Only docs user can access
            "visibility": "private"
        }
    )

    return await rag_service.query(query)

# BAD - No access control
query = RAGQuery(query=query_text, k=5)
response = await rag_service.query(query)
# User may access documents they shouldn't see
```

### 4. Rate Limiting

Implement rate limiting to prevent abuse and control costs.

```python
from portico.exceptions import RateLimitError

rate_limiter = {}  # User ID -> (count, reset_time)

async def rate_limited_rag_query(user_id: str, query: str) -> RAGResponse:
    import time

    # Check rate limit
    current_time = time.time()
    if user_id in rate_limiter:
        count, reset_time = rate_limiter[user_id]
        if current_time < reset_time:
            if count >= 10:  # 10 queries per minute
                raise RateLimitError("RAG query rate limit exceeded")
            rate_limiter[user_id] = (count + 1, reset_time)
        else:
            rate_limiter[user_id] = (1, current_time + 60)
    else:
        rate_limiter[user_id] = (1, current_time + 60)

    # Execute query
    return await rag_service.query(RAGQuery(query=query))
```

## FAQs

### 1. DIY RAG vs Managed RAG: Which should I choose?

**Use DIY RAG when**:

- You need full control over components (custom embeddings, vector stores)
- Cost optimization is critical for high-volume usage
- You have specific compliance requirements (data residency, audit trails)
- You want to use specialized embedding models or vector stores

**Use Managed RAG when**:

- You want to minimize operational overhead
- You need multi-modal document processing (PDF, audio, video)
- You require production-grade infrastructure without setup
- You want advanced features like automated feeds and reranking

```python
# DIY RAG - Full control
compose.rag(
    llm_provider="openai",
    embedding_provider="openai",
    vector_store_type="pinecone",
    # Full control over each component
)

# Managed RAG - Simplified setup
compose.rag(
    use_managed_rag=True,
    managed_rag_provider="graphlit",
    # Platform handles all components
)
```

### 2. How do I optimize retrieval performance?

**Strategies**:

1. **Use metadata filtering** to reduce search space before semantic search
2. **Choose appropriate embedding dimensions** (smaller = faster, larger = more accurate)
3. **Tune chunk size** based on content type (smaller chunks for technical docs)
4. **Use namespaces** to partition vector stores by category
5. **Enable caching** for common queries

```python
# Optimized configuration
rag_config = RAGConfig(
    default_k=5,                     # Fewer documents = faster
    max_context_tokens=2000,         # Smaller context = faster generation
    enable_caching=True,             # Cache common queries
    cache_ttl_seconds=3600,
)

query = RAGQuery(
    query="...",
    k=5,
    similarity_threshold=0.7,        # Filter irrelevant results early
    namespace="product_docs",        # Search within category
    metadata_filters={"version": "2024"}  # Pre-filter with metadata
)
```

### 3. How do I handle documents in multiple languages?

Use multilingual embedding models and specify language in queries:

```python
# Use multilingual embedding model
compose.rag(
    llm_provider="openai",
    embedding_model="text-embedding-3-small",  # Supports 100+ languages
)

# Include language in metadata
document = Document(
    id="doc_fr_1",
    content="Le Python est un langage de programmation...",
    metadata={"language": "fr", "title": "Introduction à Python"}
)

# Filter by language
query = RAGQuery(
    query="Qu'est-ce que Python?",
    k=5,
    metadata_filters={"language": "fr"},  # Only French documents
    system_prompt="Réponds en français basé sur le contexte fourni."
)
```

### 4. How do I test RAG applications?

**Testing strategies**:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_rag_service():
    service = AsyncMock()
    service.query.return_value = RAGResponse(
        query="Test query",
        response="Test response",
        sources=[
            SourceCitation(
                id="source_1",
                content="Test content",
                score=0.9,
                title="Test Document"
            )
        ],
        tokens_used=100,
        total_time_ms=500
    )
    return service

@pytest.mark.asyncio
async def test_query_handler(mock_rag_service):
    result = await handle_user_question("What is Python?", mock_rag_service)

    assert result["answer"] == "Test response"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "Test Document"

# Test with real vector store (integration test)
@pytest.mark.asyncio
async def test_rag_integration():
    # Use memory vector store for testing
    service = create_test_rag_service()  # DIY with memory store

    # Index test documents
    await service.vector_store.add_documents(test_documents)

    # Query
    response = await service.query(RAGQuery(query="test question"))

    assert response.source_count > 0
    assert "test" in response.response.lower()
```

### 5. How do I handle long documents?

Long documents are automatically chunked during ingestion:

```python
from portico.ports.document_processor import DocumentProcessorConfig

# Configure chunking
config = DocumentProcessorConfig(
    chunk_size=1000,         # Characters per chunk
    chunk_overlap=100,       # Overlap between chunks
    split_method="sentence"  # Respect sentence boundaries
)

compose.rag(
    llm_provider="openai",
    document_processor_config=config
)

# Large document automatically chunked
large_document = Document(
    id="book_1",
    content=open("long_book.txt").read(),  # 100,000+ characters
    metadata={"title": "War and Peace"}
)

await rag_service.vector_store.add_documents([large_document])
# Automatically split into multiple chunks with overlap
```

### 6. How do I implement conversation memory with RAG?

Use managed RAG platforms for built-in conversation support, or implement manually:

```python
# Managed RAG (built-in conversations)
conversation_id = await rag_service.create_conversation(
    name="Support Chat",
    system_prompt="You are a helpful support assistant."
)

response1 = await rag_service.query(
    query="How do I reset my password?",
    conversation_id=conversation_id
)

response2 = await rag_service.query(
    query="What if I don't have access to my email?",
    conversation_id=conversation_id  # History maintained
)

# DIY RAG (manual conversation tracking)
conversation_history = []

async def query_with_history(user_query: str) -> str:
    # Retrieve relevant documents
    sources = await rag_service.retrieve_sources(query=user_query, k=3)

    # Build context with history
    context = "\n---\n".join([s.content for s in sources])

    # Include conversation history in system prompt
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in conversation_history[-5:]  # Last 5 messages
    ])

    response_text = await rag_service.generate_with_context(
        query=user_query,
        context=context,
        system_prompt=f"Previous conversation:\n{history_text}\n\nAnswer the new question using the provided context."
    )

    # Update history
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": response_text})

    return response_text
```

### 7. How do I update documents in the vector store?

**DIY RAG**: Delete and re-add documents

```python
# Update document
await rag_service.vector_store.delete_document(document_id="doc_1")

updated_document = Document(
    id="doc_1",
    content="Updated content...",
    metadata={"title": "Updated Document", "version": "2.0"}
)

await rag_service.vector_store.add_documents([updated_document])
```

**Managed RAG**: Use platform update methods

```python
# Update document metadata
await rag_service.update_document_metadata(
    document_id="doc_1",
    metadata={"version": "2.0", "reviewed": True}
)

# Or delete and re-ingest
await rag_service.delete_document(document_id="doc_1")
await rag_service.ingest_document(content=new_content, metadata=new_metadata)
```

### 8. How do I measure RAG quality?

**Metrics to track**:

```python
# Automatic metrics from response
response = await rag_service.query(RAGQuery(query=user_query))

metrics = {
    "retrieval_time_ms": response.context.retrieval_time_ms if response.context else None,
    "generation_time_ms": response.generation_time_ms,
    "average_relevance": response.relevance_score,
    "source_count": response.source_count,
    "tokens_used": response.tokens_used,
}

# Manual quality evaluation
def evaluate_response_quality(response: RAGResponse, expected_answer: str) -> dict:
    return {
        "answer_relevance": calculate_relevance(response.response, expected_answer),
        "source_attribution": response.source_count > 0,
        "factual_accuracy": verify_facts(response.response, response.sources),
        "completeness": len(response.response) > 50,
    }

# Track over time
await analytics.track_rag_metrics(user_id, query, response, quality_scores)
```

### 9. How do I implement hybrid search (keyword + semantic)?

Combine metadata filtering with semantic search:

```python
# Metadata filtering acts as keyword pre-filter
query = RAGQuery(
    query="Python decorators",
    k=10,
    metadata_filters={
        "keywords": {"$contains": "python"},  # Keyword filter
        "category": "tutorial"
    }
)

# Or use query expansion
rag_config = RAGConfig(
    enable_query_expansion=True,  # Expand query with synonyms
    enable_reranking=True         # Rerank results
)

response = await rag_service.query(query)
```

### 10. How do I handle real-time document updates?

**Use automated feeds** (managed RAG):

```python
# Create feed for continuous ingestion
feed_id = await rag_service.create_feed({
    "name": "Documentation Feed",
    "type": "RSS",
    "config": {
        "url": "https://docs.example.com/rss",
        "refresh_interval_minutes": 60
    }
})

# Platform automatically ingests new content
# Query always returns latest information

# Pause/resume feed
await rag_service.pause_feed(feed_id)
await rag_service.resume_feed(feed_id)
```

**Periodic re-indexing** (DIY RAG):

```python
import asyncio

async def refresh_documents():
    while True:
        # Fetch latest documents from source
        latest_docs = await fetch_latest_documents()

        # Re-index changed documents
        for doc in latest_docs:
            await rag_service.vector_store.delete_document(doc.id)
            await rag_service.vector_store.add_documents([doc])

        # Wait 1 hour
        await asyncio.sleep(3600)

# Run in background
asyncio.create_task(refresh_documents())
```

## Related Ports

- **LLM Port** - Generation backend for RAG responses
- **Embedding Port** - Text vectorization for semantic search
- **Vector Store Port** - Document storage and similarity search
- **Document Processor Port** - Document chunking and preprocessing
- **Managed RAG Port** - Platform adapters (Graphlit, Vectara)

## Architecture Notes

The RAG Kit is a **stateless kit** that orchestrates multiple components to implement retrieval-augmented generation. It supports two architectural patterns:

### DIY RAG Pattern

Components are instantiated separately and wired together:

- **Vector Store**: Stores document embeddings for similarity search
- **Embedding Provider**: Generates vector representations of text
- **Document Processor**: Chunks and preprocesses documents
- **LLM Service**: Generates responses with context

### Managed RAG Pattern

A single platform adapter handles all RAG operations:

- **Platform Adapter**: Implements ManagedRAGPlatform interface
- **Service Layer**: ManagedRAGService wraps adapter with validation and logging

**Key Architectural Decisions**:

- **Port-based abstractions**: Application depends on interfaces, not implementations
- **Composition root pattern**: Adapters instantiated only in `compose.py`
- **Stateless design**: No database models, uses vector stores and external platforms
- **Pipeline orchestration**: RAGService coordinates retrieval, context construction, and generation
- **Error isolation**: Each pipeline step has dedicated exception types (RetrievalError, GenerationError, ContextError)

The RAG Kit demonstrates hexagonal architecture by depending on ports for all external dependencies, enabling flexible component swapping (memory vs Pinecone vector stores, DIY vs managed platforms) without changing business logic.
