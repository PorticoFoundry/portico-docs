# Vector Store Port

## Overview

The Vector Store Port defines the contract for storing and searching vector embeddings in Portico applications.

**Purpose**: Abstract vector storage and similarity search operations to enable pluggable vector database backends for semantic search and retrieval.

**Domain**: Vector similarity search, semantic retrieval, RAG (Retrieval-Augmented Generation)

**Key Capabilities**:

- Document and document chunk storage with embeddings
- Vector similarity search with multiple algorithms (cosine, euclidean, dot product)
- Namespace-based multi-tenant isolation
- Metadata filtering for refined searches
- Similarity threshold filtering
- Direct similarity computation between vectors
- Batch indexing operations
- Statistics and monitoring

**Port Type**: Adapter

**When to Use**:

- RAG (Retrieval-Augmented Generation) systems
- Semantic search applications
- Document similarity and clustering
- Question-answering systems
- Knowledge base retrieval
- Content recommendation engines
- Duplicate detection

## Domain Models

### Document

Represents a document with text content, metadata, and optional embedding. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | No | `uuid4()` | Unique document identifier |
| `content` | `str` | Yes | - | Document text content |
| `metadata` | `Dict[str, Any]` | No | `{}` | Custom metadata for filtering |
| `embedding` | `Optional[List[float]]` | No | `None` | Vector embedding (required for indexing) |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

**Example**:

```python
from portico.ports.vector_store import Document

# Document with embedding
doc = Document(
    id="doc_123",
    content="Portico is a Python framework for building GPT-powered applications.",
    metadata={
        "source": "documentation",
        "category": "overview",
        "version": "1.0"
    },
    embedding=[0.123, -0.456, 0.789, ...]  # 1536-dim vector
)
```

### DocumentChunk

Represents a chunked portion of a document with position metadata. Used for long documents that need to be split. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | No | `uuid4()` | Unique chunk identifier |
| `content` | `str` | Yes | - | Chunk text content |
| `metadata` | `Dict[str, Any]` | No | `{}` | Custom metadata for filtering |
| `embedding` | `Optional[List[float]]` | No | `None` | Vector embedding (required for indexing) |
| `document_id` | `str` | Yes | - | Parent document ID |
| `chunk_index` | `int` | Yes | - | Position index in parent document |
| `start_char` | `int` | Yes | - | Starting character position |
| `end_char` | `int` | Yes | - | Ending character position |
| `created_at` | `datetime` | No | Current UTC time | Creation timestamp |

**Example**:

```python
from portico.ports.vector_store import DocumentChunk

chunk = DocumentChunk(
    id="chunk_1",
    content="Portico uses hexagonal architecture with ports and adapters.",
    metadata={"source": "documentation", "category": "architecture"},
    embedding=[0.234, -0.567, 0.891, ...],
    document_id="doc_123",
    chunk_index=0,
    start_char=0,
    end_char=65
)
```

### SearchQuery

Query parameters for vector similarity search. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `vector` | `Optional[List[float]]` | No* | `None` | Query vector for similarity search |
| `text` | `Optional[str]` | No* | `None` | Query text (must be embedded first) |
| `k` | `int` | No | `10` | Number of results to return |
| `threshold` | `Optional[float]` | No | `None` | Minimum similarity threshold (0.0-1.0) |
| `filters` | `Dict[str, Any]` | No | `{}` | Metadata filters to apply |
| `namespace` | `Optional[str]` | No | `None` | Namespace for multi-tenant isolation |
| `include_metadata` | `bool` | No | `True` | Include metadata in results |
| `include_embeddings` | `bool` | No | `False` | Include embeddings in results |

**Validation**: Either `vector` or `text` must be provided (but not both).

**Example**:

```python
from portico.ports.vector_store import SearchQuery

# Search with vector
query = SearchQuery(
    vector=[0.123, -0.456, 0.789, ...],  # Pre-embedded query
    k=5,
    threshold=0.7,
    filters={"category": "documentation"},
    namespace="prod"
)

# Search with text (requires RAG service to embed)
query = SearchQuery(
    text="How do I use Portico?",
    k=10,
    namespace="prod"
)
```

### SearchResult

Search result with document/chunk and similarity score. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `document` | `Document \| DocumentChunk` | Yes | - | Retrieved document or chunk |
| `score` | `float` | Yes | - | Similarity score (0.0-1.0, higher is better) |

**Example**:

```python
from portico.ports.vector_store import SearchResult

results = await vector_store.search(query)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.document.content[:100]}")
    print(f"Metadata: {result.document.metadata}")
```

### SimilarityRequest

Request for computing similarity between vectors or texts. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query_vector` | `Optional[List[float]]` | No* | `None` | Query vector |
| `query_text` | `Optional[str]` | No* | `None` | Query text (must be embedded) |
| `target_vectors` | `Optional[List[List[float]]]` | No* | `None` | Target vectors to compare |
| `target_texts` | `Optional[List[str]]` | No* | `None` | Target texts (must be embedded) |
| `method` | `str` | No | `"cosine"` | Similarity method (cosine, euclidean, dot_product) |

**Validation**: Either `query_vector` or `query_text` must be provided. Either `target_vectors` or `target_texts` must be provided.

**Example**:

```python
from portico.ports.vector_store import SimilarityRequest

request = SimilarityRequest(
    query_vector=[0.1, 0.2, 0.3],
    target_vectors=[
        [0.15, 0.25, 0.35],
        [0.5, 0.6, 0.7],
        [-0.1, -0.2, -0.3]
    ],
    method="cosine"
)

response = await vector_store.compute_similarity(request)
print(response.scores)  # [0.998, 0.845, -0.123]
```

### SimilarityResponse

Response from similarity computation. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `scores` | `List[float]` | Yes | - | Similarity scores for each target |
| `method` | `str` | Yes | - | Similarity method used |

### VectorStoreStats

Statistics about the vector store. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `total_documents` | `int` | Yes | - | Total number of documents indexed |
| `total_chunks` | `int` | Yes | - | Total number of chunks indexed |
| `namespaces` | `List[str]` | Yes | - | List of namespaces in use |
| `average_embedding_dimension` | `Optional[int]` | No | `None` | Average embedding vector dimension |
| `storage_size_bytes` | `Optional[int]` | No | `None` | Storage size (if available) |

**Example**:

```python
stats = await vector_store.get_stats()

print(f"Documents: {stats.total_documents}")
print(f"Chunks: {stats.total_chunks}")
print(f"Namespaces: {stats.namespaces}")
print(f"Avg dimension: {stats.average_embedding_dimension}")
```

### VectorStoreConfig

Configuration for vector store operations. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default_namespace` | `str` | No | `"default"` | Default namespace for operations |
| `max_documents_per_namespace` | `Optional[int]` | No | `None` | Max documents per namespace (None = unlimited) |
| `default_search_k` | `int` | No | `10` | Default number of search results |
| `default_similarity_threshold` | `float` | No | `0.0` | Default minimum similarity threshold |
| `similarity_method` | `str` | No | `"cosine"` | Default similarity method |
| `batch_size` | `int` | No | `100` | Batch size for bulk operations |
| `enable_metadata_indexing` | `bool` | No | `True` | Enable metadata indexing for faster filtering |

## Port Interfaces

### VectorStore

The `VectorStore` abstract base class defines the contract for all vector storage backends.

**Location**: `portico.ports.vector_store.VectorStore`

#### Key Methods

##### index_document

```python
async def index_document(document: Document, namespace: Optional[str] = None) -> str
```

Index a document in the vector store. Primary method for storing documents with embeddings.

**Parameters**:

- `document`: Document to index (must have `embedding` set)
- `namespace`: Optional namespace for multi-tenant isolation

**Returns**: Document ID.

**Raises**: ValueError if document does not have an embedding.

**Example**:

```python
from portico.ports.vector_store import Document

# Create document with embedding
embedding = await embedding_provider.embed_text("Your document text here")
doc = Document(
    content="Your document text here",
    metadata={"source": "api", "author": "user_123"},
    embedding=embedding
)

# Index the document
doc_id = await vector_store.index_document(doc, namespace="prod")
print(f"Indexed document: {doc_id}")
```

##### search

```python
async def search(query: SearchQuery) -> List[SearchResult]
```

Search for similar documents/chunks. Primary method for semantic search and retrieval.

**Parameters**:

- `query`: Search query with vector and parameters

**Returns**: List of search results sorted by similarity score (highest first), limited to `query.k` results.

**Raises**: ValueError if query does not have a vector (text queries require embedding first).

**Example**:

```python
from portico.ports.vector_store import SearchQuery

# Generate query embedding
query_embedding = await embedding_provider.embed_text("How do I install Portico?")

# Search vector store
query = SearchQuery(
    vector=query_embedding,
    k=5,
    threshold=0.7,
    filters={"category": "documentation"},
    namespace="prod"
)

results = await vector_store.search(query)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.document.content[:100]}")
    print(f"Metadata: {result.document.metadata}")
    print("---")

# Used in RAG pipeline
async def retrieve_context(user_query: str) -> str:
    embedding = await embedding_provider.embed_text(user_query)
    results = await vector_store.search(SearchQuery(vector=embedding, k=3))
    return "\n\n".join([r.document.content for r in results])
```

#### Other Methods

##### index_documents

```python
async def index_documents(documents: List[Document], namespace: Optional[str] = None) -> List[str]
```

Index multiple documents in batch. Returns list of document IDs.

##### index_chunk

```python
async def index_chunk(chunk: DocumentChunk, namespace: Optional[str] = None) -> str
```

Index a document chunk. Returns chunk ID.

##### index_chunks

```python
async def index_chunks(chunks: List[DocumentChunk], namespace: Optional[str] = None) -> List[str]
```

Index multiple document chunks in batch. Returns list of chunk IDs.

##### get_document

```python
async def get_document(document_id: str, namespace: Optional[str] = None) -> Optional[Document]
```

Retrieve a document by ID. Returns document if found, None otherwise.

##### get_chunk

```python
async def get_chunk(chunk_id: str, namespace: Optional[str] = None) -> Optional[DocumentChunk]
```

Retrieve a document chunk by ID. Returns chunk if found, None otherwise.

##### delete_document

```python
async def delete_document(document_id: str, namespace: Optional[str] = None) -> bool
```

Delete a document from the vector store. Returns True if deleted, False if not found.

##### delete_chunk

```python
async def delete_chunk(chunk_id: str, namespace: Optional[str] = None) -> bool
```

Delete a document chunk from the vector store. Returns True if deleted, False if not found.

##### delete_by_metadata

```python
async def delete_by_metadata(filters: Dict[str, Any], namespace: Optional[str] = None) -> int
```

Delete documents/chunks by metadata filters. Returns number of items deleted.

##### clear_namespace

```python
async def clear_namespace(namespace: str) -> int
```

Clear all documents/chunks in a namespace. Returns number of items deleted.

##### clear_all

```python
async def clear_all() -> int
```

Clear all documents/chunks from the vector store. Returns number of items deleted.

##### compute_similarity

```python
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse
```

Compute similarity between query and target vectors. Returns similarity scores for each target.

##### get_stats

```python
async def get_stats(namespace: Optional[str] = None) -> VectorStoreStats
```

Get statistics about the vector store. If namespace provided, returns stats for that namespace only.

##### list_namespaces

```python
async def list_namespaces() -> List[str]
```

List all namespaces in the vector store. Returns list of namespace names.

## Common Patterns

### RAG Pipeline with Vector Search

```python
from portico.ports.embedding import EmbeddingProvider
from portico.ports.vector_store import VectorStore, SearchQuery, Document
from portico.ports.llm import Message, MessageRole

async def rag_query(
    user_query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_service: LLMService,
    k: int = 3
) -> str:
    """Complete RAG pipeline: retrieve context and generate response."""

    # Step 1: Embed the query
    query_embedding = await embedding_provider.embed_text(user_query)

    # Step 2: Search for relevant documents
    search_query = SearchQuery(
        vector=query_embedding,
        k=k,
        threshold=0.6,
        namespace="knowledge_base"
    )
    results = await vector_store.search(search_query)

    # Step 3: Build context from top results
    context = "\n\n".join([
        f"[Source {i+1}]: {r.document.content}"
        for i, r in enumerate(results)
    ])

    # Step 4: Generate response with context
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=f"Answer using this context:\n\n{context}"
        ),
        Message(
            role=MessageRole.USER,
            content=user_query
        )
    ]

    response = await llm_service.chat(messages)
    return response.content

# Usage
answer = await rag_query(
    "How do I use Portico for RAG?",
    embedding_provider=app.kits["rag"].embedding_provider,
    vector_store=app.kits["rag"].vector_store,
    llm_service=app.kits["llm"].service
)
```

### Document Chunking and Indexing

```python
from portico.ports.vector_store import Document, DocumentChunk
from portico.ports.embedding import EmbeddingProvider

async def chunk_and_index_document(
    content: str,
    metadata: dict,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """Split long document into chunks and index them."""

    # Generate document ID
    doc_id = str(uuid4())

    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size
        chunk_content = content[start:end]

        # Create chunk
        chunk = DocumentChunk(
            content=chunk_content,
            metadata=metadata,
            document_id=doc_id,
            chunk_index=len(chunks),
            start_char=start,
            end_char=end
        )

        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    # Embed all chunks
    chunk_texts = [c.content for c in chunks]
    embeddings = await embedding_provider.embed_texts(chunk_texts)

    # Add embeddings to chunks
    chunks_with_embeddings = [
        DocumentChunk(**{**chunk.model_dump(), "embedding": emb})
        for chunk, emb in zip(chunks, embeddings)
    ]

    # Index all chunks
    chunk_ids = await vector_store.index_chunks(
        chunks_with_embeddings,
        namespace="documents"
    )

    logger.info(
        "document_indexed",
        doc_id=doc_id,
        chunks=len(chunk_ids),
        total_chars=len(content)
    )

    return doc_id, chunk_ids
```

## Integration with Kits

The Vector Store Port is used by the **RAG Kit** to provide document storage and semantic search.

```python
from portico import compose

# Configure RAG kit with memory vector store (development)
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            vector_store_type="memory",  # In-memory (requires numpy)
        ),
    ],
)

# Configure with Pinecone (production)
app = compose.webapp(
    database_url="postgresql+asyncpg://user:pass@localhost/db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            vector_store_type="pinecone",
            vector_store_config={
                "api_key": "pc-...",
                "index_name": "my-app-vectors",
                "dimension": 1536,
                "cloud": "aws",
                "region": "us-east-1",
                "metric": "cosine"
            }
        ),
    ],
)

await app.initialize()

# Access vector store through RAG kit
vector_store = app.kits["rag"].vector_store

# Index a document
embedding = await app.kits["rag"].embedding_provider.embed_text("Document content")
doc = Document(content="Document content", embedding=embedding)
doc_id = await vector_store.index_document(doc, namespace="prod")

# Search
query_emb = await app.kits["rag"].embedding_provider.embed_text("search query")
results = await vector_store.search(SearchQuery(vector=query_emb, k=5))

# Get statistics
stats = await vector_store.get_stats()
print(f"Total documents: {stats.total_documents}")
```

The RAG Kit provides:

- Memory vector store adapter (in-memory with numpy)
- Pinecone vector store adapter (cloud-based)
- Integration with embedding providers
- Document chunking utilities
- Managed RAG service combining retrieval + generation

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Always Embed Before Indexing**: Documents must have embeddings before indexing

   ```python
   # ✅ GOOD: Embed first
   embedding = await embedding_provider.embed_text(content)
   doc = Document(content=content, embedding=embedding)
   await vector_store.index_document(doc)

   # ❌ BAD: No embedding
   doc = Document(content=content)
   await vector_store.index_document(doc)  # Raises ValueError!
   ```

2. **Use Namespaces for Multi-Tenancy**: Isolate data by user, project, or environment

   ```python
   # ✅ GOOD: Namespace isolation
   await vector_store.index_document(doc, namespace=f"user:{user_id}")
   results = await vector_store.search(
       SearchQuery(vector=query_emb, k=5, namespace=f"user:{user_id}")
   )

   # ❌ BAD: All users share same namespace
   await vector_store.index_document(doc)  # Default namespace
   # Risk: Users can search other users' documents!
   ```

3. **Set Similarity Thresholds**: Filter low-quality results with threshold

   ```python
   # ✅ GOOD: Filter low-relevance results
   query = SearchQuery(
       vector=query_emb,
       k=10,
       threshold=0.7  # Only results with >0.7 similarity
   )

   # ❌ BAD: Return all results regardless of relevance
   query = SearchQuery(vector=query_emb, k=10)
   # May return completely irrelevant results
   ```

4. **Use Metadata Filters for Precision**: Combine vector search with metadata filtering

   ```python
   # ✅ GOOD: Combine vector similarity + metadata filters
   query = SearchQuery(
       vector=query_emb,
       k=5,
       filters={
           "category": "documentation",
           "version": "2.0",
           "language": "en"
       }
   )

   # ❌ BAD: Filter in application code (inefficient)
   all_results = await vector_store.search(SearchQuery(vector=query_emb, k=100))
   filtered = [r for r in all_results if r.document.metadata.get("category") == "documentation"]
   ```

5. **Batch Index for Performance**: Use batch operations for multiple documents

   ```python
   # ✅ GOOD: Batch indexing
   embeddings = await embedding_provider.embed_texts([d.content for d in docs])
   docs_with_embeddings = [
       Document(**{**d.model_dump(), "embedding": emb})
       for d, emb in zip(docs, embeddings)
   ]
   doc_ids = await vector_store.index_documents(docs_with_embeddings)

   # ❌ BAD: Individual indexing (slow)
   for doc in docs:
       embedding = await embedding_provider.embed_text(doc.content)
       doc_with_emb = Document(**{**doc.model_dump(), "embedding": embedding})
       await vector_store.index_document(doc_with_emb)
   ```

## FAQs

### What similarity methods are supported?

Portico supports three similarity methods:

- **cosine** (default): Measures angle between vectors, range -1 to 1 (normalized to 0-1). Best for most use cases.
- **euclidean**: Measures distance between vectors, converted to similarity via exponential decay. Good for spatial relationships.
- **dot_product**: Raw dot product of vectors. Useful when vectors are normalized.

**Recommendation**: Use cosine similarity for most semantic search applications.

```python
config = VectorStoreConfig(similarity_method="cosine")
```

### How do I choose between Documents and DocumentChunks?

- **Use Documents** for short content that fits within embedding model token limits (~8000 tokens for OpenAI)
- **Use DocumentChunks** for long documents that need to be split into smaller pieces

**Rule of thumb**: If document is >500 words, use chunking.

```python
# Short content - use Document
doc = Document(content="Brief FAQ answer", embedding=embedding)

# Long content - use DocumentChunk
chunks = split_into_chunks(long_document, chunk_size=500)
for i, chunk in enumerate(chunks):
    chunk_obj = DocumentChunk(
        content=chunk,
        document_id=doc_id,
        chunk_index=i,
        start_char=i*500,
        end_char=(i+1)*500,
        embedding=await embed(chunk)
    )
    await vector_store.index_chunk(chunk_obj)
```

### Can I search without providing a vector?

No, the `search()` method requires a vector. If you have text, embed it first:

```python
# ❌ WRONG: SearchQuery with text only
query = SearchQuery(text="my query", k=5)
results = await vector_store.search(query)  # Raises ValueError

# ✅ CORRECT: Embed text first
embedding = await embedding_provider.embed_text("my query")
query = SearchQuery(vector=embedding, k=5)
results = await vector_store.search(query)
```

**Note**: The RAG Kit's higher-level services handle embedding automatically.

### What happens if I use different embedding dimensions?

**You must use consistent embedding dimensions throughout your vector store.** Mixing dimensions will cause errors during similarity computation.

```python
# ❌ WRONG: Inconsistent dimensions
doc1 = Document(content="Text 1", embedding=[0.1]*1536)  # 1536 dims
doc2 = Document(content="Text 2", embedding=[0.2]*3072)  # 3072 dims - ERROR!

# ✅ CORRECT: Consistent dimensions
config = VectorStoreConfig(dimensions=1536)
# All documents use 1536-dimensional embeddings
```

When switching embedding models, you must:
1. Clear the vector store
2. Re-embed all documents with the new model
3. Re-index everything

### How do I implement a custom vector store adapter?

Implement the `VectorStore` interface:

```python
from portico.ports.vector_store import VectorStore, Document, SearchQuery, SearchResult

class CustomVectorStore(VectorStore):
    async def index_document(
        self,
        document: Document,
        namespace: Optional[str] = None
    ) -> str:
        # Store document in your vector database
        await your_db.insert(document.id, document.embedding, document.metadata)
        return document.id

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        # Perform similarity search
        results = await your_db.similarity_search(
            query.vector,
            limit=query.k,
            filters=query.filters
        )
        return [SearchResult(document=r.doc, score=r.score) for r in results]

    # Implement all other abstract methods...
```

Then use it in composition:

```python
def rag(**config):
    from your_module import CustomVectorStore
    from portico.kits.rag import RAGKit

    def factory(database, events):
        vector_store = CustomVectorStore(config["vector_db_url"])
        return RAGKit.create(database, events, config, vector_store=vector_store)

    return factory
```
