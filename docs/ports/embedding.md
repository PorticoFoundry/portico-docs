# Embedding Port

## Overview

The Embedding Port defines the contract for converting text into vector embeddings in Portico applications.

**Purpose**: Abstract text embedding operations to enable pluggable embedding providers (OpenAI, Anthropic, local models, etc.).

**Domain**: Natural language processing, semantic search, vector similarity

**Key Capabilities**:

- Single and batch text embedding generation
- Multiple embedding model support
- Configurable embedding dimensions
- Token usage tracking
- Batch processing with configurable size limits
- Model dimension introspection
- Provider-agnostic interface for RAG and semantic search

**Port Type**: Adapter

**When to Use**:

- Applications requiring semantic search or similarity matching
- RAG (Retrieval-Augmented Generation) systems
- Document clustering and categorization
- Question-answering systems
- Recommendation engines based on text similarity
- Duplicate detection and content matching

## Domain Models

### EmbeddingRequest

Request for generating text embeddings. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | `str \| List[str]` | Yes | - | Single text or list of texts to embed |
| `model` | `Optional[str]` | No | `None` | Model to use (defaults to provider default) |
| `encoding_format` | `str` | No | `"float"` | Encoding format ("float" or "base64") |
| `dimensions` | `Optional[int]` | No | `None` | Embedding dimensions (for models supporting variable dims) |
| `user` | `Optional[str]` | No | `None` | Optional user identifier for tracking |

**Example**:

```python
from portico.ports.embedding import EmbeddingRequest

# Single text
request = EmbeddingRequest(texts="Hello, world!")

# Multiple texts
request = EmbeddingRequest(
    texts=["Document 1", "Document 2", "Document 3"],
    model="text-embedding-3-small",
    dimensions=1536
)

# With user tracking
request = EmbeddingRequest(
    texts="User query",
    user="user_12345"
)
```

### EmbeddingData

Individual embedding result within a response. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `embedding` | `List[float]` | Yes | - | Embedding vector (list of floats) |
| `index` | `int` | Yes | - | Index of this embedding in the batch |
| `object` | `str` | No | `"embedding"` | Object type identifier |

**Example**:

```python
from portico.ports.embedding import EmbeddingData

data = EmbeddingData(
    embedding=[0.123, -0.456, 0.789, ...],  # 1536 dimensions
    index=0
)

print(f"Vector length: {len(data.embedding)}")  # 1536
```

### EmbeddingUsage

Token usage information for embedding generation. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt_tokens` | `int` | Yes | - | Number of tokens in input texts |
| `total_tokens` | `int` | Yes | - | Total tokens consumed |

**Example**:

```python
from portico.ports.embedding import EmbeddingUsage

usage = EmbeddingUsage(
    prompt_tokens=100,
    total_tokens=100
)
```

### EmbeddingResponse

Response from embedding generation with metadata. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `str` | No | `uuid4()` | Unique response identifier |
| `object` | `str` | No | `"list"` | Object type identifier |
| `data` | `List[EmbeddingData]` | Yes | - | List of embedding results |
| `model` | `str` | Yes | - | Model used for generation |
| `usage` | `Optional[EmbeddingUsage]` | No | `None` | Token usage information |
| `created_at` | `datetime` | No | Current UTC time | Response creation timestamp |

**Properties**:

- `embeddings -> List[List[float]]` - Convenience property returning just the embedding vectors
- `single_embedding -> List[float]` - Returns single embedding vector (raises ValueError if multiple embeddings)

**Example**:

```python
from portico.ports.embedding import EmbeddingResponse

response = await embedding_provider.embed(request)

print(f"Model: {response.model}")
print(f"Embeddings count: {len(response.data)}")
print(f"Tokens used: {response.usage.total_tokens}")

# Get all embeddings
vectors = response.embeddings  # List[List[float]]

# For single embedding request
single_vector = response.single_embedding  # List[float]
print(f"Dimension: {len(single_vector)}")
```

### EmbeddingConfig

Configuration for embedding operations. Immutable.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default_model` | `str` | No | `"text-embedding-3-small"` | Default embedding model |
| `batch_size` | `int` | No | `100` | Maximum texts to process in one request |
| `max_retries` | `int` | No | `3` | Maximum retry attempts on failure |
| `timeout_seconds` | `float` | No | `30.0` | Request timeout in seconds |
| `dimensions` | `Optional[int]` | No | `None` | Embedding dimensions (None = model default) |

**Example**:

```python
from portico.ports.embedding import EmbeddingConfig

config = EmbeddingConfig(
    default_model="text-embedding-3-large",
    batch_size=50,
    max_retries=5,
    timeout_seconds=60.0,
    dimensions=3072
)
```

## Port Interfaces

### EmbeddingProvider

The `EmbeddingProvider` abstract base class defines the contract for all embedding providers.

**Location**: `portico.ports.embedding.EmbeddingProvider`

#### Key Methods

##### embed

```python
async def embed(request: EmbeddingRequest) -> EmbeddingResponse
```

Generate embeddings for the given texts. Primary method for embedding generation with full control.

**Parameters**:

- `request`: Embedding request with texts and configuration

**Returns**: EmbeddingResponse containing generated embeddings and metadata.

**Example**:

```python
from portico.ports.embedding import EmbeddingRequest

# Embed single text
request = EmbeddingRequest(texts="What is semantic search?")
response = await embedding_provider.embed(request)

vector = response.single_embedding
print(f"Embedding dimension: {len(vector)}")
print(f"Tokens used: {response.usage.total_tokens}")

# Embed multiple texts (batch)
request = EmbeddingRequest(
    texts=[
        "First document about cats",
        "Second document about dogs",
        "Third document about pets"
    ],
    model="text-embedding-3-small"
)
response = await embedding_provider.embed(request)

for i, embedding in enumerate(response.embeddings):
    print(f"Document {i}: {len(embedding)} dimensions")
```

##### embed_text

```python
async def embed_text(text: str, model: Optional[str] = None, **kwargs: Any) -> List[float]
```

Convenience method to embed a single text and return just the vector. Simpler interface for common use cases.

**Parameters**:

- `text`: Text to embed
- `model`: Model to use, defaults to configured default
- `**kwargs`: Additional provider-specific options

**Returns**: Embedding vector as a list of floats.

**Example**:

```python
# Simple single embedding
vector = await embedding_provider.embed_text("Hello, world!")
print(f"Dimension: {len(vector)}")  # e.g., 1536

# With specific model
vector = await embedding_provider.embed_text(
    "Semantic search query",
    model="text-embedding-3-large"
)

# Use in similarity search
query_vector = await embedding_provider.embed_text(user_query)
similar_docs = await vector_store.search(query_vector, top_k=10)
```

#### Other Methods

##### embed_texts

```python
async def embed_texts(
    texts: List[str],
    model: Optional[str] = None,
    **kwargs: Any
) -> List[List[float]]
```

Convenience method to embed multiple texts and return just the vectors. Returns list of embedding vectors, one for each input text.

##### get_embedding_dimension

```python
async def get_embedding_dimension(model: Optional[str] = None) -> int
```

Get the dimension of embeddings produced by the specified model. Returns dimension (length) of embedding vectors produced by the model.

##### list_models

```python
async def list_models() -> List[str]
```

List available embedding models from the provider.

## Common Patterns

### Semantic Search with Embeddings

```python
from portico.ports.embedding import EmbeddingProvider
from portico.ports.vector_store import VectorStore

async def semantic_search(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 5
) -> list:
    """Perform semantic search using embeddings."""

    # 1. Embed the query
    query_vector = await embedding_provider.embed_text(query)

    # 2. Search vector store
    results = await vector_store.search(
        vector=query_vector,
        top_k=top_k,
        namespace="documents"
    )

    # 3. Return matches
    return [
        {
            "text": result.metadata.get("text"),
            "score": result.score,
            "document_id": result.id
        }
        for result in results
    ]

# Usage
results = await semantic_search(
    query="How do I install Portico?",
    embedding_provider=app.kits["rag"].embedding_provider,
    vector_store=app.kits["rag"].vector_store,
    top_k=10
)

for result in results:
    print(f"Score: {result['score']:.3f} - {result['text'][:100]}")
```

### Batch Document Processing

```python
from portico.ports.embedding import EmbeddingProvider

async def embed_documents_in_batches(
    documents: list[str],
    embedding_provider: EmbeddingProvider,
    batch_size: int = 100
) -> list[list[float]]:
    """Embed large document collections in batches."""

    all_embeddings = []

    # Process in batches to avoid rate limits
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        # Embed batch
        embeddings = await embedding_provider.embed_texts(batch)
        all_embeddings.extend(embeddings)

        logger.info(
            "batch_embedded",
            batch_num=i // batch_size + 1,
            batch_size=len(batch),
            total_processed=len(all_embeddings)
        )

    return all_embeddings

# Usage
documents = load_document_corpus()  # 10,000 documents
embeddings = await embed_documents_in_batches(
    documents,
    embedding_provider,
    batch_size=100
)

# Store in vector database
await store_embeddings(documents, embeddings)
```

## Integration with Kits

The Embedding Port is used by the **RAG Kit** to provide text embedding services.

```python
from portico import compose

# Configure RAG kit with OpenAI embeddings
app = compose.webapp(
    database_url="sqlite+aiosqlite:///./app.db",
    kits=[
        compose.rag(
            llm_provider="openai",
            llm_api_key="sk-...",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            vector_store_type="memory"
        ),
    ],
)

await app.initialize()

# Access embedding provider through RAG kit
embedding_provider = app.kits["rag"].embedding_provider

# Single text embedding
vector = await embedding_provider.embed_text("Hello, world!")
print(f"Embedding dimension: {len(vector)}")  # 1536

# Batch embedding
vectors = await embedding_provider.embed_texts([
    "First document",
    "Second document",
    "Third document"
])
print(f"Generated {len(vectors)} embeddings")

# Get model info
dimension = await embedding_provider.get_embedding_dimension()
models = await embedding_provider.list_models()
print(f"Available models: {models}")
```

The RAG Kit provides:

- OpenAI embedding provider adapter
- Automatic batching for large document sets
- Rate limiting and retry logic
- Integration with vector stores for semantic search
- Document chunking and embedding pipeline

See the [Kits Overview](../kits/index.md) for more information about using kits.

## Best Practices

1. **Use Batch Processing for Multiple Texts**: Batch embedding is more efficient than multiple single calls

   ```python
   # ✅ GOOD: Batch processing
   vectors = await embedding_provider.embed_texts([
       "Text 1", "Text 2", "Text 3", "Text 4", "Text 5"
   ])

   # ❌ BAD: Individual calls (slower, more expensive)
   vectors = []
   for text in texts:
       vector = await embedding_provider.embed_text(text)
       vectors.append(vector)
   ```

2. **Match Embedding Dimensions to Vector Store**: Ensure embedding dimension matches vector store configuration

   ```python
   # ✅ GOOD: Consistent dimensions
   compose.rag(
       embedding_model="text-embedding-3-small",  # 1536 dims
       vector_store_config={
           "dimension": 1536,  # Matches embedding model
           ...
       }
   )

   # ❌ BAD: Mismatched dimensions
   # embedding_model: 1536 dims, vector_store: 3072 dims
   # Will cause errors!
   ```

3. **Cache Embeddings for Static Content**: Avoid re-embedding the same text repeatedly

   ```python
   # ✅ GOOD: Cache embeddings
   cached_embedding = await cache.get(f"embedding:{text_hash}")
   if not cached_embedding:
       cached_embedding = await embedding_provider.embed_text(text)
       await cache.set(f"embedding:{text_hash}", cached_embedding, ttl=timedelta(days=7))

   # ❌ BAD: Re-embed same text every time
   vector = await embedding_provider.embed_text(static_faq_text)
   ```

4. **Handle Rate Limits with Retries**: Embedding APIs have rate limits; use retry logic

   ```python
   # ✅ GOOD: Built-in retry configuration
   config = EmbeddingConfig(
       default_model="text-embedding-3-small",
       max_retries=5,
       timeout_seconds=60.0
   )

   # ❌ BAD: No retry logic
   # Single failure will crash the entire batch
   ```

5. **Use Appropriate Models for Your Use Case**: Larger models provide better quality but cost more

   ```python
   # ✅ GOOD: Match model to use case
   # text-embedding-3-small (1536 dims) - General search, lower cost
   # text-embedding-3-large (3072 dims) - High accuracy, higher cost

   # For general search
   compose.rag(embedding_model="text-embedding-3-small")

   # For specialized domains requiring high accuracy
   compose.rag(embedding_model="text-embedding-3-large")

   # ❌ BAD: Using large model when small would suffice
   # Wastes money and time
   ```

## FAQs

### What embedding models are available?

Portico includes an OpenAI embedding provider supporting:

- `text-embedding-3-small` - 1536 dimensions (default, cost-effective)
- `text-embedding-3-large` - 3072 dimensions (higher accuracy)
- `text-embedding-ada-002` - 1536 dimensions (legacy model)

You can check available models at runtime:

```python
models = await embedding_provider.list_models()
```

### How do I choose embedding dimensions?

Embedding dimensions affect quality, cost, and storage:

- **1536 dimensions** (text-embedding-3-small): Good balance, suitable for most use cases
- **3072 dimensions** (text-embedding-3-large): Better accuracy, 2x storage cost

**Rule of thumb**: Start with 1536, upgrade to 3072 if accuracy is insufficient.

### Can I use embeddings from different models together?

**No!** Embeddings from different models are not comparable. You cannot:

- Mix embeddings from `text-embedding-3-small` and `text-embedding-3-large` in the same vector store
- Compare similarity between vectors from different models

**Always use the same model for both indexing and querying.**

```python
# ❌ WRONG: Different models
# Index documents with model A
doc_embeddings = await provider.embed_texts(docs, model="text-embedding-3-small")

# Query with model B - WON'T WORK!
query_embedding = await provider.embed_text(query, model="text-embedding-3-large")
```

### How much does embedding cost?

OpenAI embedding costs (as of 2024):

- `text-embedding-3-small`: $0.02 / 1M tokens
- `text-embedding-3-large`: $0.13 / 1M tokens

A typical 500-word document ≈ 650 tokens. So:
- 1,000 documents ≈ 650K tokens ≈ $0.01 (small) or $0.08 (large)

Use `response.usage.total_tokens` to track consumption.

### How do I implement a custom embedding provider?

Implement the `EmbeddingProvider` interface:

```python
from portico.ports.embedding import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse

class CustomEmbeddingProvider(EmbeddingProvider):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        # Call your embedding service/model
        vectors = await your_model.encode(request.texts)

        # Return EmbeddingResponse
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=vec, index=i) for i, vec in enumerate(vectors)],
            model="your-model-name"
        )

    async def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        response = await self.embed(EmbeddingRequest(texts=text, model=model))
        return response.single_embedding

    # Implement other abstract methods...
```

Then use it in composition:

```python
def rag(**config):
    from your_module import CustomEmbeddingProvider
    from portico.kits.rag import RAGKit

    def factory(database, events):
        embedding_provider = CustomEmbeddingProvider(api_key=config["api_key"])
        return RAGKit.create(database, events, config, embedding_provider=embedding_provider)

    return factory
```
