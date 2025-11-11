# Adapters

This page provides a complete list of all adapters implemented in Portico, grouped by the port interface they support.

## LLM Port

Adapters for language model providers and prompt rendering.

### OpenAIProvider
OpenAI chat completion provider supporting GPT-4 and GPT-3.5 models with streaming, function calling, and response caching.

### AnthropicProvider
Anthropic (Claude) chat completion provider with support for Claude 3 models, streaming, and tool use capabilities.

### JinjaRenderer
Jinja2-based prompt template renderer supporting variable substitution, filters, and template inheritance.

### SimpleRenderer
Basic string template renderer using Python's string formatting for simple variable substitution without complex logic.

## Cache Port

Adapters for caching data with various backends.

### MemoryCacheAdapter
In-memory cache implementation using Python dictionaries with TTL support and LRU eviction for development and testing.

### RedisCacheAdapter
Redis-backed cache adapter providing distributed caching with persistence, TTL support, and high-performance access patterns.

### HybridCacheAdapter
Two-tier caching strategy combining in-memory L1 cache with Redis L2 cache for optimal performance and scalability.

## File Storage Port

Adapters for storing and retrieving files across different storage backends.

### LocalFileStorageAdapter
Local filesystem storage adapter for development and single-server deployments with configurable upload directories.

### GCSFileStorageAdapter
Google Cloud Storage adapter for production deployments with bucket management, signed URLs, and streaming uploads.

### DatabaseBlobStorageAdapter
Database-backed blob storage using chunked storage for simple deployments without external storage dependencies.

## Audit Port

Adapters for capturing and storing audit trails.

### SqlAlchemyAuditAdapter
Database-backed audit logging with queryable audit records, filtering by user/resource/action, and long-term retention.

### MemoryAuditAdapter
In-memory audit log storage for testing and development environments without database persistence.

### StructuredLoggingAuditAdapter
Structured logging-based audit trail using Python's logging framework for integration with log aggregation systems.

### CompositeAuditAdapter
Combines multiple audit adapters to write audit events to multiple destinations simultaneously (e.g., database + logging).

## Template Port

Adapters for template storage and rendering.

### Jinja2TemplateRenderer
Jinja2 template rendering engine with support for filters, macros, and template inheritance for complex template logic.

### MemoryTemplateRegistry
In-memory template storage for development and testing with support for template versioning and user ownership.

## Managed RAG Port

Adapters for managed Retrieval-Augmented Generation platforms.

### GraphlitPlatform
Graphlit managed RAG platform integration with document ingestion, semantic search, and conversation management.

## Vector Store Port

Adapters for vector similarity search and storage.

### MemoryVectorStore
In-memory vector store implementation using numpy for development and testing without external dependencies.

### PineconeVectorStore
Pinecone vector database adapter providing production-grade vector search with metadata filtering and namespace support.

## Embedding Port

Adapters for generating text embeddings.

### OpenAIEmbeddingProvider
OpenAI embedding generation using text-embedding-3-small, text-embedding-3-large, or ada-002 models with batch support.

## Document Processor Port

Adapters for processing and chunking documents.

### BasicDocumentProcessor
Basic document processor supporting text extraction from common formats (PDF, DOCX, TXT, MD) with metadata extraction.

### ContentTypeDetector
Content type detection utility using file extensions and magic bytes to identify document formats.

### FixedSizeChunker
Fixed-size text chunking strategy splitting documents into chunks of specified character length with overlap support.

### SentenceChunker
Sentence-aware text chunking that preserves sentence boundaries using natural language processing for better semantic coherence.

### ParagraphChunker
Paragraph-based text chunking that splits on double newlines to maintain document structure and context.

### MarkdownChunker
Markdown-aware chunking that respects heading hierarchy and code blocks for structured document processing.

## Job Queue Port

Adapters for background job queue management.

### DatabaseJobQueueAdapter
Database-backed job queue with persistent storage, priority support, and distributed worker coordination using polling.

### MemoryJobQueueAdapter
In-memory job queue for development and testing with support for job priorities and status tracking.

## Job Trigger Port

Adapters for creating jobs from external events.

### ScheduleTrigger
Schedule-based job trigger using APScheduler for cron-like job scheduling with configurable intervals and time zones.

### WebhookTrigger
Webhook-based job trigger for creating jobs from HTTP POST requests with payload validation and signature verification.

## Notification Port

Adapters for delivering notifications to users.

### BrowserNotificationAdapter
Database-backed browser notification adapter storing notifications for in-app display with read/unread tracking.

### MemoryNotificationAdapter
In-memory notification storage for development and testing without database persistence.

## Database and Repository Adapters

Adapters for database connections and domain repository implementations.

### PostgresAdapter
PostgreSQL database adapter with async SQLAlchemy support, connection pooling, and production-grade reliability.

### SqliteAdapter
SQLite database adapter for development and testing with async support via aiosqlite.

### SqlAlchemyUserRepository
Repository implementation for user management operations (create, read, update, delete) with password hashing.

### SqlAlchemyGroupRepository
Repository implementation for group management with hierarchical group support and membership tracking.

### SqlAlchemyTemplateRepository
Repository implementation for template storage with versioning, user ownership, and public/private visibility.

### SqlAlchemyConversationRepository
Repository implementation for conversation and message storage with support for LLM conversations and context management.

### SqlAlchemyFileRepository
Repository implementation for file metadata storage tracking uploads, content types, and storage locations.

### SqlAlchemyVariableRepository
Repository implementation for variable definition storage supporting template and conversation variable management.

## Related Documentation

- [Ports Overview](../ports/index.md) - Port interface definitions
- [Kits Overview](../kits/index.md) - Business logic implementations
- [Compose](../compose.md) - Adapter composition and dependency injection
- [Philosophy](../philosophy.md) - Hexagonal architecture principles
