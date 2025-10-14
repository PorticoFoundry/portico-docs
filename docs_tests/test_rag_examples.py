"""Test examples for RAG documentation.

This module tests code examples from docs/kits/rag.md to ensure they remain
accurate and working.
"""

from typing import List
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from portico.kits.llm import LLMService

# --8<-- [start:imports]
from portico.kits.rag import (
    RAGConfig,
    RAGQuery,
    RAGResponse,
    RAGService,
    SourceCitation,
)

# --8<-- [end:imports]


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    provider = AsyncMock()
    # Return fake embeddings
    provider.create_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return provider


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = AsyncMock()
    store.add_documents = AsyncMock(return_value=None)
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    mock_response = Mock()
    mock_response.message.content = "This is a test response."
    provider.complete = AsyncMock(return_value=mock_response)
    return provider


@pytest.fixture
def mock_llm_service(mock_llm_provider):
    """Mock LLM service."""
    return LLMService(mock_llm_provider)


# --8<-- [start:rag-config-example]
def test_rag_config_example():
    """Custom RAG configuration example."""
    config = RAGConfig(
        default_k=5,
        max_k=20,
        default_similarity_threshold=0.7,
        max_context_tokens=4000,
        default_temperature=0.1,
        system_prompt=(
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information from the context to answer questions. "
            "If the context doesn't contain enough information, say so clearly."
        ),
    )

    assert config.default_k == 5
    assert config.max_k == 20
    assert config.default_similarity_threshold == 0.7
    assert config.max_context_tokens == 4000
    assert config.default_temperature == 0.1
    assert "helpful assistant" in config.system_prompt


# --8<-- [end:rag-config-example]


# --8<-- [start:document-metadata-structure]
def test_document_metadata_structure():
    """Document metadata structure example."""
    metadata = {
        "title": "Introduction to AI",
        "author": "Expert",
        "source_url": "https://example.com/ai-guide",
    }

    assert metadata["title"] == "Introduction to AI"
    assert metadata["author"] == "Expert"
    assert metadata["source_url"] == "https://example.com/ai-guide"


# --8<-- [end:document-metadata-structure]


# --8<-- [start:rag-query-basic]
def test_rag_query_basic():
    """Basic RAG query construction."""
    query = RAGQuery(
        query="What is machine learning?",
        k=5,
        similarity_threshold=0.7,
    )

    assert query.query == "What is machine learning?"
    assert query.k == 5
    assert query.similarity_threshold == 0.7


# --8<-- [end:rag-query-basic]


# --8<-- [start:rag-query-with-filters]
def test_rag_query_with_filters():
    """RAG query with metadata filters."""
    query = RAGQuery(
        query="What are the benefits?",
        k=5,
        metadata_filters={
            "document_type": "research_paper",
            "year": 2024,
            "category": "machine_learning",
        },
        namespace="research-docs",
    )

    assert query.metadata_filters["document_type"] == "research_paper"
    assert query.metadata_filters["year"] == 2024
    assert query.namespace == "research-docs"


# --8<-- [end:rag-query-with-filters]


# --8<-- [start:rag-query-custom-prompt]
def test_rag_query_custom_prompt():
    """RAG query with custom system prompt."""
    query = RAGQuery(
        query="Summarize the key findings",
        k=5,
        system_prompt=(
            "You are an expert research analyst. Summarize the key findings "
            "from the provided research papers. Focus on methodology, results, "
            "and conclusions. Be concise and accurate."
        ),
        temperature=0.2,
    )

    assert query.query == "Summarize the key findings"
    assert "expert research analyst" in query.system_prompt
    assert query.temperature == 0.2


# --8<-- [end:rag-query-custom-prompt]


# --8<-- [start:chunking-strategy-good]
def test_chunking_strategy_good():
    """Good chunking strategy with appropriate size and overlap."""
    chunk_size = 1000
    chunk_overlap = 200

    assert chunk_size == 1000
    assert chunk_overlap == 200
    assert chunk_overlap / chunk_size == 0.2  # 20% overlap


# --8<-- [end:chunking-strategy-good]


# --8<-- [start:similarity-threshold-focused]
def test_similarity_threshold_focused():
    """Similarity threshold for focused queries."""
    query = RAGQuery(
        query="What is the main conclusion?",
        k=5,
        similarity_threshold=0.7,
    )

    assert query.similarity_threshold == 0.7


# --8<-- [end:similarity-threshold-focused]


# --8<-- [start:similarity-threshold-broad]
def test_similarity_threshold_broad():
    """Lower similarity threshold for broader queries."""
    query = RAGQuery(
        query="Tell me about this document",
        k=10,
        similarity_threshold=0.5,
    )

    assert query.k == 10
    assert query.similarity_threshold == 0.5


# --8<-- [end:similarity-threshold-broad]


# --8<-- [start:temperature-control]
def test_temperature_control():
    """Temperature control for different tasks."""
    TASK_TEMPERATURES = {
        "factual_qa": 0.1,
        "summarization": 0.3,
        "creative_writing": 0.8,
    }

    factual_query = RAGQuery(
        query="What are the facts?",
        temperature=TASK_TEMPERATURES["factual_qa"],
    )

    assert factual_query.temperature == 0.1

    summary_query = RAGQuery(
        query="Summarize this",
        temperature=TASK_TEMPERATURES["summarization"],
    )

    assert summary_query.temperature == 0.3


# --8<-- [end:temperature-control]


# --8<-- [start:metadata-organization]
def test_metadata_organization():
    """Structured metadata for filtering."""
    metadata = {
        "document_id": str(uuid4()),
        "document_type": "research_paper",
        "title": "AI Safety Research",
        "author": "Dr. Smith",
        "year": 2024,
        "category": "machine_learning",
        "tags": ["safety", "alignment", "ai"],
        "source_url": "https://example.com/paper",
    }

    query = RAGQuery(
        query="What are the safety concerns?",
        metadata_filters={
            "category": "machine_learning",
            "tags": "safety",
            "year": 2024,
        },
    )

    assert query.metadata_filters["category"] == "machine_learning"
    assert query.metadata_filters["tags"] == "safety"
    assert query.metadata_filters["year"] == 2024


# --8<-- [end:metadata-organization]


# --8<-- [start:multi-document-query]
def test_multi_document_query():
    """Query across multiple documents."""
    query = RAGQuery(
        query="Compare the approaches in these papers",
        k=10,
        metadata_filters={
            "document_id": ["doc-1", "doc-2", "doc-3"],
        },
    )

    assert query.k == 10
    assert len(query.metadata_filters["document_id"]) == 3


# --8<-- [end:multi-document-query]


# --8<-- [start:rag-response-properties]
@pytest.mark.asyncio
async def test_rag_response_properties():
    """RAG response properties."""
    # Create source citations
    source1 = SourceCitation(
        id="chunk-1",
        content="Machine learning is a subset of AI.",
        score=0.9,
        metadata={"document_id": "doc-1"},
    )
    source2 = SourceCitation(
        id="chunk-2",
        content="Neural networks are key to ML.",
        score=0.8,
        metadata={"document_id": "doc-1"},
    )

    response = RAGResponse(
        query="What is machine learning?",
        response="Machine learning is a subset of artificial intelligence.",
        sources=[source1, source2],
        model="gpt-4",
        tokens_used=150,
        relevance_score=0.85,
    )

    assert response.source_count == 2
    assert response.has_sources is True
    assert response.relevance_score == 0.85


# --8<-- [end:rag-response-properties]


# --8<-- [start:check-response-quality]
@pytest.mark.asyncio
async def test_check_response_quality():
    """Check response quality."""
    # Response with no sources
    response_no_sources = RAGResponse(
        query="test",
        response="answer",
        sources=[],
        relevance_score=0.3,
        total_time_ms=6000,
    )

    if not response_no_sources.has_sources:
        warning_issued = True
    else:
        warning_issued = False

    assert warning_issued is True

    if (
        response_no_sources.relevance_score
        and response_no_sources.relevance_score < 0.5
    ):
        low_relevance = True
    else:
        low_relevance = False

    assert low_relevance is True

    if response_no_sources.total_time_ms and response_no_sources.total_time_ms > 5000:
        slow_query = True
    else:
        slow_query = False

    assert slow_query is True


# --8<-- [end:check-response-quality]


# --8<-- [start:cache-key-generation]
def test_cache_key_generation():
    """Generate cache key from query."""
    import hashlib

    query = RAGQuery(
        query="What is AI?",
        k=5,
        similarity_threshold=0.7,
    )

    key_data = f"{query.query}:{query.k}:{query.similarity_threshold}"
    cache_key = f"rag:{hashlib.md5(key_data.encode()).hexdigest()}"

    assert cache_key.startswith("rag:")
    assert len(cache_key) == 36  # "rag:" + 32 hex chars


# --8<-- [end:cache-key-generation]


# --8<-- [start:conversation-history-class]
def test_conversation_history_class():
    """Conversational RAG class structure."""

    class ConversationalRAG:
        """RAG with conversation history."""

        def __init__(self, rag_service: RAGService):
            self.rag_service = rag_service
            self.history: List[tuple[str, str]] = []

        def add_turn(self, user_query: str, assistant_response: str):
            """Add a conversation turn."""
            self.history.append((user_query, assistant_response))

        def get_recent_history(self, n: int = 3) -> List[tuple[str, str]]:
            """Get recent conversation history."""
            return self.history[-n:]

    # Mock RAG service
    mock_rag = Mock(spec=RAGService)
    conv_rag = ConversationalRAG(mock_rag)

    conv_rag.add_turn("What is AI?", "AI is artificial intelligence.")
    conv_rag.add_turn("How does it work?", "It uses algorithms and data.")

    assert len(conv_rag.history) == 2
    assert conv_rag.get_recent_history(1) == [
        ("How does it work?", "It uses algorithms and data.")
    ]


# --8<-- [end:conversation-history-class]


# --8<-- [start:rag-config-defaults]
def test_rag_config_defaults():
    """RAG configuration with defaults."""
    config = RAGConfig()

    assert config.default_k == 5
    assert config.max_k == 20
    assert config.default_similarity_threshold == 0.0
    assert config.max_context_tokens == 4000
    assert config.default_temperature == 0.1
    assert config.default_max_tokens == 500
    assert config.min_sources_for_response == 1
    assert config.require_source_citation is True


# --8<-- [end:rag-config-defaults]


# --8<-- [start:process-document-with-metadata]
def test_process_document_with_metadata():
    """Process document with comprehensive metadata."""
    doc_id = uuid4()
    metadata = {
        "document_id": str(doc_id),
        "document_type": "research_paper",
        "title": "AI Safety Research",
        "author": "Dr. Smith",
        "year": 2024,
        "category": "machine_learning",
        "tags": ["safety", "alignment", "ai"],
        "source_url": "https://example.com/paper",
    }

    assert metadata["document_type"] == "research_paper"
    assert metadata["year"] == 2024
    assert "safety" in metadata["tags"]


# --8<-- [end:process-document-with-metadata]


# --8<-- [start:rag-query-model-params]
def test_rag_query_model_params():
    """RAG query with specific model parameters."""
    query = RAGQuery(
        query="Explain quantum computing",
        k=5,
        model="gpt-4",
        temperature=0.3,
        max_tokens=1000,
        include_sources=True,
    )

    assert query.model == "gpt-4"
    assert query.temperature == 0.3
    assert query.max_tokens == 1000
    assert query.include_sources is True


# --8<-- [end:rag-query-model-params]


# --8<-- [start:rag-context-template]
def test_rag_context_template():
    """RAG configuration with context template."""
    config = RAGConfig(
        context_template="Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        source_separator="\n\n---\n\n",
        include_source_metadata=True,
    )

    assert "{context}" in config.context_template
    assert "{query}" in config.context_template
    assert config.source_separator == "\n\n---\n\n"
    assert config.include_source_metadata is True


# --8<-- [end:rag-context-template]


# --8<-- [start:namespace-organization]
def test_namespace_organization():
    """Using namespaces for document organization."""
    user_id = uuid4()

    # User-specific namespace
    user_query = RAGQuery(
        query="Find my documents",
        namespace=f"user-{user_id}",
    )

    assert user_query.namespace == f"user-{user_id}"

    # Team namespace
    team_query = RAGQuery(
        query="Find team documents",
        namespace="team-engineering",
    )

    assert team_query.namespace == "team-engineering"


# --8<-- [end:namespace-organization]
