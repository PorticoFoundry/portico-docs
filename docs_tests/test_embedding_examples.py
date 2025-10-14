"""Test examples for Embedding documentation.

This module tests code examples from embedding-related documentation to ensure
they remain accurate and working.
"""

from datetime import datetime
from typing import List

import pytest

# --8<-- [start:imports]
from portico.ports.embedding import (
    EmbeddingConfig,
    EmbeddingData,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

# --8<-- [end:imports]


# --8<-- [start:embedding-request-single]
def test_embedding_request_single():
    """Embedding request for a single text."""
    request = EmbeddingRequest(
        texts="Hello, world!",
        model="text-embedding-3-small",
    )

    assert request.texts == "Hello, world!"
    assert request.model == "text-embedding-3-small"
    assert request.encoding_format == "float"


# --8<-- [end:embedding-request-single]


# --8<-- [start:embedding-request-multiple]
def test_embedding_request_multiple():
    """Embedding request for multiple texts."""
    request = EmbeddingRequest(
        texts=[
            "First document",
            "Second document",
            "Third document",
        ],
        model="text-embedding-3-small",
    )

    assert isinstance(request.texts, list)
    assert len(request.texts) == 3
    assert request.texts[0] == "First document"


# --8<-- [end:embedding-request-multiple]


# --8<-- [start:embedding-request-dimensions]
def test_embedding_request_dimensions():
    """Embedding request with custom dimensions."""
    request = EmbeddingRequest(
        texts="Sample text",
        model="text-embedding-3-small",
        dimensions=512,  # Reduce from default 1536
    )

    assert request.dimensions == 512
    assert request.model == "text-embedding-3-small"


# --8<-- [end:embedding-request-dimensions]


# --8<-- [start:embedding-request-user-tracking]
def test_embedding_request_user_tracking():
    """Embedding request with user tracking."""
    request = EmbeddingRequest(
        texts="User query text",
        model="text-embedding-3-small",
        user="user_12345",  # For usage tracking
    )

    assert request.user == "user_12345"


# --8<-- [end:embedding-request-user-tracking]


# --8<-- [start:embedding-data]
def test_embedding_data():
    """Individual embedding data."""
    data = EmbeddingData(
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        index=0,
        object="embedding",
    )

    assert len(data.embedding) == 5
    assert data.embedding[0] == 0.1
    assert data.index == 0
    assert data.object == "embedding"


# --8<-- [end:embedding-data]


# --8<-- [start:embedding-usage]
def test_embedding_usage():
    """Token usage information."""
    usage = EmbeddingUsage(
        prompt_tokens=15,
        total_tokens=15,
    )

    assert usage.prompt_tokens == 15
    assert usage.total_tokens == 15


# --8<-- [end:embedding-usage]


# --8<-- [start:embedding-response-single]
def test_embedding_response_single():
    """Embedding response for single text."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(
                embedding=[0.1, 0.2, 0.3],
                index=0,
            )
        ],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
    )

    assert len(response.data) == 1
    assert response.model == "text-embedding-3-small"
    assert response.usage.prompt_tokens == 5

    # Test single_embedding property
    single = response.single_embedding
    assert single == [0.1, 0.2, 0.3]


# --8<-- [end:embedding-response-single]


# --8<-- [start:embedding-response-multiple]
def test_embedding_response_multiple():
    """Embedding response for multiple texts."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0),
            EmbeddingData(embedding=[0.4, 0.5, 0.6], index=1),
            EmbeddingData(embedding=[0.7, 0.8, 0.9], index=2),
        ],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=15, total_tokens=15),
    )

    assert len(response.data) == 3
    assert response.data[0].index == 0
    assert response.data[1].index == 1
    assert response.data[2].index == 2

    # Test embeddings property
    embeddings = response.embeddings
    assert len(embeddings) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[2] == [0.7, 0.8, 0.9]


# --8<-- [end:embedding-response-multiple]


# --8<-- [start:embedding-response-properties]
def test_embedding_response_properties():
    """Embedding response convenience properties."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2], index=0),
            EmbeddingData(embedding=[0.3, 0.4], index=1),
        ],
        model="text-embedding-3-small",
    )

    # embeddings property returns list of vectors
    embeddings = response.embeddings
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)
    assert embeddings[0] == [0.1, 0.2]


# --8<-- [end:embedding-response-properties]


# --8<-- [start:embedding-single-property-error]
def test_embedding_single_property_error():
    """single_embedding property errors on multiple embeddings."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1], index=0),
            EmbeddingData(embedding=[0.2], index=1),
        ],
        model="test-model",
    )

    # Should raise error when multiple embeddings
    with pytest.raises(ValueError, match="Expected single embedding, got 2"):
        _ = response.single_embedding


# --8<-- [end:embedding-single-property-error]


# --8<-- [start:embedding-config]
def test_embedding_config():
    """Embedding configuration."""
    config = EmbeddingConfig(
        default_model="text-embedding-3-small",
        batch_size=100,
        max_retries=3,
        timeout_seconds=30.0,
    )

    assert config.default_model == "text-embedding-3-small"
    assert config.batch_size == 100
    assert config.max_retries == 3
    assert config.timeout_seconds == 30.0
    assert config.dimensions is None  # Use model default


# --8<-- [end:embedding-config]


# --8<-- [start:embedding-config-dimensions]
def test_embedding_config_dimensions():
    """Embedding configuration with custom dimensions."""
    config = EmbeddingConfig(
        default_model="text-embedding-3-small",
        dimensions=512,  # Custom dimension
        batch_size=50,
    )

    assert config.dimensions == 512
    assert config.batch_size == 50


# --8<-- [end:embedding-config-dimensions]


# --8<-- [start:embedding-config-defaults]
def test_embedding_config_defaults():
    """Embedding configuration default values."""
    config = EmbeddingConfig()

    assert config.default_model == "text-embedding-3-small"
    assert config.batch_size == 100
    assert config.max_retries == 3
    assert config.timeout_seconds == 30.0
    assert config.dimensions is None


# --8<-- [end:embedding-config-defaults]


# --8<-- [start:embedding-provider-interface]
@pytest.mark.asyncio
async def test_embedding_provider_interface():
    """Embedding provider interface implementation."""

    class MockEmbeddingProvider(EmbeddingProvider):
        """Mock embedding provider for testing."""

        async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            """Generate mock embeddings."""
            # Convert single text to list if needed
            texts = (
                request.texts if isinstance(request.texts, list) else [request.texts]
            )

            # Create mock embeddings
            data = [
                EmbeddingData(
                    embedding=[0.1 * (i + 1), 0.2 * (i + 1)],
                    index=i,
                )
                for i in range(len(texts))
            ]

            return EmbeddingResponse(
                data=data,
                model=request.model or "mock-model",
                usage=EmbeddingUsage(
                    prompt_tokens=len(texts) * 5,
                    total_tokens=len(texts) * 5,
                ),
            )

        async def embed_text(
            self, text: str, model: str | None = None, **kwargs
        ) -> List[float]:
            """Embed single text."""
            request = EmbeddingRequest(texts=text, model=model)
            response = await self.embed(request)
            return response.single_embedding

        async def embed_texts(
            self, texts: List[str], model: str | None = None, **kwargs
        ) -> List[List[float]]:
            """Embed multiple texts."""
            request = EmbeddingRequest(texts=texts, model=model)
            response = await self.embed(request)
            return response.embeddings

        async def get_embedding_dimension(self, model: str | None = None) -> int:
            """Get embedding dimension."""
            return 2  # Mock dimension

        async def list_models(self) -> List[str]:
            """List available models."""
            return ["mock-model-1", "mock-model-2"]

    provider = MockEmbeddingProvider()

    # Test embed
    request = EmbeddingRequest(texts="Hello", model="mock-model")
    response = await provider.embed(request)
    assert len(response.data) == 1
    assert response.model == "mock-model"

    # Test embed_text
    embedding = await provider.embed_text("Hello")
    assert isinstance(embedding, list)
    assert len(embedding) == 2

    # Test embed_texts
    embeddings = await provider.embed_texts(["Hello", "World"])
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], list)

    # Test get_embedding_dimension
    dim = await provider.get_embedding_dimension()
    assert dim == 2

    # Test list_models
    models = await provider.list_models()
    assert len(models) == 2
    assert "mock-model-1" in models


# --8<-- [end:embedding-provider-interface]


# --8<-- [start:embedding-batch-processing]
def test_embedding_batch_processing():
    """Batch processing configuration."""
    config = EmbeddingConfig(
        batch_size=50,  # Process 50 texts at a time
    )

    # Simulate large batch
    total_texts = 200
    batch_size = config.batch_size
    num_batches = (total_texts + batch_size - 1) // batch_size

    assert num_batches == 4  # 200 / 50 = 4 batches


# --8<-- [end:embedding-batch-processing]


# --8<-- [start:embedding-vector-operations]
def test_embedding_vector_operations():
    """Embedding vector operations."""
    # Create sample embeddings
    emb1 = [0.1, 0.2, 0.3]
    emb2 = [0.4, 0.5, 0.6]

    # Cosine similarity (simplified - not normalized)
    dot_product = sum(a * b for a, b in zip(emb1, emb2))

    assert dot_product == pytest.approx(0.32, rel=0.01)

    # Euclidean distance
    distance = sum((a - b) ** 2 for a, b in zip(emb1, emb2)) ** 0.5

    assert distance == pytest.approx(0.52, rel=0.01)


# --8<-- [end:embedding-vector-operations]


# --8<-- [start:embedding-response-metadata]
def test_embedding_response_metadata():
    """Embedding response with metadata."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2], index=0),
        ],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
    )

    # Check automatic fields
    assert response.id is not None
    assert response.object == "list"
    assert isinstance(response.created_at, datetime)
    assert response.created_at.tzinfo is not None


# --8<-- [end:embedding-response-metadata]


# --8<-- [start:embedding-encoding-formats]
def test_embedding_encoding_formats():
    """Embedding encoding format options."""
    # Float encoding (default)
    float_request = EmbeddingRequest(
        texts="Sample text",
        encoding_format="float",
    )
    assert float_request.encoding_format == "float"

    # Base64 encoding (for smaller payload)
    base64_request = EmbeddingRequest(
        texts="Sample text",
        encoding_format="base64",
    )
    assert base64_request.encoding_format == "base64"


# --8<-- [end:embedding-encoding-formats]


# --8<-- [start:embedding-model-selection]
def test_embedding_model_selection():
    """Different embedding model options."""
    # Small model (faster, cheaper)
    small_request = EmbeddingRequest(
        texts="Query text",
        model="text-embedding-3-small",
    )
    assert small_request.model == "text-embedding-3-small"

    # Large model (more accurate)
    large_request = EmbeddingRequest(
        texts="Query text",
        model="text-embedding-3-large",
    )
    assert large_request.model == "text-embedding-3-large"

    # No model specified (use provider default)
    default_request = EmbeddingRequest(
        texts="Query text",
    )
    assert default_request.model is None


# --8<-- [end:embedding-model-selection]


# --8<-- [start:embedding-response-iteration]
def test_embedding_response_iteration():
    """Iterate over embedding response data."""
    response = EmbeddingResponse(
        data=[
            EmbeddingData(embedding=[0.1, 0.2], index=0),
            EmbeddingData(embedding=[0.3, 0.4], index=1),
            EmbeddingData(embedding=[0.5, 0.6], index=2),
        ],
        model="test-model",
    )

    # Iterate over embeddings
    embeddings_list = []
    for item in response.data:
        embeddings_list.append(item.embedding)

    assert len(embeddings_list) == 3
    assert embeddings_list[0] == [0.1, 0.2]
    assert embeddings_list[2] == [0.5, 0.6]


# --8<-- [end:embedding-response-iteration]


# --8<-- [start:embedding-dimension-sizes]
def test_embedding_dimension_sizes():
    """Common embedding dimension sizes."""
    # OpenAI text-embedding-3-small default
    small_dim = 1536

    # OpenAI text-embedding-3-large default
    large_dim = 3072

    # Custom reduced dimensions
    reduced_dim = 512

    assert small_dim == 1536
    assert large_dim == 3072
    assert reduced_dim < small_dim


# --8<-- [end:embedding-dimension-sizes]
