"""
Dependency injection module for FastAPI services and database session.

This module provides dependency functions for injecting core services
such as EmbeddingService, VectorStoreService, StorageService, GeneratorService,
and RagService, as well as managing database session lifecycle.

Usage:
    Use `Depends(get_<service_name>)` in your route or other dependency functions
    to inject the appropriate service instance.
"""

from fastapi import Depends
from app.db.database import SessionLocal
from app.services.rag_service import RagService
from app.db.vector.vector_store_factory import get_vector_store
from app.services.embedding.embedder_factory import get_embedder
from app.services.generator.generator_factory import get_generator
from app.services.storage.storage_factory import get_storage_backend
from app.services.reranking.reranker_factory import get_reranker
from app.services.chunking.chunker_factory import get_chunker
from app.services.embedding.embedding_service import EmbeddingService
from app.db.vector.vector_store_service import VectorStoreService
from app.services.storage.storage_service import StorageService
from app.services.generator.generator_service import GeneratorService
from app.services.chunking.chunking_service import ChunkingService
from app.services.reranking.reranking_service import RerankingService
from typing import Optional


def get_db():
    """
    Dependency that provides a new SQLAlchemy session per request.

    Yields:
        Session: A SQLAlchemy session to interact with the database.
    Ensures:
        The session is properly closed after the request is completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_chunking_service(strategy: str ="word", chunk_size: int = 150, overlap: int = 30) -> ChunkingService:
    """
    Provides an instance of ChunkingService with default configuration.

    This function initializes a ChunkingService using the "word" strategy,
    splitting text into chunks of 150 words with an overlap of 30 words.
    This configuration helps retain context between chunks during downstream
    processing such as embedding generation or vector storage.

    Returns
    -------
    ChunkingService
        An initialized ChunkingService instance with word-based chunking strategy.
    """
    chunkingStrategy = get_chunker(strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    return ChunkingService(chunker=chunkingStrategy)


def get_embedding_service(backend: str = "local") -> EmbeddingService:
    """
    Provides an instance of EmbeddingService with a specified backend.

    Parameters
    ----------
    backend : str, optional
        The name of the embedding backend to use. Defaults to "openai".
        Supported values include "openai", "local", etc.

    Returns
    -------
    EmbeddingService
        Configured embedding service instance.
    """
    embedder = get_embedder(backend=backend)
    return EmbeddingService(embedder=embedder)


def get_vector_store_service(
    strategy: str = "inmemory",
    memory_strategy: Optional[str] = None,
    db=Depends(get_db)
) -> VectorStoreService:
    """
    Dependency that provides an instance of VectorStoreService.

    Args:
        strategy (str): The vector store strategy to use ("inmemory", "db", or "hybrid").
        memory_strategy (Optional[str]): Required if strategy is "hybrid" to determine which store to use in memory.
        db (Session): A SQLAlchemy session provided by get_db.

    Returns:
        VectorStoreService: The service responsible for vector storage and retrieval.

    Raises:
        ValueError: If hybrid strategy is selected but memory_strategy is not provided.
    """
    if strategy.lower() == "hybrid":
        if not memory_strategy:
            raise ValueError("memory_strategy is required when using 'hybrid' vector store strategy.")
        vector_store = get_vector_store(strategy=strategy, db_session=db, memory_strategy=memory_strategy)
    else:
        vector_store = get_vector_store(strategy=strategy, db_session=db)

    return VectorStoreService(vector_store=vector_store)


def get_storage_service(backend: str = "sqlite") -> StorageService:
    """
    Dependency provider for StorageService.

    This function returns an instance of StorageService with the specified backend.
    Can be overridden in tests or configured dynamically for different storage systems.

    Args:
        backend (str): The storage backend to use. Supported values include:
                       - "sqlite" (default)
                       - "supabase"
                       - "qdrant"
        **kwargs: Additional keyword arguments passed to the storage backend.

    Returns:
        StorageService: A configured instance of StorageService.
    """
    storage_backend = get_storage_backend(backend=backend)
    return StorageService(backend=storage_backend)


def get_generator_service(provider: str = "openai") -> GeneratorService:
    """
    Creates a GeneratorService instance using the specified LLM provider.

    Parameters
    ----------
    provider : str, optional
        The name of the generator provider to use. Default is "openai".
        Supported options may include "openai", "anthropic", etc.
    **kwargs : dict
        Additional keyword arguments to configure the generator (e.g., model name, API key).

    Returns
    -------
    GeneratorService
        A generator service instance with the specified backend.
    """
    generator = get_generator(provider=provider)
    return GeneratorService(generator=generator)

def get_reranking_service(strategy: str = "bge") -> RerankingService:
    """
    Creates a RerankingService instance using the specified reranker strategy.

    Parameters
    ----------
    strategy : str, optional
        The reranking strategy to use. Default is "bge".
        Supported options include "bge", "none", etc.

    Returns
    -------
    RerankingService
        A reranking service instance with the specified backend strategy.
    """
    reranker = get_reranker(strategy=strategy)
    return RerankingService(reranker=reranker)


def get_rag_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    storage_service: StorageService = Depends(get_storage_service),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    generator_service: GeneratorService = Depends(get_generator_service),
    reranking_service: RerankingService = Depends(get_reranking_service)
) -> RagService:
    """
    Dependency that provides an instance of RagService.

    Args:
        embedding_service (EmbeddingService): Service for embedding queries/documents.
        storage_service (StorageService): Service for managing stored documents.
        vector_store_service (VectorStoreService): Service for vector-based search.
        generator_service (GeneratorService): Service for generating responses.

    Returns:
        RagService: The main RAG pipeline service coordinating the above services.
    """
    return RagService(
        embedding_service=embedding_service,
        storage_service=storage_service,
        vector_store_service=vector_store_service,
        generator_service=generator_service,
        reranking_service=reranking_service
    )
