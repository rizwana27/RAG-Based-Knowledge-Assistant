"""
Routes for handling chat and semantic search functionality in the RAG API.

These endpoints expose:
- `/chat`: Chat completion using RAG pipeline.
- `/search`: Semantic vector search based on query embeddings.

Dependencies are injected using FastAPI's Depends mechanism for testability
and modular design.
"""

from fastapi import APIRouter, HTTPException, Depends
from app.api.models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    SearchRequest,
    SearchResponse,
    DocumentChunk
)
from app.api.dependencies import (
    get_embedding_service,
    get_vector_store_service,
    get_rag_service
)
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service=Depends(get_rag_service)
):
    """
    Handle a chat request using the RAG pipeline.

    Args:
        request (ChatRequest): Input request containing query and optional conversation/KB context.
        rag_service (RagService): Injected RAG service instance.

    Returns:
        ChatResponse: Contains assistant's reply, context sources, and conversation metadata.

    Raises:
        HTTPException: 400 if query is missing, 500 for internal processing errors.
    """
    if not request.query:
        logger.warning("Received empty query in /chat request.")
        raise HTTPException(status_code=400, detail="No user message found in request.")

    logger.info(f"Received chat request | Query: {request.query} | Conversation ID: {request.conversation_id} | Knowledge Base ID: {request.knowledge_base_id}")
    try:
        rag_result = rag_service.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            knowledge_base_id=request.knowledge_base_id,
            top_k=5,
            min_score=0.0,
        )

        logger.info(f"Generated response for conversation_id={rag_result['conversation_id']} | Retrieved {len(rag_result.get('context_chunks', []))} context chunks")

        response_message = ChatMessage(role="assistant", content=rag_result["answer"])

        return ChatResponse(
            message=response_message,
            conversation_id=rag_result["conversation_id"],
            created_at=datetime.now(),
            sources=[
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("chunk_metadata", {}),
                    "similarity_score": chunk.get("similarity"),
                }
                for chunk in rag_result.get("context_chunks", [])
            ],
        )
    except Exception as e:
        logger.exception(f"Error during chat request processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    embedding_service=Depends(get_embedding_service),
    vector_store_service=Depends(get_vector_store_service),
):
    """
    Perform semantic vector search for similar documents.

    Args:
        request (SearchRequest): Input query string, filters, top-k, and score threshold.
        embedding_service (EmbeddingService): Injected service to generate query embeddings.
        vector_store_service (VectorStoreService): Injected service to perform similarity search.

    Returns:
        SearchResponse: List of most relevant document chunks and total results found.

    Raises:
        HTTPException: 500 if any error occurs during embedding or search.
    """
    logger.info(f"Received search request | Query: {request.query} | TopK: {request.limit} | MinScore: {request.min_score} | Filters: {request.filters}")

    try:
        query_embedding = embedding_service.get_embedding(request.query)
        logger.debug(f"Query embedding generated with dimension: {len(query_embedding)}")

        results = vector_store_service.query(
            query_embedding=query_embedding,
            top_k=request.limit,
            filters=request.filters,
            min_score=request.min_score or 0.0
        )

        logger.info(f"Search completed | Found {len(results)} matching chunks")

        chunks: List[DocumentChunk] = [
            DocumentChunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata=chunk.get("chunk_metadata"),
                similarity_score=chunk.get("similarity")
            )
            for chunk in results
        ]

        return SearchResponse(
            query=request.query,
            results=chunks,
            total_found=len(results)
        )
    except Exception as e:
        logger.exception("Unexpected error during semantic search.")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error. Please try again later."
        )
