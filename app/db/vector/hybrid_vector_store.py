import logging
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy import func, String, cast
from app.db.models import Chunk
from app.db.vector.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

class HybridVectorStore(BaseVectorStore):
    """
    A hybrid vector store that combines vector-based similarity search with keyword-based search.

    This class delegates vector operations to another vector store (e.g., DBVectorStore or InMemoryVectorStore)
    and combines its results with full-text search using SQL queries for improved relevance.

    The final similarity score is a weighted combination of both methods using alpha and beta parameters.
    """

    def __init__(self, db_session: Session, vector_store: BaseVectorStore):
        """
        Initializes the HybridVectorStore with a database session and an underlying vector store.

        Args:
            db_session (Session): SQLAlchemy session for executing keyword-based queries.
            vector_store (BaseVectorStore): An instance of a vector store to handle embedding-based search.
        """
        self.db = db_session
        self.vector_store = vector_store
        self.alpha = 0.7  # weight for vector similarity
        self.beta = 0.3   # weight for keyword similarity
        logger.info(f"HybridVectorStore initialized with backend: {type(vector_store).__name__}")

    def store_chunks(self, document_id: int, chunks: List[Union[str, Dict[str, Any]]], embeddings: List[List[float]]) -> None:
        """
        Stores document chunks and their embeddings using the underlying vector store.

        Args:
            document_id (int): The ID of the document the chunks belong to.
            chunks (List[Union[str, Dict[str, Any]]]): Text chunks or dictionaries with optional metadata.
            embeddings (List[List[float]]): Vector embeddings corresponding to the chunks.

        Returns:
            None
        """
        logger.info(f"Storing {len(chunks)} chunks for document_id={document_id}")
        self.vector_store.store_chunks(document_id, chunks, embeddings)
        logger.info(f"Chunks successfully stored for document_id={document_id}")

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Union[str, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a keyword-based search over stored chunks using SQL `ILIKE` filtering.

        Args:
            query_text (str): The textual query string to search for in chunks.
            top_k (int, optional): Maximum number of results to return. Defaults to 5.
            knowledge_base_id (Optional[str], optional): Filter to restrict results to a specific knowledge base.
            filters (Optional[Dict[str, Union[str, int]]], optional): Additional metadata filters.

        Returns:
            List[Dict[str, Any]]: Matching chunks with a default similarity score of 1.0.
        """
        logger.info(f"Starting keyword search | query_text='{query_text}' | top_k={top_k} | kb_id={knowledge_base_id} | filters={filters}")
        query = self.db.query(Chunk)

        if knowledge_base_id:
            logger.debug(f"Applying knowledge_base_id filter: {knowledge_base_id}")
            query = query.filter(Chunk.document_id == knowledge_base_id)

        dialect = self.db.get_bind().dialect.name
        if filters:
            for key, value in filters.items():
                logger.debug(f"Applying metadata filter: {key}={value}")
                if dialect == "sqlite":
                    query = query.filter(
                        cast(func.json_extract(Chunk.chunk_metadata, f'$.{key}'), String) == str(value)
                    )
                else:
                    query = query.filter(
                        Chunk.chunk_metadata[key].astext == str(value)
                    )

        query = query.filter(Chunk.text.ilike(f"%{query_text}%"))
        results = query.limit(top_k).all()

        logger.info(f"Keyword search matched {len(results)} chunks")
        return [
            {
                "chunk_id": chunk.id,
                "text": chunk.text,
                "similarity": 1.0,
                "chunk_metadata": chunk.chunk_metadata,
                "document_id": chunk.document_id
            }
            for chunk in results
        ]

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Union[str, int]]] = None,
        min_score: float = 0.0,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search combining vector-based similarity and keyword relevance.

        Vector and keyword results are merged using a weighted score:
            final_similarity = alpha * vector_similarity + beta * keyword_similarity

        Args:
            query_embedding (List[float]): The embedding vector for semantic search.
            top_k (int, optional): Number of results to return. Defaults to 5.
            knowledge_base_id (Optional[str], optional): Restrict search to a specific document collection.
            filters (Optional[Dict[str, Union[str, int]]], optional): Metadata filters.
            min_score (float, optional): Minimum similarity threshold for vector results. Defaults to 0.0.
            query_text (Optional[str], optional): Query text for keyword search.

        Returns:
            List[Dict[str, Any]]: Combined and ranked list of relevant chunks.
        """
        logger.info("Starting hybrid query")
        logger.debug(f"Params | top_k={top_k} | kb_id={knowledge_base_id} | min_score={min_score} | query_text='{query_text}' | filters={filters}")

        # Step 1: Vector search
        vector_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
            filters=filters,
            min_score=min_score,
            query_text=query_text
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

        # Step 2: Keyword search
        keyword_results = self.keyword_search(
            query_text=query_text or "",
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
            filters=filters
        )
        logger.info(f"Keyword search returned {len(keyword_results)} results")

        # Step 3: Merge and re-rank
        combined = {item["chunk_id"]: item for item in vector_results}
        logger.debug("Merging results")

        for item in keyword_results:
            if item["chunk_id"] in combined:
                combined_score = (
                    self.alpha * combined[item["chunk_id"]]["similarity"]
                    + self.beta * item["similarity"]
                )
                logger.debug(f"Re-weighted similarity for chunk_id={item['chunk_id']}: {combined_score}")
                combined[item["chunk_id"]]["similarity"] = combined_score
            else:
                combined[item["chunk_id"]] = item

        # Step 4: Sort by final similarity
        sorted_results = sorted(combined.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
        logger.info(f"Returning top {len(sorted_results)} hybrid results")

        return sorted_results
