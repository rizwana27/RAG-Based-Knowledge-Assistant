from typing import List, Dict, Union, Optional, Any
from app.db.vector.base_vector_store import BaseVectorStore
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Service layer that interacts with a vector store implementation.

    This class serves as an abstraction over the underlying vector store,
    handling validation and forwarding requests for storing and querying vector data.
    """

    def __init__(self, vector_store: BaseVectorStore):
        """
        Initializes the VectorStoreService with a specific vector store backend.

        Args:
            vector_store (BaseVectorStore): An instance of a class implementing the BaseVectorStore interface.
        """
        self.vector_store = vector_store
        logger.info(f"Initialized VectorStoreService with vector store backend: {type(vector_store).__name__}")

    def store_chunks(
        self,
        document_id: int,
        chunks: List[Union[str, Dict[str, Union[str, Dict[str, Union[str, int]]]]]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Stores chunks and their corresponding vector embeddings in the vector store.

        Args:
            document_id (int): Identifier of the document these chunks belong to.
            chunks (List[Union[str, Dict]]): A list of text chunks or dictionaries containing 'text' and optional metadata.
            embeddings (List[List[float]]): A list of embedding vectors corresponding to the chunks.

        Raises:
            ValueError: If the number of chunks does not match the number of embeddings.

        Returns:
            None
        """
        if len(chunks) != len(embeddings):
            logger.error(f"store_chunks error | document_id={document_id} | chunks={len(chunks)} | embeddings={len(embeddings)}")
            raise ValueError("Number of chunks and embeddings must be the same.")

        logger.info(f"Storing {len(chunks)} chunks for document_id={document_id}")
        self.vector_store.store_chunks(document_id, chunks, embeddings)
        logger.info(f"Successfully stored chunks for document_id={document_id}")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Union[str, int]]] = None,
        min_score: float = 0.0,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Union[str, float, Dict[str, Any], int]]]:
        """
        Queries the vector store for the most relevant chunks based on the provided query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query.
            top_k (int, optional): Maximum number of top results to return. Defaults to 5.
            knowledge_base_id (Optional[str], optional): Optional knowledge base ID to scope the search.
            filters (Optional[Dict[str, Union[str, int]]], optional): Optional metadata filters.
            min_score (float, optional): Minimum similarity score required to include results. Defaults to 0.0.
            query_text (Optional[str], optional): Optional keyword query to support hybrid search strategies.

        Returns:
            List[Dict[str, Union[str, float, Dict[str, Any], int]]]: A list of matched chunks with metadata and scores.
        """
        logger.info(
            f"Performing vector search | top_k={top_k} | knowledge_base_id={knowledge_base_id} | min_score={min_score} "
            f"| filters={filters} | query_text={query_text}"
        )

        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            knowledge_base_id=knowledge_base_id,
            filters=filters,
            min_score=min_score,
            query_text=query_text
        )

        logger.info(f"Vector search complete | Results found: {len(results)}")
        return results
