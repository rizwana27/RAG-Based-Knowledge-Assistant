import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import cast, String
from typing import Dict, Optional, Union, List, Any
from concurrent.futures import ThreadPoolExecutor
from app.db.models import Chunk
from app.db.vector.base_vector_store import BaseVectorStore
from app.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory implementation of the BaseVectorStore using a SQLAlchemy database session.

    This class simulates a vector store by storing text chunks and their embeddings
    in a relational database and uses cosine similarity for querying.
    """

    def __init__(self, db_session: Session):
        """
        Initializes the in-memory vector store.

        Args:
            db_session (Session): SQLAlchemy database session used for persistence.
        """
        self.db = db_session
        logger.info("InMemoryVectorStore initialized with a database session.")

    def store_chunks(
        self,
        document_id: Union[int, str],
        chunks: List[Union[str, Dict[str, Union[str, Dict[str, Union[str, int]]]]]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Stores text chunks and their corresponding vector embeddings.

        Args:
            document_id (Union[int, str]): ID of the document to which the chunks belong.
            chunks (List[Union[str, Dict]]): Text chunks or dicts with 'text' and optional 'metadata'.
            embeddings (List[List[float]]): Corresponding list of vector embeddings.

        Returns:
            None
        """
        logger.info(f"Storing {len(chunks)} chunks for document_id={document_id}")

        for idx, emb in enumerate(embeddings):
            chunk_data = chunks[idx]
            text = chunk_data["text"] if isinstance(chunk_data, dict) else chunk_data
            chunk_metadata = chunk_data.get("metadata", {}) if isinstance(chunk_data, dict) else {}

            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                text=text,
                embedding=emb,
                chunk_metadata=chunk_metadata,
            )
            self.db.add(chunk)
            logger.debug(f"Added chunk index={idx} to session")

        self.db.commit()
        logger.info(f"Successfully stored chunks for document_id={document_id}")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        knowledge_base_id: Optional[Union[int, str]] = None,
        filters: Optional[Dict[str, Union[str, int]]] = None,
        min_score: float = 0.0,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Union[str, float, Dict[str, Any], int]]]:
        """
        Queries for the most similar chunks based on the input query embedding.

        Args:
            query_embedding (List[float]): The vector embedding of the search query.
            top_k (int, optional): Number of top results to return. Defaults to 5.
            knowledge_base_id (Optional[Union[int, str]], optional): Restrict search to this document ID.
            filters (Optional[Dict[str, Union[str, int]]], optional): Metadata filters to apply.
            min_score (float, optional): Minimum similarity score threshold. Defaults to 0.0.
            query_text (Optional[str], optional): Not used in this implementation.

        Returns:
            List[Dict]: List of top matching chunks with metadata and similarity score.
        """
        logger.info(f"Querying chunks | top_k={top_k} | kb_id={knowledge_base_id} | filters={filters} | min_score={min_score}")

        query = self.db.query(Chunk)

        if knowledge_base_id:
            query = query.filter(Chunk.document_id == knowledge_base_id)
            logger.debug(f"Filtered by knowledge_base_id={knowledge_base_id}")

        if filters:
            for key, value in filters.items():
                query = query.filter(cast(Chunk.chunk_metadata[key], String) == value)
                logger.debug(f"Applied metadata filter: {key}={value}")

        chunks = query.all()
        logger.info(f"Retrieved {len(chunks)} chunks from DB for similarity scoring")

        def compute_score(chunk: Chunk) -> Optional[Dict[str, Any]]:
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                return {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "similarity": score,
                    "chunk_metadata": chunk.chunk_metadata,
                    "document_id": chunk.document_id,
                }
            return None

        with ThreadPoolExecutor() as executor:
            results = list(filter(None, executor.map(compute_score, chunks)))

        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"{len(results)} results passed the min_score filter. Returning top {top_k}.")
        return results[:top_k]
