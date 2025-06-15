import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.models import Chunk
from app.db.vector.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

class DBVectorStore(BaseVectorStore):
    """
    Vector store implementation that persists chunks and embeddings in a database.

    This implementation uses SQL queries for similarity search using PGVector or similar
    extensions and stores chunk metadata and embeddings in a relational schema.
    """

    def __init__(self, db_session: Session):
        """
        Initializes the DBVectorStore with a SQLAlchemy session.

        Args:
            db_session (Session): SQLAlchemy database session used for database operations.
        """
        self.db = db_session
        logger.info("Initialized DBVectorStore with SQLAlchemy session")

    def store_chunks(self, document_id, chunks, embeddings):
        """
        Stores chunks and their embeddings in the database.

        Args:
            document_id (int): ID of the document these chunks belong to.
            chunks (List[Union[str, Dict]]): List of chunk texts or dicts containing 'text' and optional 'metadata'.
            embeddings (List[List[float]]): List of corresponding embedding vectors.

        Returns:
            None
        """
        logger.info(f"Storing {len(chunks)} chunks for document_id={document_id}")
        for idx, emb in enumerate(embeddings):
            chunk_data = chunks[idx]
            text_val = chunk_data["text"] if isinstance(chunk_data, dict) else chunk_data
            metadata = chunk_data.get("metadata", {}) if isinstance(chunk_data, dict) else {}

            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                text=text_val,
                embedding=emb,
                chunk_metadata=metadata,
            )
            self.db.add(chunk)
        self.db.commit()
        logger.info(f"Successfully stored all chunks for document_id={document_id}")

    def query(
        self,
        query_embedding,
        top_k=5,
        knowledge_base_id=None,
        filters=None,
        min_score=0.0,
        query_text=None,
    ):
        """
        Queries the database for top-k similar chunks based on vector similarity.

        Args:
            query_embedding (List[float]): The embedding vector of the query.
            top_k (int, optional): Number of top results to return. Defaults to 5.
            knowledge_base_id (Optional[str], optional): If provided, restricts results to the specified knowledge base.
            filters (Optional[Dict[str, Union[str, int]]], optional): Metadata filters to apply on the chunks.
            min_score (float, optional): Minimum similarity score threshold. Defaults to 0.0.
            query_text (Optional[str], optional): Currently unused; included for interface compatibility.

        Returns:
            List[Dict[str, Any]]: List of matched chunks with metadata and similarity scores.
        """
        logger.info("Starting vector similarity query")
        logger.debug(f"Query params: top_k={top_k}, min_score={min_score}, kb_id={knowledge_base_id}, filters={filters}")

        # Format the embedding for SQL (PGVector expects array like '{1.0, 0.5, ...}')
        embedding_str = "{" + ",".join(map(str, query_embedding)) + "}"

        base_sql = """
            SELECT id, text, metadata AS chunk_metadata, document_id,
                1 - (embedding <=> :query_embedding) AS similarity
            FROM chunks
            WHERE (1 - (embedding <=> :query_embedding)) >= :min_score
        """
        params = {
            "query_embedding": embedding_str,
            "min_score": min_score,
            "top_k": top_k,
        }

        # Apply knowledge base filter
        if knowledge_base_id:
            logger.debug(f"Applying knowledge_base_id filter: {knowledge_base_id}")
            base_sql += " AND document_id = :knowledge_base_id"
            params["knowledge_base_id"] = knowledge_base_id

        # Apply metadata filters
        if filters:
            for i, (key, value) in enumerate(filters.items()):
                logger.debug(f"Applying metadata filter: {key}={value}")
                base_sql += f" AND metadata->>:key_{i} = :filter_val_{i}"
                params[f"key_{i}"] = key
                params[f"filter_val_{i}"] = str(value)

        base_sql += " ORDER BY embedding <=> :query_embedding LIMIT :top_k"
        sql = text(base_sql)

        logger.debug(f"Executing similarity SQL query:\n{base_sql}\nWith params: {params}")
        result = self.db.execute(sql, params).fetchall()

        logger.info(f"Vector query returned {len(result)} results")

        return [
            {
                "chunk_id": row["id"],
                "text": row["text"],
                "similarity": row["similarity"],
                "chunk_metadata": row["chunk_metadata"],
                "document_id": row["document_id"],
            }
            for row in result
        ]
