import logging
from app.db.vector.in_memory_vector_store import InMemoryVectorStore
from app.db.vector.db_vector_store import DBVectorStore
from app.db.vector.hybrid_vector_store import HybridVectorStore
from app.db.vector.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

def get_vector_store(strategy: str, db_session, **kwargs) -> BaseVectorStore:
    """
    Factory function to initialize the appropriate vector store strategy.

    Args:
        strategy (str): The vector store strategy to use ("inmemory", "db", "hybrid").
        db_session: SQLAlchemy session.
        **kwargs: Additional parameters. For hybrid, requires `memory_strategy`.

    Returns:
        BaseVectorStore
    """
    strategy = strategy.lower()
    logger.info(f"Vector store factory called with strategy: '{strategy}'")

    if strategy == "inmemory":
        logger.info("Initializing InMemoryVectorStore")
        return InMemoryVectorStore(db_session, **kwargs)

    elif strategy == "db":
        logger.info("Initializing DBVectorStore")
        return DBVectorStore(db_session, **kwargs)

    elif strategy == "hybrid":
        memory_strategy = kwargs.pop("memory_strategy", None)
        if not memory_strategy:
            logger.error("Missing `memory_strategy` for hybrid vector store strategy")
            raise ValueError("memory_strategy is required when using hybrid strategy")

        logger.info(f"Hybrid strategy selected. Inner memory strategy: '{memory_strategy}'")

        # Instantiate the inner memory vector store
        if memory_strategy == "inmemory":
            memory_store = InMemoryVectorStore(db_session)
            logger.info("Inner store: InMemoryVectorStore initialized")
        elif memory_strategy == "db":
            memory_store = DBVectorStore(db_session)
            logger.info("Inner store: DBVectorStore initialized")
        else:
            logger.error(f"Unsupported memory strategy: '{memory_strategy}'")
            raise ValueError(f"Unsupported memory strategy: {memory_strategy}")

        logger.info("Initializing HybridVectorStore")
        return HybridVectorStore(db_session=db_session, vector_store=memory_store)

    else:
        logger.error(f"Unsupported vector store strategy: '{strategy}'")
        raise ValueError(f"Unsupported vector store strategy: {strategy}")
