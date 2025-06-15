import logging
from typing import List, Dict, Union, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Document, Chunk, Conversation, Message
from app.services.storage.base_storage import BaseStorage

logger = logging.getLogger(__name__)

class SQLiteStorage(BaseStorage):
    """
    SQLite-based implementation of the BaseStorage interface.

    This class provides methods to persist document metadata and associated
    text chunks along with their vector embeddings into a local SQLite database.
    It uses SQLAlchemy ORM models defined in the app.db.models module.

    Attributes:
        db (Session): A SQLAlchemy session used to interact with the SQLite database.
    """

    def __init__(self):
        """
        Initializes the SQLiteStorage with a new SQLAlchemy database session.
        """
        self.db: Session = SessionLocal()
        logger.info("Initialized SQLiteStorage with new DB session")

    def store_document(self, name: str, document_metadata: dict, path: str) -> Document:
        """
        Store a document entry in the database.

        Parameters
        ----------
        name : str
            The name of the document (e.g., filename).
        document_metadata : dict
            A dictionary containing metadata about the document.
        path : str
            The file path or URI where the document is stored.

        Returns
        -------
        Document
            The SQLAlchemy Document model instance representing the stored document.
        """
        logger.info(f"Storing document: {name}, path: {path}")
        document = Document(name=name, document_metadata=document_metadata, path=path)
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        logger.info(f"Document stored with ID: {document.id}")
        return document

    def store_chunks(
        self,
        document_id: int,
        chunks: List[Dict[str, Union[str, dict]]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Store text chunks along with their embeddings and metadata in the database.

        Parameters
        ----------
        document_id : int
            The ID of the document to which these chunks belong.
        chunks : List[Dict[str, Union[str, dict]]]
            A list of dictionaries, each containing:
                - 'text' (str): The chunk text.
                - 'metadata' (dict, optional): Metadata associated with the chunk.
        embeddings : List[List[float]]
            A list of embedding vectors corresponding to each chunk.

        Returns
        -------
        None
            Commits the chunks and embeddings to the database.
        """
        logger.info(f"Storing {len(chunks)} chunks for document ID: {document_id}")
        for i, chunk_info in enumerate(chunks):
            chunk_text = chunk_info["text"]
            chunk_metadata = chunk_info.get("metadata", {})

            new_chunk = Chunk(
                document_id=document_id,
                chunk_index=i,
                text=chunk_text,
                embedding=embeddings[i],
                chunk_metadata=chunk_metadata,
                created_at=datetime.utcnow()
            )
            self.db.add(new_chunk)
            logger.debug(f"Added chunk index {i} for document ID {document_id}")
        self.db.commit()
        logger.info(f"Committed all chunks for document ID: {document_id}")

    def get_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by its unique identifier.

        Parameters
        ----------
        conversation_id : str
            The unique ID of the conversation.

        Returns
        -------
        Optional[Conversation]
            The Conversation instance if found; otherwise None.
        """
        logger.info(f"Fetching conversation with ID: {conversation_id}")
        conv = self.db.query(Conversation).filter_by(id=conversation_id).first()
        if conv:
            logger.info(f"Conversation found with ID: {conversation_id}")
        else:
            logger.warning(f"No conversation found with ID: {conversation_id}")
        return conv

    def create_conversation(self, conversation: Conversation) -> None:
        """
        Persist a new conversation record to the database.

        Parameters
        ----------
        conversation : Conversation
            The Conversation object to be stored.

        Returns
        -------
        None
        """
        logger.info(f"Creating new conversation with ID: {conversation.id}")
        self.db.add(conversation)
        self.db.commit()
        logger.info(f"Conversation created with ID: {conversation.id}")

    def add_message(self, message: Message) -> None:
        """
        Persist a new chat message to the database.

        Parameters
        ----------
        message : Message
            The Message object to be stored.

        Returns
        -------
        None
        """
        logger.info(f"Adding message with ID: {message.id} to conversation ID: {message.conversation_id}")
        self.db.add(message)
        self.db.commit()
        logger.info(f"Message added with ID: {message.id}")

    def get_messages_by_conversation(self, conversation_id: str) -> List[Message]:
        """
        Retrieve all messages associated with a specific conversation,
        ordered chronologically by creation time.

        Parameters
        ----------
        conversation_id : str
            The unique ID of the conversation.

        Returns
        -------
        List[Message]
            A list of Message objects for the specified conversation.
        """
        logger.info(f"Fetching messages for conversation ID: {conversation_id}")
        messages = (
            self.db.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at)
            .all()
        )
        logger.info(f"Fetched {len(messages)} messages for conversation ID: {conversation_id}")
        return messages
    
    def document_exists(self, name: str) -> bool:
        """
        Query the database to check if a document with the given name exists.

        Parameters
        ----------
        name : str
            Name of the document.

        Returns
        -------
        bool
            True if document exists, False otherwise.
        """
        return self.db.query(Document).filter(Document.name == name).first() is not None
