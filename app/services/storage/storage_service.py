import logging
from typing import List, Dict, Union, Optional
from app.services.storage.base_storage import BaseStorage
from app.db.models import Conversation, Message, Document

logger = logging.getLogger(__name__)

class StorageService:
    """
    A high-level service class that abstracts storage operations.

    This service delegates the actual persistence logic to a `BaseStorage` implementation,
    enabling plug-and-play support for various backends (e.g., SQLite, Supabase, Qdrant).
    """

    def __init__(self, backend: BaseStorage):
        """
        Initialize the StorageService with a specific backend implementation.

        Args:
            backend (BaseStorage): An instance of a storage backend implementing `BaseStorage`.
        """
        self.backend = backend
        logger.info(f"StorageService initialized with backend: {backend.__class__.__name__}")

    # ========== Document Storage ==========
    def store_document(self, name: str, document_metadata: dict, path: str):
        """
        Store a document record via the backend storage.

        Parameters
        ----------
        name : str
            The name or title of the document.
        document_metadata : dict
            Metadata dictionary associated with the document.
        path : str
            File path or URI where the document is stored.

        Returns
        -------
        Any
            Return value from the backend's store_document method.
        """
        logger.debug(f"Storing document '{name}' with metadata {document_metadata} at path '{path}'")
        result = self.backend.store_document(name, document_metadata, path)
        logger.info(f"Document '{name}' stored successfully")
        return result

    def store_chunks(
        self,
        document_id: int,
        chunks: List[Dict[str, Union[str, dict]]],
        embeddings: List[List[float]]
    ):
        """
        Store document chunks and their associated embeddings.

        Parameters
        ----------
        document_id : int
            The ID of the parent document.
        chunks : List[Dict[str, Union[str, dict]]]
            List of chunk dictionaries containing text and metadata.
        embeddings : List[List[float]]
            List of embedding vectors corresponding to each chunk.
        """
        logger.debug(f"Storing {len(chunks)} chunks for document ID {document_id}")
        self.backend.store_chunks(document_id, chunks, embeddings)
        logger.info(f"Chunks stored successfully for document ID {document_id}")

    # ========== Conversation & Messages ==========
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation record by its unique identifier.

        Parameters
        ----------
        conversation_id : str
            The unique ID of the conversation.

        Returns
        -------
        Optional[Conversation]
            The conversation object if found, otherwise None.
        """
        logger.debug(f"Fetching conversation with ID '{conversation_id}'")
        conversation = self.backend.get_conversation_by_id(conversation_id)
        if conversation:
            logger.info(f"Conversation '{conversation_id}' retrieved successfully")
        else:
            logger.warning(f"Conversation '{conversation_id}' not found")
        return conversation

    def create_conversation(self, conversation: Conversation) -> None:
        """
        Persist a new conversation record.

        Parameters
        ----------
        conversation : Conversation
            The conversation object to be stored.
        """
        logger.debug(f"Creating conversation with ID '{conversation.id}'")
        self.backend.create_conversation(conversation)
        logger.info(f"Conversation '{conversation.id}' created successfully")

    def add_message(self, message: Message) -> None:
        """
        Add a message to a conversation.

        Parameters
        ----------
        message : Message
            The message object to add.
        """
        logger.debug(f"Adding message with ID '{message.id}' to conversation '{message.conversation_id}'")
        self.backend.add_message(message)
        logger.info(f"Message '{message.id}' added successfully to conversation '{message.conversation_id}'")

    def get_messages_by_conversation(self, conversation_id: str) -> List[Message]:
        """
        Retrieve all messages for a given conversation.

        Parameters
        ----------
        conversation_id : str
            The unique ID of the conversation.

        Returns
        -------
        List[Message]
            List of message objects associated with the conversation.
        """
        logger.debug(f"Fetching messages for conversation ID '{conversation_id}'")
        messages = self.backend.get_messages_by_conversation(conversation_id)
        logger.info(f"Retrieved {len(messages)} messages for conversation '{conversation_id}'")
        return messages
    
    def document_exists(self, name: str) -> bool:
        """
        Check if a document with the given name exists in the storage backend.

        Parameters
        ----------
        name : str
            The name of the document to check for existence.

        Returns
        -------
        bool
            True if the document exists, False otherwise.
        """
        logger.debug(f"Checking if document with name '{name}' exists in storage backend.")
        try:
            exists = self.backend.document_exists(name)
            if exists:
                logger.info(f"Document '{name}' already exists in storage.")
            else:
                logger.info(f"Document '{name}' does not exist in storage.")
            return exists
        except Exception as e:
            logger.exception(f"Failed to check document existence for '{name}': {e}")
            raise