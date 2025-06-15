import pytest
from unittest.mock import MagicMock
from app.services.storage.storage_service import StorageService
from app.db.models import Conversation, Message, Document


@pytest.fixture
def mock_backend():
    return MagicMock()

@pytest.fixture
def storage_service(mock_backend):
    return StorageService(backend=mock_backend)


def test_store_document_calls_backend(storage_service, mock_backend):
    storage_service.store_document("doc.txt", {"source": "user"}, "/path/doc.txt")
    mock_backend.store_document.assert_called_once_with("doc.txt", {"source": "user"}, "/path/doc.txt")


def test_store_chunks_calls_backend(storage_service, mock_backend):
    storage_service.store_chunks(1, [{"text": "chunk1"}], [[0.1, 0.2]])
    mock_backend.store_chunks.assert_called_once_with(1, [{"text": "chunk1"}], [[0.1, 0.2]])


def test_get_conversation_by_id_calls_backend(storage_service, mock_backend):
    storage_service.get_conversation_by_id("conv-123")
    mock_backend.get_conversation_by_id.assert_called_once_with("conv-123")


def test_create_conversation_calls_backend(storage_service, mock_backend):
    conversation = Conversation(id="conv-123")
    storage_service.create_conversation(conversation)
    mock_backend.create_conversation.assert_called_once_with(conversation)


def test_add_message_calls_backend(storage_service, mock_backend):
    message = Message(id="msg-1", content="Hi", role="user", conversation_id="conv-123")
    storage_service.add_message(message)
    mock_backend.add_message.assert_called_once_with(message)


def test_get_messages_by_conversation_calls_backend(storage_service, mock_backend):
    storage_service.get_messages_by_conversation("conv-123")
    mock_backend.get_messages_by_conversation.assert_called_once_with("conv-123")

def test_document_exists_calls_backend():
    mock_backend = MagicMock()
    mock_backend.document_exists.return_value = True

    service = StorageService(backend=mock_backend)
    exists = service.document_exists("test.pdf")

    mock_backend.document_exists.assert_called_once_with("test.pdf")
    assert exists is True