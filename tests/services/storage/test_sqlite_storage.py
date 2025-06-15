# tests/services/storage/test_sqlite_storage.py

import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Chunk, Conversation, Message
from app.services.storage.sqlite_storage import SQLiteStorage


# ---- Fixtures ----

@pytest.fixture()
def db_session():
    """Creates a fresh in-memory SQLite database per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()


@pytest.fixture
def storage(db_session):
    """Returns a SQLiteStorage with an injected session."""
    class TestSQLiteStorage(SQLiteStorage):
        def __init__(self, session):
            self.db = session

    return TestSQLiteStorage(db_session)


# ---- Tests ----

def test_store_document(storage):
    doc = storage.store_document(
        name="test.txt",
        document_metadata={"source": "unit"},
        path="/path/to/test.txt"
    )
    assert doc.id is not None
    assert doc.name == "test.txt"
    assert doc.path == "/path/to/test.txt"


def test_store_chunks(storage):
    doc = storage.store_document("doc.txt", {}, "/path/to/doc.txt")

    chunks = [
        {"text": "chunk one", "metadata": {"page": 1}},
        {"text": "chunk two", "metadata": {"page": 2}}
    ]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    storage.store_chunks(doc.id, chunks, embeddings)

    # ✅ Directly query the known Chunk model
    stored = storage.db.query(Chunk).filter_by(document_id=doc.id).all()

    assert len(stored) == 2
    assert stored[0].text == "chunk one"
    assert stored[1].embedding == [0.4, 0.5, 0.6]



def test_create_and_get_conversation(storage):
    conv_id = str(uuid.uuid4())
    convo = Conversation(id=conv_id, created_at=datetime.utcnow())
    storage.create_conversation(convo)

    fetched = storage.get_conversation_by_id(conv_id)
    assert fetched is not None
    assert fetched.id == conv_id


def test_add_and_get_messages(storage):
    conv_id = str(uuid.uuid4())
    convo = Conversation(id=conv_id, created_at=datetime.utcnow())
    storage.create_conversation(convo)

    msg1 = Message(
        conversation_id=conv_id,
        role="user",
        content="Hi there!",
        created_at=datetime.utcnow()
    )
    msg2 = Message(
        conversation_id=conv_id,
        role="assistant",
        content="Hello!",
        created_at=datetime.utcnow()
    )

    storage.add_message(msg1)
    storage.add_message(msg2)

    messages = storage.get_messages_by_conversation(conv_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_document_exists(storage):
    doc_name = "existing_doc.txt"
    storage.store_document(name=doc_name, document_metadata={}, path="/some/path")

    # Act & Assert
    assert storage.document_exists(doc_name) is True
    assert storage.document_exists("non_existing_doc.txt") is False