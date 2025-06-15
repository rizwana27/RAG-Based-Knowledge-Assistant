# RAG Pipeline Implementation Challenge

## Overview

At Docsum, we're building an AI-powered document analysis platform that helps users extract insights and answer questions from their document repositories. Your challenge is to implement a Retrieval-Augmented Generation (RAG) pipeline that enables conversational AI interactions with a knowledge base of documents.

## Background

In our current architecture, documents go through several processing steps:

1. Document upload and parsing
2. Text extraction and chunking
3. Embedding generation
4. Storage in vector database
5. RAG pipeline for retrieving relevant context and generating responses

## Requirements

### Core Requirements

1. Design and implement a RAG pipeline that:
   - Takes a user query as input
   - Searches for relevant document chunks
   - Uses an LLM to generate a coherent answer based on the retrieved context
   - Returns comprehensive, accurate responses that cite sources from the documents

2. Implement appropriate API endpoints that:
   - Allow for chat functionality (send questions, receive answers with sources)
   - Support conversation history (maintaining context across multiple queries)
   - Handle error cases gracefully

3. Include appropriate validation, error handling, and logging

4. For this challenge, the API consumer should assume that their documents are already uploaded and processed. The API does not need to support document upload as an endpoint. On startup, load and process the documents from the `sample_data` folder into the database.

### Technical Constraints

- Use Python for backend implementation.
- Implement RESTful API endpoints using FastAPI.
- Follow clean code principles and provide proper documentation. It should be clear how the API should be used without reading the code.
- Must be compatible with OpenAI's API for LLM integration.
- No frontend is required.
- **No RAG Frameworks**: Implement the RAG pipeline from scratch without using frameworks like LangChain, LlamaIndex, or Haystack.
- **Database Integration**: Use either SQLite or Supabase to store documents, chunks, and metadata.
- **Vector Storage**: Implement vector storage and similarity search using either SQLite with vector extensions or a simple in-memory solution.

### Bonus Objectives

- **Database Performance**: Implement database indexing and query optimization for better performance.
- **Advanced Vector Search**: Implement re-ranking strategies to improve retrieval quality.
- **Hybrid Search**: Combine vector similarity search with traditional keyword search.
- **Metrics & Analytics**: Add performance metrics and usage analytics stored in the database.
- **MCP**: Implement an MCP server that wraps this API so it can be used by other LLM clients.

## Project Structure

We've provided a starter project with the following components:

```plaintext
/rag-pipeline-challenge
├── /app
│   ├── /api
│   │   ├── models.py
│   │   └── routes.py
│   ├── /db
│   │   ├── database.py
│   │   ├── models.py
│   │   └── vector_store.py
│   ├── /services
│   │   ├── embedding_service.py
│   │   ├── storage_service.py
│   │   ├── chunking_service.py
│   │   └── rag_service.py
│   └── main.py
├── /sample_data
├── /docs
├── /tests
├── README.md
└── requirements.txt
```

## Service Interfaces

We've provided class stubs for the following services that you need to implement:

1. **Embedding Service**: Interface for generating embeddings for text chunks
2. **Chunking Service**: Interface for splitting documents into manageable chunks  
3. **Storage Service**: Interface for storing and retrieving document metadata and chunks
4. **RAG Service**: Interface for the retrieval and generation process

**Important**: All service classes contain only method signatures with `# TODO: Implement` stubs. You must implement all functionality from scratch, including:

- Database integration for persistent storage
- Actual embedding generation (you can use OpenAI's embedding API or implement your own)
- Text chunking
- Vector similarity search
- All business logic

## Your Task

Your primary tasks are to:

1. **Database Design**: Create a database schema and implement a data access layer for documents, chunks, and conversations.
2. **RAG Service**: Design and implement `rag_service.py` to handle the retrieval and generation process.
3. **Vector Storage**: Implement a vector database interface in `vector_store.py` with similarity search capabilities.
4. **API Layer**: Build the API routes in `routes.py` to expose your RAG functionality.
5. **Documentation**: Create architecture diagrams, ERD, and document your design decisions.

## Evaluation Criteria

We'll evaluate your submission based on:

1. **Functionality**: Does it work as expected and meet all requirements?
2. **Code Quality**: Is your code clean, well-structured, and maintainable?
3. **System Design**: How well have you designed the components and their interactions?
4. **Error Handling**: How robust is your solution to edge cases and failures?
5. **Documentation**: How well have you documented your code and design decisions?
6. **Testing**: Have you included appropriate tests for your implementation?

## Submission Guidelines

1. Clone this repository
2. Implement your solution
3. **Required Documentation:**
   - Include a README in the `docs` folder that explains your approach, design decisions, and setup instructions
   - **Architecture Diagram**: Create a system architecture diagram showing how components interact
   - **Entity Relationship Diagram (ERD)**: Design and include an ERD showing your database schema for documents, chunks, conversations, and any other entities
   - Provide clear code documentation with docstrings and comments
4. Create a zip file of your solution and submit it via the Google Drive link provided to you.

## Time Expectation

You have **48 hours** from the time you receive this assignment to complete and submit your solution. We recommend spending 6-8 hours on implementation and 1-2 hours on documentation.

## Questions?

If you have any questions or need clarification, please don't hesitate to reach out to us at [founders@docsum.ai](mailto:founders@docsum.ai).

Good luck!
