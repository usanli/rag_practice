<!-- ebefd694-232e-404b-9ba6-1418eda41f51 ee4d00ad-fbd7-412e-87db-1120ca2767ef -->
# RAG Chat Application with Streamlit

## Overview

Create a Python application with Streamlit interface that processes documents (PDF, TXT, DOCX), embeds them using OpenAI API, stores vectors in Pinecone cloud, and provides a chat interface for RAG-based question answering.

## Implementation Steps

### 1. Project Setup

- Create `requirements.txt` with dependencies:
- streamlit
- openai
- pinecone-client
- python-dotenv
- pypdf2
- python-docx
- Create `.env.example` template for API keys
- Create `.gitignore` to exclude `.env` file

### 2. Document Processing Module (`document_processor.py`)

- Implement text extraction functions:
- `extract_text_from_pdf()`: Handle PDF files using PyPDF2
- `extract_text_from_txt()`: Handle plain text files
- `extract_text_from_docx()`: Handle Word documents using python-docx
- Implement text chunking function to split documents into manageable pieces (e.g., 500-1000 tokens per chunk with overlap)

### 3. Vector Store Module (`vector_store.py`)

- Initialize Pinecone client with API key from .env
- Create or connect to Pinecone index
- Implement functions:
- `embed_and_store()`: Generate embeddings using OpenAI and store in Pinecone
- `query_vectors()`: Search similar chunks from Pinecone

### 4. Main Streamlit Application (`app.py`)

- **Sidebar**: 
- Document upload section with file uploader (accept PDF, TXT, DOCX)
- Display uploaded document count
- Process and embed button
- **Main Area**:
- Chat interface with message history
- Text input for user questions
- Display RAG responses with sources
- Implement RAG pipeline:
- Embed user query
- Retrieve relevant chunks from Pinecone
- Create context from retrieved chunks
- Send to OpenAI with context for answer generation

### 5. Configuration and README

- Create `config.py` for settings (chunk size, embedding model, etc.)
- Create `README.md` with:
- Setup instructions
- How to add API keys to .env
- How to run the application
- Usage guide

## Key Technical Decisions

- Use OpenAI's `text-embedding-3-small` model for embeddings (cost-effective)
- Use GPT-4 or GPT-3.5-turbo for chat completions
- Chunk size: ~500-1000 characters with 100 character overlap
- Pinecone index dimension: 1536 (matches OpenAI embedding size)
- Store metadata with vectors (filename, chunk index, original text)

### To-dos

- [ ] Create project structure with requirements.txt, .env.example, and .gitignore
- [ ] Implement document_processor.py with text extraction for PDF, TXT, DOCX and chunking logic
- [ ] Implement vector_store.py with Pinecone initialization and embedding/query functions
- [ ] Create app.py with Streamlit UI including document upload, processing, and chat interface
- [ ] Create README.md with setup and usage instructions