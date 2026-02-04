# endee-code-rag

Project: AI Codebase Intelligence using Endee as the vector database.

## Overview

endee-code-rag is an AI/ML project that indexes a GitHub repository’s source code into a vector database (Endee) and enables Retrieval-Augmented Generation (RAG) over the codebase.
Users can ask natural-language questions about a repository and receive context-aware answers based on the actual code.endee-code-rag provides a small pipeline to index a GitHub repository's code files (.py, .js, .java) into a vector store (Endee) and run Retrieval-Augmented Generation (RAG) using OpenAI to answer code questions.

## Problem Statement

Understanding large codebases is time-consuming.
Developers often need to:
- Locate where a feature is implemented.
- Understand how a function works.
- Trace configuration or database logic.

Traditional keyword search is inefficient for this.
We need semantic search + LLM-based reasoning over code.

## Solution

This project provides:
- Cloning of a GitHub repository
- Extraction of source code files (.py, .js, .java)
- Chunking of code using RecursiveCharacterTextSplitter
- Embedding generation using OpenAI
- Storage and retrieval of embeddings using Endee vector database
- Retrieval-Augmented Generation (RAG) to answer questions about the code

## Architecture

- `ingest.py`: clones the repo, extracts files, chunks them, and stores into Endee via `endee_store.py`.
- `endee_store.py`: wrapper that uses Endee REST APIs when configured, otherwise persists an in-memory store under `repo/endee_local_store.json`. Embeddings are generated via OpenAI.
- `query.py`: retrieves top-3 chunks and builds a prompt for OpenAI Chat Completion.
- `app.py`: Streamlit UI to index a repo and ask questions.

## Practical Use Case

This project demonstrates a RAG (Retrieval Augmented Generation) workflow for:
- Semantic search over code
- AI-assisted code understanding
- Repository Q&A
- Vector search is the core component of this system.


## Forked Endee Repository

This project uses a forked version of Endee as the vector database backend.
Forked repository:
https://github.com/khushipriyadarshni/endee

Endee is used to:
- Store embeddings of code chunks
- Perform similarity search for retrieval
- Act as the vector database in the RAG pipeline
  
This satisfies the requirement:
  “Fork the repository and start using it.”

## How Endee is used

Set `ENDEE_URL` and `ENDEE_API_KEY` environment variables to point to your Endee instance. The code will attempt to call the following REST endpoints:

- `POST {ENDEE_URL}/vectors/upsert` to insert vectors
- `POST {ENDEE_URL}/vectors/query` to query vectors

If Endee is not configured, the project falls back to a local persisted store at `repo/endee_local_store.json`.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file or set environment variables in your shell:

```
OPENAI_API_KEY=sk-...
# optional Endee settings
ENDEE_URL=https://example-endee.local
ENDEE_API_KEY=...
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```
Open in browser:
```bash
http://localhost:8501
```

5. In the app, provide a GitHub repo URL and click `Index Repo`. Then ask questions about the code.

## Example queries

- "How does authentication work in this repo?"
- "Where is the database connection initialized?"
- "What's the purpose of function X in file Y?"

## Notes

- This project uses OpenAI for embeddings and completions; API costs apply.
- The Endee REST contract used here is a simple upsert/query shape — adapt paths/payload keys if your Endee endpoint differs.
