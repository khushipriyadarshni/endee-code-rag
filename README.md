# endee-code-rag

Project: AI Codebase Intelligence using Endee as the vector database.

## Overview

endee-code-rag provides a small pipeline to index a GitHub repository's code files (.py, .js, .java) into a vector store (Endee) and run Retrieval-Augmented Generation (RAG) using OpenAI to answer code questions.

## Problem Statement

Understanding large codebases is time consuming. Developers need a quick way to ask natural-language questions about code and receive precise answers using the repository's own code as context.

## Solution

This project:
- Clones a GitHub repository
- Extracts code files and chunks them using RecursiveCharacterTextSplitter
- Stores chunk embeddings and metadata in Endee (or a local fallback store)
- Retrieves top-k relevant chunks for a query and uses OpenAI to generate an answer (RAG)

## Architecture

- `ingest.py`: clones the repo, extracts files, chunks them, and stores into Endee via `endee_store.py`.
- `endee_store.py`: wrapper that uses Endee REST APIs when configured, otherwise persists an in-memory store under `repo/endee_local_store.json`. Embeddings are generated via OpenAI.
- `query.py`: retrieves top-3 chunks and builds a prompt for OpenAI Chat Completion.
- `app.py`: Streamlit UI to index a repo and ask questions.

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

5. In the app, provide a GitHub repo URL and click `Index Repo`. Then ask questions about the code.

## Example queries

- "How does authentication work in this repo?"
- "Where is the database connection initialized?"
- "What's the purpose of function X in file Y?"

## Notes

- This project uses OpenAI for embeddings and completions; API costs apply.
- The Endee REST contract used here is a simple upsert/query shape — adapt paths/payload keys if your Endee endpoint differs.
