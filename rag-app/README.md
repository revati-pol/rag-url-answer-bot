# RAG URL Answer Bot

A Retrieval-Augmented Generation (RAG) application that allows you to ask questions directly against the content of live web URLs.

The system ingests web pages, converts them into embeddings, stores them in a vector database, and uses an LLM to generate answers grounded in the source content.

---

## What this app does

- Loads and parses content from one or more URLs
- Splits text into semantic chunks
- Generates embeddings using Hugging Face models
- Stores embeddings in a Chroma vector database
- Uses an LLM (via Groq) to answer questions with source awareness

---

## Tech Stack

- **LangChain** – orchestration and RAG pipeline
- **Groq API** – LLM inference
- **Hugging Face Embeddings** – text vectorization
- **ChromaDB** – vector storage
- **Unstructured** – web content extraction
- **Streamlit** – user interface


