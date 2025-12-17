# RAG URL Answer Bot

A Retrieval-Augmented Generation (RAG) application that ingests web URLs,
indexes their content, and answers user queries using an LLM.

## What this project does
- Loads content from one or more URLs
- Splits text into chunks
- Generates embeddings
- Stores vectors in a vector database
- Retrieves relevant context for a query
- Produces grounded answers using an LLM