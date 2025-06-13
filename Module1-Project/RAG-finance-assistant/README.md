# RAG-Based Financial Research Assistant

A Retrieval-Augmented Generation (RAG) assistant for analyzing SEC filings (10-K, 10-Q) and earnings transcripts using open-source LLMs (Mistral), LangChain, FAISS, and Streamlit/Gradio UI.

## Features

- Contextual search and Q&A over financial filings
- Handles 20+ finance use cases (revenue, risks, management analysis, etc.)
- Local vectorstore for privacy (FAISS + embeddings)
- Intuitive UI (Streamlit and Gradio options)
- Fast, accurate, and explainable

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
