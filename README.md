---
title: RAG Document Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# RAG Document Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions about it in natural language. Built using Retrieval-Augmented Generation (RAG).

## Features

- Upload any PDF and start chatting instantly
- Answers grounded strictly in the document
- Shows source page numbers for every answer
- Full conversation history with a clean chat UI
- Handles multiple PDFs seamlessly

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via Groq API |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| Frontend | Streamlit |

## Run Locally

Clone the repo and install dependencies:

    git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
    cd rag-chatbot
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Create a .env file with your Groq API key:

    GROQ_API_KEY=your_groq_api_key_here

Run the app:

    streamlit run app.py

## Project Structure

    app.py                  Main Streamlit UI
    rag_chain.py            LangChain RAG pipeline
    vector_store.py         ChromaDB embeddings and search
    document_processor.py   PDF loading and chunking
    requirements.txt        Python dependencies

## Author

Your Name