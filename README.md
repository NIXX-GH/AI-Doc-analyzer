---
title: RAG Document Chatbot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---# RAG Document Chatbot 

An AI-powered chatbot that lets you upload any PDF and ask questions about it 
in natural language. Built using Retrieval-Augmented Generation (RAG) — the 
same architecture used in enterprise AI products.

![App Demo](POAICB.png)

---

## What is RAG?

Standard LLMs like ChatGPT only know what they were trained on. RAG solves this 
by letting the model *retrieve* relevant information from your own documents before 
answering — making responses accurate, grounded, and source-cited.

**Pipeline:**
PDF → chunks → embeddings → ChromaDB vector store
↓
User question → semantic search → top relevant chunks
↓
Llama  reads chunks → accurate, grounded answer

---

## Features

- Upload any PDF and start chatting instantly
- Answers grounded strictly in the document — no hallucination
- Shows source page numbers for every answer
- Full conversation history with a clean chat UI
- Handles multiple PDFs (automatically rebuilds vector store)

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via Groq API |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| Frontend | Streamlit |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/NIXX-GH/rag-chatbot.git
cd rag-chatbot
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**4. Set up your API key**

Create a `.env` file in the project root:


Get a free Groq API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure
rag-chatbot/
├── app.py                  # Streamlit UI — main entry point
├── rag_chain.py            # LangChain RAG pipeline
├── vector_store.py         # ChromaDB embeddings and search
├── document_processor.py   # PDF loading and text chunking
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed to Git)

---

## How It Works

1. **Document Processing** — PyPDFLoader reads the PDF page by page.
   RecursiveCharacterTextSplitter cuts it into 1000-character overlapping chunks.

2. **Embedding** — Each chunk is converted into a 384-dimensional vector
   using HuggingFace's `all-MiniLM-L6-v2` model, which runs locally.

3. **Vector Storage** — ChromaDB stores every chunk alongside its vector.
   This enables semantic similarity search — finding meaning, not just keywords.

4. **Retrieval** — When the user asks a question, it is also embedded into a
   vector. ChromaDB returns the top 4 chunks whose vectors are closest.

5. **Generation** — The 4 retrieved chunks are injected into a prompt template
   alongside the question. Groq's Llama 3.1 reads this and generates a grounded answer.

---

## Author

**Your Name** · [LinkedIn](https://linkedin.com/in/nikhil-samantray) · [GitHub](https://github.com/NIXX-GH)