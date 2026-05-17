import os
import sys
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from document_processor import load_and_split_document

CHROMA_DB_PATH = "./chroma_db"

# Detect platform once at import time.
# sys.platform == "win32" covers all Windows versions (32 and 64 bit).
# On HuggingFace Spaces (Linux) this is "linux" — uses disk persistence.
# In the  Windows machine — uses in-memory to avoid WinError 32.
IS_WINDOWS = sys.platform == "win32"


def get_embedding_model():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("  Embedding model loaded.")
    return embeddings


def build_vector_store(pdf_path):
  
    chunks = load_and_split_document(pdf_path)
    embeddings = get_embedding_model()

    if IS_WINDOWS:
       
       # Windows: pure in-memory, no disk, no file locks
        print("Windows detected → using in-memory ChromaDB")
        client = chromadb.EphemeralClient()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name="rag_collection"
        )

    else:
        # ─ Linux/Mac: persist to disk as normal 
        print("Linux/Mac detected → using persistent ChromaDB")
        
        # Clean up old store if it exists
        import shutil
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name="rag_collection"
        )

    print("  Vector store ready.")
    return vector_store


def close_vector_store(vector_store):
    """
    Windows:   Nothing to do — EphemeralClient has no file handles.
    Linux/Mac: Could persist here, but since we delete on next upload anyway,
               just let gc.collect() handle cleanup.
    """
    pass


def search_vector_store(vector_store, query, k=4):
    results = vector_store.similarity_search(query, k=k)
    return results