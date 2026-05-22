import os
import gc
import time 
import shutil
import tempfile
  
import streamlit as st
from dotenv import load_dotenv  
from rag_chain import build_rag_chain, ask_question
from vector_store import CHROMA_DB_PATH, close_vector_store

# Load environment variables from .env file
load_dotenv()

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# This must be the very first Streamlit command in your script.
# It sets the browser tab title, icon, and layout.
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="centered"
)

# ── INITIALISE SESSION STATE ──────────────────────────────────────────────────
# These run only when the key doesn't exist yet (i.e. first load of the app).
# After first load, Streamlit preserves these values across all reruns.

if "messages" not in st.session_state:
    # messages is a list of dicts: {"role": "user"/"assistant", "content": "..."}
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    # rag_chain holds the loaded LangChain pipeline.
    # None means no PDF has been processed yet.
    st.session_state.rag_chain = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "pdf_name" not in st.session_state:
    # Tracks which PDF is currently loaded, so we don't reload unnecessarily.
    st.session_state.pdf_name = None


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def force_delete_chroma(path, retries=8, delay=0.8):
    """
    Tries to delete the chroma_db folder multiple times with a pause
    between each attempt.

    Why this is needed on Windows:
    ChromaDB uses memory-mapped .bin files and a SQLite .db file.
    Even after calling .reset() and gc.collect(), Windows holds a
    low-level OS file handle on these files for a short time.
    shutil.rmtree() called immediately will hit WinError 32.
    Waiting 0.8 seconds between retries gives Windows enough time
    to release those handles naturally — by retry 3 or 4 it always
    succeeds.

    Returns True if deletion succeeded, False if all retries failed.
    """
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path)
            print(f"ChromaDB folder deleted on attempt {attempt}.")
            return True
        except PermissionError as e:
            print(f"Attempt {attempt}/{retries} failed — waiting {delay}s... ({e})")
            time.sleep(delay)

    return False   # all retries exhausted


def process_uploaded_pdf(uploaded_file):
    """
    Windows-proof PDF processor.
    Closes ChromaDB, forces garbage collection, then retries
    folder deletion until Windows releases the file handles.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # ── 1. Close ChromaDB and release all Python references ──────────
        # close_vector_store() calls _client.reset() which flushes
        # ChromaDB's write buffer and signals it to close file handles.
        if st.session_state.get("vector_store") is not None:
            close_vector_store(st.session_state.vector_store)
            st.session_state.vector_store = None

        # Dropping these references means Python's reference counter
        # reaches zero and those objects become eligible for collection.
        st.session_state.rag_chain = None

        # ── 2. Force immediate garbage collection ─────────────────────────
        # Without this, Python might not destroy the ChromaDB objects
        # until the next GC cycle — keeping file handles alive.
        gc.collect()

        # ── 3. Delete the folder with retries ────────────────────────────
        # Windows needs a moment after gc.collect() to fully release
        # the memory-mapped file handles. The retry loop handles this.
        if os.path.exists(CHROMA_DB_PATH):
            deleted = force_delete_chroma(CHROMA_DB_PATH)

            if not deleted:
                # All 8 retries failed — extremely rare but handle it cleanly.
                st.error(
                    "Could not delete the old ChromaDB folder after several attempts. "
                    "Please close all other programs that might be accessing it, "
                    "or manually delete the 'chroma_db' folder and try again."
                )
                return False

        # ── 4. Build fresh vector store and chain ────────────────────────
        chain, vector_store = build_rag_chain(pdf_path=tmp_path)

        st.session_state.rag_chain = chain
        st.session_state.vector_store = vector_store
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.messages = []

        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

    finally:
        os.unlink(tmp_path)

def display_chat_history():
    """
    Loops through all messages in session_state and renders them
    as chat bubbles using st.chat_message().
    Called on every rerun so the full history is always visible.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If this was an assistant message that has source pages, show them.
            if message["role"] == "assistant" and "pages" in message:
                if message["pages"]:
                    pages_str = ", ".join([str(p + 1) for p in message["pages"]])
                    st.caption(f"📄 Sources: page(s) {pages_str}")


def handle_user_input(user_question):
    """
    Called when the user submits a question.
    1. Adds user message to history.
    2. Gets answer from RAG chain.
    3. Adds assistant message to history.
    4. Streamlit reruns and display_chat_history() shows everything.
    """
    
    # Add the user's message to history immediately.
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    
    # Show a spinner while the LLM is thinking.
    with st.spinner("Searching document and thinking..."):
        response = ask_question(st.session_state.rag_chain, user_question)
    
    # Add the assistant's answer to history.
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "pages": response["pages"]
    })


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# The sidebar stays visible at all times and handles PDF uploading.

with st.sidebar:
    st.title("📄 DocAnalyzer ")
    st.markdown("Upload a PDF and ask questions about it.")
    
    st.divider()
    
    # File uploader widget.
    # type=["pdf"] restricts it to PDF files only.
    # When a file is uploaded, Streamlit reruns and uploaded_file is not None.
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Max file size: 200MB"
    )
    
    # Only process the PDF if:
    # 1. A file was actually uploaded (not None)
    # 2. It's a different file from what's already loaded
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF... this may take a minute."):
                success = process_uploaded_pdf(uploaded_file)
            
            if success:
                st.success(f"✅ '{uploaded_file.name}' is ready!")
    
    # Show which PDF is currently loaded (or a prompt to upload one).
    if st.session_state.pdf_name:
        st.info(f"📂 Active: {st.session_state.pdf_name}")
    else:
        st.warning("⬆️ Upload a PDF to begin")
    
    st.divider()
    
    # Button to clear the chat history without unloading the PDF.
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("Built with LangChain · Groq · ChromaDB · Streamlit")


# ── MAIN CHAT AREA ────────────────────────────────────────────────────────────

st.title("Ask your document anything")

# If no PDF is loaded yet, show a friendly welcome screen instead of the chat.
if st.session_state.rag_chain is None:
    st.markdown(
        """
        ### How to get started
        1. Upload a PDF using the sidebar on the left
        2. Wait for it to process (usually 30-60 seconds)
        3. Ask any question about the document
        
        ---
        **What this app does:**  
        It reads your PDF, understands the content with using AI embeddings, 
        and answers your questions based only on what's in the document.
        """
    )

else:
    # PDF is loaded — show the full chat interface.
    
    # Render all previous messages first.
    display_chat_history()
    
    # Show a helpful first message if chat is empty.
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                f"I've read **{st.session_state.pdf_name}**. "
                f"What would you like to know about it?"
            )
    
    # The chat input box — always pinned to the bottom of the page.
    # st.chat_input() returns the typed text when user presses Enter,
    # and returns None when the input is empty.
    user_input = st.chat_input("Ask a question about the document...")
    
    if user_input:
        # Process the question and update session_state.
        handle_user_input(user_input)
        # Rerun so the new messages appear immediately.
        st.rerun()