from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(pdf_path):
    """
    Takes a PDF file path.
    Returns a list of text chunks ready for embedding.
    """
    
    # Step 1: Load the PDF
    # PyPDFLoader reads each page of the PDF and gives us a list of Document objects.
    # Each Document has .page_content (the text) and .metadata (page number, source file).
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"  Loaded {len(pages)} pages")
    
    # Step 2: Split into chunks
    # RecursiveCharacterTextSplitter tries to split at natural boundaries:
    # first at paragraph breaks (\n\n), then at sentence ends (\n), then at words.
    # chunk_size=1600 means each chunk is at most 1600 characters.
    # chunk_overlap=600 means consecutive chunks share 600 characters — this ensures
    # that a sentence split across chunk boundaries doesn't lose context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=600,
        length_function=len,
    )
    
    chunks = splitter.split_documents(pages)
    
    print(f"  Split into {len(chunks)} chunks")
    print(f"  Example chunk:\n  '{chunks[0].page_content[:200]}...'")
    
    return chunks


# Test it directly
if __name__ == "__main__":
    chunks = load_and_split_document("ds.pdf")
    print(f"\nDone! Got {len(chunks)} chunks ready for embedding.")