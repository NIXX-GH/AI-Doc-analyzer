import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_store import build_vector_store, CHROMA_DB_PATH

# Load the GROQ_API_KEY from your .env file
load_dotenv()


def get_llm():
    """
    Creates and returns the Groq LLM object.
    
    We use 'llama-3.1-8b-instant' — Meta's Llama 3.1 model with 8 billion parameters.
    'instant' means Groq runs it extremely fast (usually under 1 second response).
    temperature=0 means the model gives consistent, factual answers rather than
    creative/random ones. For a document QA chatbot, you always want temperature=0.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,          # 0 = factual and consistent. 1 = creative and varied.
        max_tokens=1024,        # maximum length of the answer
    )
    return llm


def create_prompt_template():

    template = """
You are an advanced AI document analysis assistant.

Your task is to answer the user's question using ONLY the provided document context.
Your goal is to help the user truly understand the document — not just extract sentences from it.

==================================================
CORE BEHAVIOR
==================================================

- Answer naturally, intelligently, and conversationally.
- Sound like an expert tutor or research assistant.
- Do NOT sound like a search engine or keyword matcher.
- Do NOT simply copy text from the context unless quoting is necessary.
- Synthesize information across multiple context sections whenever relevant.
- Focus on meaning, intent, implications, and relationships between ideas.

==================================================
ANSWERING STRATEGY
==================================================

1. FIRST understand what the user is actually asking.
   - Determine whether the question is:
     - factual,
     - conceptual,
     - analytical,
     - comparative,
     - summarization,
     - step-by-step,
     - opinion-oriented,
     - or inference-based.

2. Then construct the response accordingly.

3. If the question is broad:
   - Give a structured answer.
   - Use sections or bullet points when useful.
   - Cover the important aspects progressively.

4. If the question is specific:
   - Give a direct answer first.
   - Then provide supporting explanation if helpful.

5. If the document contains technical concepts:
   - Explain them in simple language.
   - Define jargon naturally when needed.

6. If information is spread across multiple chunks:
   - Combine them into one coherent explanation.

==================================================
IMPORTANT REASONING RULES
==================================================

- Never invent facts that are not supported by the context.
- Do not hallucinate missing details.
- If the answer is partially available:
  - provide the available information,
  - clearly mention what is uncertain or missing.
- If multiple interpretations are possible, explain the ambiguity.
- If the context contains conflicting information, mention the conflict clearly.
- If the user asks "why", "how", or "what does this mean":
  - explain the reasoning and implications,
  - not just the literal statement.

==================================================
CONVERSATIONAL QUALITY
==================================================

- Maintain a helpful and intelligent tone.
- Avoid robotic phrases like:
  - "According to the context..."
  - "The document states..."
  - "Based on the provided excerpts..."
- Instead, answer smoothly and naturally.

Bad example:
"The context mentions that neural networks use layers."

Better example:
"Neural networks work by passing information through multiple layers, where each layer learns increasingly complex patterns."

==================================================
EDGE CASE HANDLING
==================================================

- If the question cannot be answered fully from the context:
  - say what information is available,
  - explain the limitation clearly,
  - avoid pretending to know more.

- Only say:
  "I couldn't find enough information in the document to answer that."
  if the context is genuinely irrelevant.

- If the user asks follow-up questions:
  - maintain continuity naturally.

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}

==================================================
FINAL ANSWER
==================================================
"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def build_rag_chain(pdf_path=None):
    
    if pdf_path and not os.path.exists(CHROMA_DB_PATH):
        print("Building vector store from PDF...")
        vector_store = build_vector_store(pdf_path)

    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    
    llm = get_llm()
    prompt = create_prompt_template()
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("RAG chain is ready!")
    return chain, vector_store      


def ask_question(chain, question):
    """
    Takes the chain and a question string.
    Returns a dict with:
      - 'answer'   : the LLM's answer as a string
      - 'sources'  : list of source chunks used (with page numbers)
    """
    print(f"\nQuestion: {question}")
    print("Thinking...")
    
    # chain.invoke() triggers all 4 steps:
    # retrieve → build prompt → send to LLM → return answer
    result = chain.invoke({"query": question})
    
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # Extract page numbers from source documents for citation
    pages_used = list(set([
        doc.metadata.get("page", "unknown") 
        for doc in source_docs
    ]))
    pages_used.sort()
    
    return {
        "answer": answer,
        "sources": source_docs,
        "pages": pages_used
    }


# ── TEST THE FULL PIPELINE ────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # Build the chain
    # Change "document.pdf" to whatever PDF you used in Stage 2/3
    chain = build_rag_chain(pdf_path="ds.pdf")
    
    print("\n" + "="*60)
    print("RAG CHAIN TEST — Ask 3 questions")
    print("="*60)
    
    # Test with 3 different questions about your document
    test_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What conclusions does the document reach?",
        "what is cleaning and munging ?"
    ]
    
    for question in test_questions:
        response = ask_question(chain, question)
        
        print(f"\nAnswer: {response['answer']}")
        print(f"Source pages: {response['pages']}")
        print("-" * 60)
    
    print("\nStage 4 complete! Your RAG chain is fully working.") 