#!/usr/bin/env python3
"""Test the complete RAG + web search flow for HL7 queries"""

import os
import sys
sys.path.insert(0, '/Users/abhinav/Documents/medical-chatbot')

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from ddgs import DDGS

# Initialize components
print("Initializing components...")
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
print("✓ Vector store loaded")

# Initialize LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("✗ GROQ_API_KEY not found in environment")
    sys.exit(1)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.5,
    max_tokens=512
)
print("✓ LLM initialized")

# Test queries
test_queries = [
    "What is HL7?",
    "Tell me about HL7 healthcare standard",
    "How does HL7 work in medical systems?"
]

print("\n" + "="*60)
print("TESTING HL7 QUERIES WITH HYBRID RETRIEVAL")
print("="*60)

for query in test_queries:
    print(f"\n--- Query: {query} ---")
    
    # Step 1: Try local search
    print("\n1. Searching local vector database...")
    local_docs = db.similarity_search(query, k=3)
    
    if local_docs:
        print(f"   ✓ Found {len(local_docs)} results")
        for i, doc in enumerate(local_docs, 1):
            print(f"   Result {i}: {doc.page_content[:100]}...")
    else:
        print("   ✗ No local results")
    
    # Step 2: Web search for HL7
    print("\n2. Performing web search...")
    ddgs = DDGS()
    web_results = ddgs.text(f"{query} medical information healthcare", max_results=3)
    
    if web_results:
        print(f"   ✓ Found {len(web_results)} web results")
        for i, result in enumerate(web_results, 1):
            print(f"   Result {i}: {result['title'][:80]}")
    else:
        print("   ✗ No web results")
    
    # Step 3: Combine results
    print("\n3. Combining results for LLM...")
    context_text = "Local Database Results:\n"
    if local_docs:
        for i, doc in enumerate(local_docs, 1):
            context_text += f"\n{i}. {doc.page_content[:200]}\n"
    else:
        context_text += "No local results.\n"
    
    context_text += "\n\nWeb Search Results:\n"
    if web_results:
        for i, result in enumerate(web_results, 1):
            context_text += f"\n{i}. {result['title']}: {result['body'][:200]}\n"
    else:
        context_text += "No web results.\n"
    
    print(f"   Total context length: {len(context_text)} characters")
    
    # Step 4: Generate response with LLM
    print("\n4. Generating LLM response...")
    prompt_template = f"""You are a medical chatbot assistant. Use the provided context to answer medical questions.
    
Context:
{context_text}

Question: {query}

Provide a comprehensive answer based on the context above."""
    
    try:
        response = llm.invoke(prompt_template)
        print(f"   ✓ Response generated:")
        print(f"\n   {response.content[:300]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
