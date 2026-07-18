#!/usr/bin/env python3
"""Test script to verify web search functionality for HL7 queries"""

import os
import sys
sys.path.insert(0, '/Users/abhinav/Documents/medical-chatbot')

from dotenv import load_dotenv
load_dotenv()

# Test imports
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test web search tool
print("\n--- Testing Web Search Tool ---")
try:
    search_tool = DuckDuckGoSearchRun()
    print("✓ DuckDuckGoSearchRun initialized")
    
    # Test HL7 search
    print("\nSearching for: 'HL7 medical information'")
    results = search_tool.run("HL7 medical information")
    print(f"\n✓ Web search result preview:")
    print(results[:500] + "..." if len(results) > 500 else results)
    
except Exception as e:
    print(f"✗ Web search error: {e}")
    import traceback
    traceback.print_exc()

# Test local vector store
print("\n--- Testing Local Vector Store ---")
try:
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✓ FAISS vector store loaded")
    
    # Test similarity search for HL7
    print("\nSearching local DB for: 'HL7'")
    docs = db.similarity_search("HL7", k=3)
    if docs:
        print(f"✓ Found {len(docs)} results in local database")
        for i, doc in enumerate(docs, 1):
            print(f"\nResult {i}:")
            print(doc.page_content[:200] + "...")
    else:
        print("✗ No results found in local database for 'HL7'")
        
except Exception as e:
    print(f"✗ Vector store error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---")
