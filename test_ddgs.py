#!/usr/bin/env python3
"""Test script to verify ddgs web search functionality for HL7 queries"""

import sys
sys.path.insert(0, '/Users/abhinav/Documents/medical-chatbot')

from dotenv import load_dotenv
load_dotenv()

# Test imports
try:
    from ddgs import DDGS
    print("✓ DDGS imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test web search tool
print("\n--- Testing DDGS Web Search Tool ---")
try:
    ddgs = DDGS()
    print("✓ DDGS initialized")
    
    # Test HL7 search
    query = "HL7 healthcare standard medical"
    print(f"\nSearching for: '{query}'")
    results = ddgs.text(query, max_results=5)
    
    if results:
        print(f"\n✓ Web search returned {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Title: {result['title']}")
            print(f"  Body: {result['body'][:200]}...")
    else:
        print("✗ No results found")
        
except Exception as e:
    print(f"✗ Web search error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---")
