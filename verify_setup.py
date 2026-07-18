#!/usr/bin/env python3
"""
Medical Chatbot - Setup Verification Script
Run this to verify your environment is properly configured
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_mark(condition, message):
    symbol = "✓" if condition else "✗"
    status = "PASS" if condition else "FAIL"
    print(f"{symbol} [{status}] {message}")
    return condition

def main():
    print_header("MEDICAL CHATBOT - SETUP VERIFICATION")
    
    all_checks_passed = True
    
    # Check Python version
    print("1. Python Environment")
    py_version = sys.version_info
    all_checks_passed &= check_mark(
        py_version.major == 3 and py_version.minor >= 11,
        f"Python 3.11+ (Current: {py_version.major}.{py_version.minor}.{py_version.micro})"
    )
    
    # Check required directories
    print("\n2. Project Structure")
    project_dir = Path.cwd()
    all_checks_passed &= check_mark(
        (project_dir / "medibot.py").exists(),
        "medibot.py exists"
    )
    all_checks_passed &= check_mark(
        (project_dir / "data").is_dir(),
        "data/ directory exists"
    )
    all_checks_passed &= check_mark(
        (project_dir / "vectorstore").is_dir(),
        "vectorstore/ directory exists"
    )
    all_checks_passed &= check_mark(
        (project_dir / "vectorstore" / "db_faiss").is_dir(),
        "vectorstore/db_faiss/ exists"
    )
    
    # Check environment variables
    print("\n3. Environment Variables")
    env_file = Path.cwd() / ".env"
    all_checks_passed &= check_mark(
        env_file.exists(),
        ".env file exists"
    )
    
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        # Try loading from .env
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY"):
                        groq_key = line.split("=")[1].strip()
                        break
    
    all_checks_passed &= check_mark(
        bool(groq_key),
        "GROQ_API_KEY configured"
    )
    
    # Check imports
    print("\n4. Required Packages")
    
    packages = [
        ("langchain", "LangChain"),
        ("langchain_groq", "LangChain GROQ"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("faiss", "FAISS"),
        ("streamlit", "Streamlit"),
        ("ddgs", "DDGS (Web Search)"),
        ("dotenv", "python-dotenv"),
        ("pypdf", "PyPDF"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            all_checks_passed &= check_mark(True, f"{name}")
        except ImportError:
            all_checks_passed &= check_mark(False, f"{name}")
    
    # Check vector store
    print("\n5. Vector Store")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
        all_checks_passed &= check_mark(True, "Vector store loads successfully")
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Vector store error: {str(e)}")
    
    # Check web search
    print("\n6. Web Search")
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        results = ddgs.text("medical test", max_results=1)
        all_checks_passed &= check_mark(
            results is not None,
            "Web search working"
        )
    except Exception as e:
        print(f"⚠ [WARN] Web search: {str(e)}")
        print("  Note: This is non-critical. Bot will still work with local database.")
    
    # Check LLM connection
    print("\n7. LLM Connection")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                api_key=groq_key,
                temperature=0.5,
                max_tokens=512
            )
            all_checks_passed &= check_mark(True, "GROQ LLM initialized")
        except Exception as e:
            all_checks_passed &= check_mark(False, f"GROQ LLM error: {str(e)}")
    else:
        print("⚠ [SKIP] GROQ_API_KEY not found")
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    if all_checks_passed:
        print("✓ All checks passed! Your bot is ready to run.")
        print("\nTo start the bot, run:")
        print("  streamlit run medibot.py")
        print("\nThe app will open at: http://localhost:8501")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create vector store: python create_memory_for_llm.py")
        print("  3. Set GROQ_API_KEY: echo 'GROQ_API_KEY=your_key' > .env")
        print("  4. Check internet connection for web search")
        return 1

if __name__ == "__main__":
    sys.exit(main())
