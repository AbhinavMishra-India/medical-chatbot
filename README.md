# Medical Chatbot

## Links
- **Live Demo:** [Streamlit App](https://abhinavmishra.streamlit.app/)
- **GitHub Repository:** [github.com/AbhinavMishra-India/medical-chatbot](https://github.com/AbhinavMishra-India/medical-chatbot)

An intelligent medical chatbot that leverages LLMs (Large Language Models), vector search, and web search to answer comprehensive medical queries using information from the Gale Encyclopedia of Medicine and real-time web sources.

## ⚠️ Disclaimer
This chatbot uses curated and reliable medical data.
Current coverage is focused and may not include all diseases.
The system is continuously being improved to expand its knowledge base and accuracy.

## ✨ Features
- **Natural Language Medical Q&A** - Ask questions in plain English
- **Hybrid Retrieval System** - Combines local medical database with web search
- **HL7 & Healthcare Standards Support** - Answers questions about healthcare interoperability standards
- **LLM Integration** - Uses LangChain and HuggingFace for advanced language understanding
- **Vector Search** - FAISS-powered semantic search through medical content
- **Web Search Fallback** - DDGS integration for current information on topics not in local database
- **Source Attribution** - Clearly indicates whether answers come from local or web sources
- **Streamlit Interface** - Clean, user-friendly web interface
- **Multi-turn Conversations** - Maintains chat history within sessions

## Project Structure
- `medibot.py` - Main Streamlit app with hybrid retrieval and web search
- `create_memory_for_llm.py` - Processes medical PDFs into vector embeddings
- `connect_memory_with_llm.py` - Standalone script for LLM + vector store integration testing
- `data/` - Medical reference materials (PDF files)
- `vectorstore/db_faiss/` - FAISS vector database files
- `requirements.txt` - Python package dependencies
- `pyproject.toml` - Project configuration and dependencies


## Environment Variables & Secrets Management

This project uses API keys and secrets for LLM access. These should not be committed to version control. Use the following files:

- `.env` - For local development, store environment variables (e.g., `GROQ_API_KEY`) here. This file is ignored by git.
- `.streamlit/secrets.toml` - For Streamlit Cloud deployment, add secrets (e.g., `GROQ_API_KEY`) here. This file is also ignored by git.

**Never share or commit your API keys or secrets.**

### Required Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- pip or pipenv
- GROQ API Key (get it from [console.groq.com](https://console.groq.com))

### Step 1: Clone or Navigate to Project
```bash
cd /Users/abhinav/Documents/medical-chatbot
```

### Step 2: Set Up Virtual Environment
If using venv:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

If using pipenv:
```bash
pipenv install
pipenv shell
```

### Step 3: Install Dependencies
Using pip:
```bash
pip install -r requirements.txt
```

Or using pipenv:
```bash
pipenv install
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```bash
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
EOF
```

### Step 5: (Optional) Create/Update Vector Database
If you have new medical PDF files to add, run:
```bash
python create_memory_for_llm.py
```

### Step 6: Launch the Chatbot
```bash
streamlit run medibot.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💻 Running Commands by Environment

### **On macOS/Linux:**

**With Virtual Environment (venv):**
```bash
# Activate environment
source /Users/abhinav/Documents/medical-chatbot/.venv/bin/activate

# Run chatbot
streamlit run medibot.py

# Or run in one command
/Users/abhinav/Documents/medical-chatbot/.venv/bin/python3 -m streamlit run medibot.py
```

**With Pipenv:**
```bash
cd /Users/abhinav/Documents/medical-chatbot
pipenv run streamlit run medibot.py
```

### **On Windows:**

**With Virtual Environment (venv):**
```bash
# Activate environment
.venv\Scripts\activate

# Run chatbot
streamlit run medibot.py
```

**With Pipenv:**
```bash
cd C:\path\to\medical-chatbot
pipenv run streamlit run medibot.py
```

---

## 🔧 Development & Testing

### Test Web Search Integration
```bash
/Users/abhinav/Documents/medical-chatbot/.venv/bin/python3 test_ddgs.py
```

### Test Complete Flow (Local + Web Search)
```bash
/Users/abhinav/Documents/medical-chatbot/.venv/bin/python3 test_complete_flow.py
```

### Test LLM + Vector Store Connection
```bash
python connect_memory_with_llm.py
```

---

## 📖 How the Hybrid Retrieval Works

1. **User Input** → User asks a medical question
2. **Local Search** → Bot searches the FAISS vector database
3. **Smart Decision**:
   - If good results found → Uses local context
   - If HL7/standards query → Triggers web search
   - If limited results → Tries web search as fallback
4. **Web Search** (if needed) → DDGS searches current web information
5. **LLM Processing** → Combines context and generates response
6. **Source Attribution** → Clearly marks information source
7. **Response** → Returns answer with source transparency

---

## 🎯 Example Queries

### Local Database (Works Best)
- "What is diabetes?"
- "Explain heart disease symptoms"
- "How is pneumonia treated?"

### Web Search (Enhanced)
- "What is HL7?"
- "Healthcare interoperability standards"
- "FHIR medical standard"
- "Latest COVID-19 treatments"
- "Recent medical breakthroughs"

> **Disclaimer:** Web search results are used to improve coverage for topics not present in the local medical database. Internet content may change and may not always be fully verified, so please confirm any medical advice with trusted sources or a healthcare professional.

---

## 📋 Troubleshooting

### Error: "GROQ_API_KEY not found"
**Solution:** Create a `.env` file with your GROQ API key:
```bash
echo "GROQ_API_KEY=your_key" > .env
```

### Error: "Vector store not found"
**Solution:** Run the vector database creation script:
```bash
python create_memory_for_llm.py
```

### Error: "Web search unavailable"
**Solution:** This is non-critical. The bot will still use local database. Check internet connection.

### Streamlit Port Already in Use
**Solution:** Run on a different port:
```bash
streamlit run medibot.py --server.port 8502
```

---

## 📚 Dependencies

Key packages:
- `langchain` - LLM orchestration
- `langchain-groq` - GROQ API integration
- `faiss-cpu` - Vector search
- `sentence-transformers` - Embeddings
- `ddgs` - Web search (DuckDuckGo)
- `streamlit` - Web interface
- `pypdf` - PDF processing

See `requirements.txt` for full list.

---

## 📝 Usage Tips

1. **For Disease Information** - Ask about specific diseases, symptoms, treatments
2. **For Standards** - Ask about HL7, FHIR, healthcare standards
3. **For Current Info** - Ask about recent developments (bot will search web)
4. **Multi-turn Chat** - Click "🆕 New Chat" to start fresh conversation

---

## ⚖️ License & Disclaimer
This project is for educational purposes. Please consult a licensed medical professional for real medical advice. The information provided by this chatbot should not be used as a substitute for professional medical consultation.

