# Medical Chatbot - Configuration Guide

## Overview
This guide covers advanced configuration options for the Medical Chatbot.

---

## Environment Variables

### Required
- `GROQ_API_KEY` - Your Groq API key for LLM access
  - Get from: https://console.groq.com

### Optional
```bash
# For Streamlit Cloud deployment
STREAMLIT_SERVER_HEADLESS=true

# Custom model (default: llama-3.1-8b-instant)
LLM_MODEL=llama-3.2-3b-preview

# Web search timeout (default: 10 seconds)
WEB_SEARCH_TIMEOUT=15
```

---

## Configuration Files

### .env (Local Development)
```bash
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# Optional
LLM_TEMPERATURE=0.5
MAX_TOKENS=512
WEB_SEARCH_ENABLED=true
```

### .streamlit/secrets.toml (Cloud Deployment)
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxx"
```

### pyproject.toml
- Project metadata
- Dependency specifications
- Python version requirements (3.11+)

---

## LLM Models

### Available Groq Models
Use these in the code:

```python
# Fast & Efficient
GROQ_MODEL_NAME = "llama-3.2-3b-preview"

# Balanced (Default)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# More Capable (Slower)
GROQ_MODEL_NAME = "llama-3.1-70b-versatile"
```

### Customizing Model in medibot.py
```python
# Line ~153 in medibot.py
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change this
```

---

## Vector Store Configuration

### Location
- Path: `vectorstore/db_faiss/`
- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Dimension: 384

### Recreating Vector Store
```bash
python create_memory_for_llm.py
```

### Customizing Vector Store Creation
Edit `create_memory_for_llm.py`:

```python
# Chunk size (default: 500)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Smaller = more chunks
    chunk_overlap=50     # Overlap for context
)

# Model (default: all-MiniLM-L6-v2)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## Web Search Configuration

### Customize Web Search
Edit `medibot.py`:

```python
def perform_web_search(query: str, search_tool: DDGS) -> Optional[str]:
    search_query = f"{query} medical information healthcare"  # Customize
    results = search_tool.text(search_query, max_results=5)    # Change max_results
```

### Trigger Web Search
Currently triggers on:
- Limited local results
- HL7 queries
- "standard" queries

Modify in `medibot.py` (~line 177):
```python
if not has_good_local_results or "hl7" in prompt.lower() or "standard" in prompt.lower():
    # Also add custom triggers here
    if "your_keyword" in prompt.lower():
        web_results = perform_web_search(prompt, web_search_tool)
```

---

## Streamlit UI Configuration

### Custom CSS Styling
Edit the CSS in `medibot.py` (~line 57-73) to customize:
- Colors
- Fonts
- Chat bubble appearance
- Button styles

Example:
```python
st.markdown(
    """
    <style>
    .stChatMessage {
        background: rgba(255,255,255,0.07) !important;
        padding: 1em !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
```

### Port Configuration
Run on custom port:
```bash
streamlit run medibot.py --server.port 8502
```

### Server Configuration
Create `~/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2a5298"
backgroundColor = "#1e3c72"

[server]
headless = true
port = 8501

[logger]
level = "info"
```

---

## Data Management

### Adding New Medical Documents
1. Place PDF files in `data/` folder
2. Run: `python create_memory_for_llm.py`
3. Restart the bot

### Supported Formats
- `.pdf` files (via PyPDF)
- Text extraction from documents

### Directory Structure
```
data/
├── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── additional_medical_resource.pdf  (optional)
└── more_content.pdf  (optional)
```

---

## Performance Tuning

### Vector Retrieval
```python
# In medibot.py, adjust retriever parameters
rag_chain = create_retrieval_chain(
    vectorstore.as_retriever(
        search_kwargs={'k': 3}  # Number of results (default: 3)
    ),
    combine_docs_chain
)
```

### LLM Parameters
```python
llm = ChatGroq(
    model_name=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.5,        # Lower = more deterministic
    max_tokens=512         # Higher = longer responses
)
```

---

## Troubleshooting

### High Latency
- Reduce `max_tokens` in LLM config
- Use smaller model: `llama-3.2-3b-preview`
- Disable web search for faster responses

### Web Search Not Working
- Check internet connection
- Verify DDGS can reach DuckDuckGo
- Try with VPN if blocked

### Out of Memory
- Reduce chunk size in vector store creation
- Use smaller embedding model
- Process smaller PDF files

### API Rate Limits
- Add delays between requests
- Use smaller model (3B instead of 70B)
- Check Groq quota

---

## Security Best Practices

1. **Never commit .env files**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Rotate API keys periodically**
   - Delete old keys on Groq console
   - Create new ones

3. **Use environment variables in production**
   - Don't hardcode secrets
   - Use managed secrets services

4. **Limit API access**
   - Restrict Groq API key to your IP
   - Set usage limits

---

## Monitoring

### Enable Debug Logging
Edit `medibot.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitor API Usage
Visit: https://console.groq.com/usage

### Chat Logs
Logs are printed to console when running:
```bash
streamlit run medibot.py
```

---

## Advanced Features

### Custom Prompt Templates
Modify `CUSTOM_PROMPT_TEMPLATE` in `medibot.py`:

```python
CUSTOM_PROMPT_TEMPLATE = """
You are a specialized [SPECIALTY] medical assistant...
Context: {context}
Question: {question}
Respond with [FORMAT]...
"""
```

### Multi-turn Conversation
Already implemented via `st.session_state.messages`

### Custom Document Processing
Modify `create_memory_for_llm.py`:
```python
# Add custom text processing
def preprocess_text(text):
    # Your custom logic here
    return processed_text
```

---

## Deployment

### Local
```bash
streamlit run medibot.py
```

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "medibot.py"]
```

### Streamlit Cloud
1. Push to GitHub
2. Connect on streamlit.app
3. Add secrets in Streamlit dashboard
4. Deploy

---

## Support & Resources

- **Groq**: https://console.groq.com/docs
- **LangChain**: https://python.langchain.com
- **Streamlit**: https://docs.streamlit.io
- **DDGS**: https://github.com/deedy5/duckduckgo_search

