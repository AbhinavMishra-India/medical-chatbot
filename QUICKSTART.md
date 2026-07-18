# Medical Chatbot - Quick Start Commands

## 🚀 Fastest Way to Get Started

### For macOS/Linux Users:

```bash
# 1. Navigate to project
cd /Users/abhinav/Documents/medical-chatbot

# 2. Activate environment
source .venv/bin/activate

# 3. Set up API key (if not already done)
echo "GROQ_API_KEY=your_api_key_here" > .env

# 4. Run the bot
streamlit run medibot.py
```

The bot will open at: **http://localhost:8501**

---

### For Windows Users:

```bash
# 1. Navigate to project
cd C:\path\to\medical-chatbot

# 2. Activate environment
.venv\Scripts\activate

# 3. Set up API key (if not already done)
echo GROQ_API_KEY=your_api_key_here > .env

# 4. Run the bot
streamlit run medibot.py
```

The bot will open at: **http://localhost:8501**

---

## 📝 One-Liner Commands

### macOS/Linux (with existing .env):
```bash
cd /Users/abhinav/Documents/medical-chatbot && source .venv/bin/activate && streamlit run medibot.py
```

### Windows (with existing .env):
```bash
cd C:\path\to\medical-chatbot && .venv\Scripts\activate && streamlit run medibot.py
```

---

## 🧪 Testing Commands

### Test Web Search (HL7 Support):
```bash
/Users/abhinav/Documents/medical-chatbot/.venv/bin/python3 test_ddgs.py
```

### Test Complete Workflow:
```bash
/Users/abhinav/Documents/medical-chatbot/.venv/bin/python3 test_complete_flow.py
```

---

## 🔑 Getting Your GROQ API Key

1. Visit: https://console.groq.com
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and paste in your `.env` file

---

## ⚡ Troubleshooting

### "Command not found: streamlit"
- Make sure virtual environment is activated: `source .venv/bin/activate`

### "GROQ_API_KEY not found"
- Create `.env` file with: `echo "GROQ_API_KEY=your_key" > .env`

### "Vector store not found"
- Run: `python create_memory_for_llm.py`

### Port 8501 already in use
- Use different port: `streamlit run medibot.py --server.port 8502`

---

## 📚 What's New (Latest Updates)

✅ **Web Search Integration** - Now answers HL7 and healthcare standard questions
✅ **Hybrid Retrieval** - Combines local database with real-time web search
✅ **Source Attribution** - Shows where information comes from
✅ **HL7 Support** - Full support for healthcare interoperability queries
✅ **Better Error Handling** - Graceful fallbacks when web search unavailable

---

## 🎯 Example Queries to Try

### Local Database (Disease Info):
- "What is diabetes?"
- "Symptoms of pneumonia"
- "How is asthma treated?"

### Web Search (Enhanced):
- "What is HL7?" ✨
- "FHIR healthcare standard"
- "Latest medical breakthroughs"
- "COVID-19 treatments 2026"

---

Happy chatting! 🏥
