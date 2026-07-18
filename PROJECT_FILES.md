# Medical Chatbot - Project Files Guide

## 📂 Complete File Structure

```
medical-chatbot/
├── 📄 Core Application Files
│   ├── medibot.py                  # Main Streamlit app (web interface)
│   ├── create_memory_for_llm.py   # Vector database creation script
│   └── connect_memory_with_llm.py # Standalone LLM + retrieval testing
│
├── 📂 Data & Storage
│   ├── data/                       # Medical reference documents
│   │   └── disease_info.csv        # Disease reference data
│   └── vectorstore/db_faiss/       # FAISS vector database (auto-generated)
│       ├── index.faiss
│       └── workspace
│
├── 🧪 Testing & Verification
│   ├── test_ddgs.py               # Web search functionality test
│   ├── test_hl7.py                # HL7 query test (legacy)
│   ├── test_complete_flow.py      # End-to-end RAG flow test
│   └── verify_setup.py             # Environment verification script
│
├── 📚 Documentation
│   ├── README.md                   # Main documentation (START HERE!)
│   ├── QUICKSTART.md               # Quick reference guide
│   ├── CHANGELOG.md                # What's new & updates
│   ├── CONFIGURATION.md            # Advanced config options
│   └── PROJECT_FILES.md            # This file
│
├── ⚙️ Configuration Files
│   ├── requirements.txt             # Python dependencies (pip)
│   ├── pyproject.toml               # Project config & dependencies (modern)
│   └── .env                         # Environment variables (local dev)
│
└── 📋 Other
    ├── .gitignore                  # Git ignore rules
    ├── .venv/                      # Python virtual environment
    └── xl.sx.numbers               # Data file
```

---

## 📄 File Descriptions

### Core Application Files

#### **medibot.py** ⭐
- **Purpose**: Main Streamlit web application
- **Key Functions**:
  - `get_vectorstore()` - Loads FAISS database
  - `get_web_search_tool()` - Initializes DDGS
  - `perform_web_search()` - Executes web searches
  - `main()` - UI and chat logic
- **When to Edit**: Customize UI, prompts, or retrieval logic
- **Dependencies**: langchain, streamlit, ddgs, faiss
- **Run**: `streamlit run medibot.py`

#### **create_memory_for_llm.py**
- **Purpose**: Converts PDF documents into vector embeddings
- **Key Functions**:
  - `load_pdf_files()` - Reads PDFs from data/
  - `create_chunks()` - Splits text into chunks
  - `get_embedding_model()` - Initializes embeddings
- **When to Run**: When adding new medical documents
- **When to Edit**: To customize chunk size or embedding model
- **Run**: `python create_memory_for_llm.py`

#### **connect_memory_with_llm.py**
- **Purpose**: Standalone script for testing RAG pipeline
- **Use Cases**: Debug vector retrieval, test LLM responses
- **When to Run**: For troubleshooting or development
- **Run**: `python connect_memory_with_llm.py`

---

### Testing Files

#### **verify_setup.py** ✅
- **Purpose**: Validates entire setup is working
- **Checks**:
  - Python version
  - File structure
  - Dependencies installed
  - Vector store loads
  - Web search works
  - LLM connection
- **When to Run**: Before starting bot for first time
- **Run**: `python verify_setup.py`

#### **test_ddgs.py**
- **Purpose**: Tests web search functionality
- **What it Tests**: DDGS import, search execution, result parsing
- **When to Run**: To verify web search works
- **Run**: `python test_ddgs.py`

#### **test_complete_flow.py**
- **Purpose**: End-to-end test of hybrid retrieval
- **Simulates**: 
  - Local vector search
  - Web search
  - LLM response generation
- **When to Run**: To verify complete system works
- **Run**: `python test_complete_flow.py`

---

### Documentation Files

#### **README.md** 📖
- **What**: Complete project documentation
- **Contains**:
  - Features overview
  - Setup instructions
  - Running commands (macOS/Linux/Windows)
  - Troubleshooting guide
  - Example queries
  - Dependencies list
- **When to Read**: First time setup, for comprehensive info

#### **QUICKSTART.md** ⚡
- **What**: Quick reference guide
- **Contains**:
  - One-liner commands
  - Platform-specific commands
  - API key setup
  - Common errors & fixes
- **When to Read**: For quick command reference

#### **CHANGELOG.md** 📝
- **What**: Version history and updates
- **Contains**:
  - New features
  - Bug fixes
  - Technical improvements
  - How to update
- **When to Read**: To understand what changed

#### **CONFIGURATION.md** 🔧
- **What**: Advanced configuration guide
- **Contains**:
  - Environment variables
  - Model customization
  - Performance tuning
  - Deployment options
  - Security best practices
- **When to Read**: For advanced customization

#### **PROJECT_FILES.md**
- **What**: This file
- **Contains**: File structure and descriptions
- **When to Read**: To understand project layout

---

### Configuration Files

#### **requirements.txt**
- **Purpose**: Python package dependencies (pip format)
- **Usage**: `pip install -r requirements.txt`
- **When to Update**: Adding new dependencies

#### **pyproject.toml**
- **Purpose**: Modern Python project configuration
- **Contains**: 
  - Project metadata
  - Dependency specifications
  - Python version requirements
- **Usage**: `pipenv install` (uses this file)
- **When to Update**: For modern Python projects

#### **.env**
- **Purpose**: Local environment variables
- **Contains**: `GROQ_API_KEY=your_key_here`
- **Security**: NEVER commit to git (in .gitignore)
- **When to Create**: Before first run
- **Create**: `echo "GROQ_API_KEY=your_key" > .env`

#### **.gitignore**
- **Purpose**: Tells git which files to ignore
- **Includes**: .env, .venv, __pycache__, etc.
- **When to Edit**: If adding new non-tracked files

---

### Data & Storage

#### **data/ Directory**
- **Purpose**: Stores medical reference documents
- **Contents**:
  - `disease_info.csv` - Disease information
  - `The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf` (if added)
  - Other PDFs (optional)
- **Size**: Can be large with many documents
- **Processing**: Run `create_memory_for_llm.py` after adding files

#### **vectorstore/db_faiss/ Directory**
- **Purpose**: Stores generated vector database
- **Auto-Generated**: By `create_memory_for_llm.py`
- **Do Not Edit Manually**: Delete and regenerate if corrupted
- **Size**: Scales with amount of documents

---

## 🔄 File Dependencies

```
medibot.py
├── Imports: langchain, streamlit, ddgs
├── Loads: vectorstore/db_faiss
├── Uses: create_memory_for_llm.py (for setup)
└── Needs: .env (GROQ_API_KEY)

create_memory_for_llm.py
├── Imports: langchain
├── Reads: data/*.pdf
└── Generates: vectorstore/db_faiss

connect_memory_with_llm.py
├── Imports: langchain
├── Loads: vectorstore/db_faiss
└── Needs: .env (GROQ_API_KEY)
```

---

## 📋 Quick Reference Table

| File | Type | Purpose | When to Use |
|------|------|---------|-------------|
| medibot.py | App | Main chatbot | Run bot |
| create_memory_for_llm.py | Script | Build vectors | Add documents |
| connect_memory_with_llm.py | Script | Test retrieval | Debug |
| verify_setup.py | Script | Check setup | First time |
| test_ddgs.py | Test | Test web search | Verify web search |
| test_complete_flow.py | Test | Test full flow | System test |
| README.md | Docs | Main guide | Setup |
| QUICKSTART.md | Docs | Quick ref | Commands |
| CHANGELOG.md | Docs | Updates | Version info |
| CONFIGURATION.md | Docs | Advanced | Customization |
| requirements.txt | Config | Dependencies | pip install |
| pyproject.toml | Config | Project config | pipenv install |
| .env | Config | API keys | Keep private |

---

## 🚀 Typical Workflows

### First Time Setup
1. Read: `README.md`
2. Run: `python verify_setup.py`
3. Execute: `streamlit run medibot.py`

### Adding New Documents
1. Place PDFs in `data/`
2. Run: `python create_memory_for_llm.py`
3. Restart: `streamlit run medibot.py`

### Troubleshooting
1. Check: `verify_setup.py`
2. Read: `README.md` troubleshooting
3. Test: `test_complete_flow.py`

### Advanced Customization
1. Read: `CONFIGURATION.md`
2. Edit: Specific files mentioned
3. Test: `test_complete_flow.py`

---

## 📞 Support Resources

- **Setup Issues**: See README.md troubleshooting
- **Commands**: See QUICKSTART.md
- **Configuration**: See CONFIGURATION.md
- **What Changed**: See CHANGELOG.md
- **System Check**: Run `verify_setup.py`

---

## 🎯 File Modification Guide

| File | Safe to Edit | Frequency | Impact |
|------|------|----------|--------|
| medibot.py | ✅ | Often | Affects UI/logic |
| create_memory_for_llm.py | ✅ | Rarely | Affects vectors |
| test_*.py | ✅ | Rarely | Testing only |
| requirements.txt | ⚠️ | Rarely | Dependencies |
| pyproject.toml | ⚠️ | Rarely | Dependencies |
| .env | ✅ | Once | API keys |
| Documentation | ✅ | Often | Info only |

Legend: ✅ Safe, ⚠️ Use caution

