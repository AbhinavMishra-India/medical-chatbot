# Medical Chatbot

## Links
- **Live Demo:** [Streamlit App](https://abhinavmishra.streamlit.app/)
- **GitHub Repository:** [github.com/AbhinavMishra-India/medical-chatbot](https://github.com/AbhinavMishra-India/medical-chatbot)

An intelligent medical chatbot that leverages LLMs (Large Language Models) and vector search to answer medical queries using information from the Gale Encyclopedia of Medicine.

## ⚠️ Disclaimer
This chatbot uses curated and reliable medical data.  
Current coverage is focused and may not include all diseases.  
The system is continuously being improved to expand its knowledge base and accuracy.

## Features
- Natural language medical Q&A
- Uses LangChain and HuggingFace for LLM integration
- Retrieves context from Gale Encyclopedia of Medicine (PDF)
- Vector search powered by FAISS
- Streamlit web interface for easy interaction

## Project Structure
- `medibot.py`: Main Streamlit app for the chatbot interface
- `create_memory_for_llm.py`: Script to create vector memory from the medical encyclopedia
- `connect_memory_with_llm.py`: Connects the vector memory to the LLM for retrieval
- `data/`: Contains the Gale Encyclopedia of Medicine PDF
- `vectorstore/`: Stores FAISS vector database files


## Environment Variables & Secrets Management

This project uses API keys and secrets for LLM access. These should not be committed to version control. Use the following files:

- `.env`: For local development, store environment variables (e.g., `GROQ_API_KEY`) here. This file is ignored by git.
- `.streamlit/secrets.toml`: For Streamlit Cloud deployment, add secrets (e.g., `GROQ_API_KEY`) here. This file is also ignored by git.

**Never share or commit your API keys or secrets.**

---

## Setup Instructions

### 1. Prerequisite: Install Pipenv
Follow the official Pipenv installation guide: [Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

### 2. Install Required Packages
Run the following commands in your terminal:

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf huggingface_hub streamlit
```

### 3. Prepare the Vector Database
Run the following script to process the medical encyclopedia and create the vector store:

```bash
pipenv run python create_memory_for_llm.py
```

### 4. Launch the Chatbot
Start the Streamlit app:

```bash
pipenv run streamlit run medibot.py
```

## Usage
Interact with the chatbot via the Streamlit web interface. Ask medical questions and receive context-aware answers sourced from the Gale Encyclopedia of Medicine.

## Notes
- Ensure `data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf` is present.
- Vector database files are stored in `vectorstore/db_faiss/`.
- For best results, use up-to-date versions of all dependencies.

## License
This project is for educational purposes. Please consult a medical professional for real medical advice.

