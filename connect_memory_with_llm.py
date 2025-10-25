import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup LLM (GROQ)
# HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME="llama-3.1-8b-instant"   #or "llama-3.2-3b-preview"


llm = ChatGroq(
    model_name=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.5,
    max_tokens=512
)

# Step 2: Connect LLM with FAISS and Create chain

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Build RAG Chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#Document Combining Chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

#Retrieval chain (retriever + doc combiner)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k' : 3}), combine_docs_chain)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=rag_chain.invoke({'input': user_query})
# print("RESULT: ", response["answer"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
print ("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
