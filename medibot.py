import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    # Custom CSS for background and chat container
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        }
        .stChatMessage, .stTextInput, .stButton {
            border-radius: 12px !important;
        }
        .stChatMessage {
            background: rgba(255,255,255,0.07) !important;
            padding: 1em !important;
            margin-bottom: 0.5em !important;
        }
        .stTextInput>div>input {
            background: #f0f4fa !important;
            color: #222 !important;
            border-radius: 8px !important;
            border: 1px solid #2a5298 !important;
        }
        .stButton>button {
            background: #2a5298 !important;
            color: #fff !important;
            border-radius: 8px !important;
        }
        .medical-icon {
            display: block;
            margin: 0 auto 1em auto;
            width: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.sidebar.title("Medical Chatbot")
    st.sidebar.markdown("""
    **Welcome!**
    
    - Ask any medical question.
    - Answers are sourced from the Gale Encyclopedia of Medicine.
    - Your privacy is respected.
    """)

    st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" class="medical-icon" />', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #fff;'>Ask Medical Chatbot!</h1>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Type your medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        #HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            GROQ_MODEL_NAME="llama-3.1-8b-instant"   #or "llama-3.2-3b-preview"

            # GROQ_API_KEY=os.environ.get("GROQ_API_KEY")        #Use this for local testing

            GROQ_API_KEY = st.secrets['GROQ_API_KEY']  # Use this for deployment on Streamlit Cloud

            llm = ChatGroq(
                model_name=GROQ_MODEL_NAME,
                api_key=GROQ_API_KEY,
                temperature=0.5,
                max_tokens=512
            )

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            #Document Combining Chain (stuff documents into prompt)
            combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)

            #Retrieval chain (retriever + doc combiner)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k' : 3}), combine_docs_chain)

            response=rag_chain.invoke({'input': prompt})

            result=response["answer"]
            print ("\nSOURCE DOCUMENTS:")
            for doc in response["context"]:
                print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()