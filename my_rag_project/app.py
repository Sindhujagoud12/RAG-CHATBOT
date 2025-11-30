print('DEBUG: app.py loaded')
import streamlit as st
import os
import io
from dotenv import load_dotenv
from langchain_core.documents import Document
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# ------------------------------
# Configuration & Setup
# ------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Try to get the key from environment variables (local .env) or Streamlit secrets
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

# Initialize session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------------------
# UI Layout
# ------------------------------
st.title("RAG Based Question Answering App")
print('DEBUG: UI built')
# RuntimeError removed
st.write("Ask any question based on your uploaded documents.")

# ------------------------------
# File Upload & Processing
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF/Text file", type=["pdf", "txt"])

if uploaded_file:
    # We use a button to trigger processing so it doesn't re-run on every interaction
    if st.button("Process File"):
        with st.spinner("Processing file..."):
            try:
                # Load file
                file_content = uploaded_file.getvalue()
                # Handle PDF files
                if uploaded_file.type == "application/pdf":
                    pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
                
                # Handle Text files
                else:
                    text = file_content.decode("utf-8")
                    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
                
                # Split text
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                
                # Create embeddings and vectorstore
                # Using the same model as in the notebook
                emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Create Chroma vectorstore in memory
                vectordb = Chroma.from_documents(chunks, emb)
                
                st.session_state.vectorstore = vectordb
                st.success("File processed successfully! You can now ask questions.")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")

# ------------------------------
# Question Answering
# ------------------------------
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload and process a file first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Retrieve context
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.invoke(question)
                context = "\n".join([d.page_content for d in docs])
                
                # Generate answer using Groq
                llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)
                
                prompt = f"""
                You are an assistant. Answer using ONLY the context below.

                CONTEXT:
                {context}

                QUESTION:
                {question}

                Answer clearly.
                """
                
                response = llm.invoke(prompt)
                
                # Display result
                st.subheader("Answer")
                st.write(response.content)
                
                with st.expander("Retrieved Context"):
                    st.write(context)
                    
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("About")
    st.write("""
        This is a Retrieval-Augmented Generation (RAG) application.
        It uses embeddings + vector search + LLM response generation.
    """)
    st.write("Built with **Streamlit**, **LangChain**, **Chroma**, and **Groq**.")
