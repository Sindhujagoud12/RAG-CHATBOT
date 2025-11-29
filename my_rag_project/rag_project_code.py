!pip install -U langchain langchain-community \
    langchain-core langchain-text-splitters \
    chromadb sentence-transformers \
    unstructured "unstructured[pdf]" \
    langchain-groq

from google.colab import files
uploaded = files.upload()

file_path = list(uploaded.keys())[0]
file_path

from langchain_community.document_loaders import UnstructuredFileIOLoader
import io

loader = UnstructuredFileIOLoader(io.BytesIO(uploaded[file_path]))
docs = loader.load()

len(docs), docs[0].page_content[:500]

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
len(chunks)

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(chunks, emb)
retriever = vectordb.as_retriever()

import os
groq_api_key=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def rag_query(query):
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an assistant. Answer using ONLY the context below.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Answer clearly.
    """

    return llm.invoke(prompt)

print(f"DEBUG: Type of 'retriever' before calling rag_query: {type(retriever)}")
print(f"DEBUG: dir(retriever) before calling rag_query: {dir(retriever)}")
result = rag_query("what are the different commands in sql ")
print(result)

result = rag_query("explain about generative ai")
print(result)

result = rag_query("explain in simple words about generative ai lifecycle")
print(result)

