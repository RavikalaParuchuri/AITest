import streamlit as st 
from pdfminer.high_level import extract_text 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_classic.chains import RetrievalQA 
import tempfile 
import os 
import httpx

import tiktoken 
tiktoken_cache_dir = "./token" 
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

client = httpx.Client(verify=False)

# LLM and Embedding setup 
llm = ChatOpenAI( 
    base_url="https://genailab.yst.in", 
    model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="sk-7987yoiuhoi", 
    http_client=client 
)

embedding_model = OpenAIEmbeddings( 
    base_url="https://genailab.url.in", 
    model="azure/genailab-maas-text-embedding-3-large", 
    api_key="sk-yt97y9o8y0", 
    http_client=client)

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("RAG Chat App")

# -------------------------
# 1. LOAD / BUILD DB FIRST
# -------------------------
vectordb = None

# Initialize session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# File upload
upload_file = st.file_uploader("Upload PDF", type="pdf")

if upload_file:
    raw_text = extract_text(upload_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(raw_text)

    st.session_state.vectordb = Chroma.from_texts(
        chunks,
        embedding_model,
        persist_directory="./chroma_index"
    )

    st.success("Document indexed!")

# Load DB
vectordb = st.session_state.vectordb

# Chat input
query = st.chat_input("Ask a question")

if query and vectordb is not None:
    #retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = vectordb.similarity_search(query, k=5)
    #docs = retriever.invoke(query)
    st.write(docs)
    context = "\n\n".join(d.page_content for d in docs)

    result = llm.invoke(
        f"""
Answer ONLY using context.

Context:
{context}

Question:
{query}
"""
    )

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result.content)
