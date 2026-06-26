import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from PyPDF2 import PdfReader
import os
import httpx

import tiktoken 
tiktoken_cache_dir = "./token" 
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

client = httpx.Client(verify=False)

# LLM and Embedding setup 
llm = ChatOpenAI( 
    base_url="https://genailab.test.in", 
    model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="sk-uiygt9i767869yu", 
    http_client=client 
)

embedding_model = OpenAIEmbeddings( 
    base_url="https://genailab.test.in", 
    model="azure/genailab-maas-text-embedding-3-large", 
    api_key="sk-uyiugioiuohi", 
    http_client=client)

# --- Helper: Extract text from PDF ---
def extract_text(upload_file):
    reader = PdfReader(upload_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --- Initialize session state ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# --- File upload ---
upload_file = st.file_uploader("Upload PDF", type="pdf")

if upload_file:
    raw_text = extract_text(upload_file)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    # Wrap chunks into Document objects with metadata
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": f"chunk_{i}"}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Create embeddings + Chroma DB
    vectordb = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory="./chroma_index"
    )

    st.session_state.vectordb = vectordb
    st.success("Document indexed!")

# --- Load DB ---
vectordb = st.session_state.vectordb

# --- Chat input ---
query = st.chat_input("Ask a question")

if query and vectordb is not None:
    # Use similarity_search instead of invoke
    docs = vectordb.similarity_search(query, k=5)

    for d in docs:
        print(d.metadata)

    docs = [
    Document(
        page_content=chunk,
        metadata={"chunk_id": i, "page": i}
    )
    for i, chunk in enumerate(chunks)
    ]


    # Build context
    context = "\n\n".join(
    f"Chunk: {d.metadata.get('chunk_id', 'N/A')} | Page: {d.metadata.get('page', 'N/A')}\n{d.page_content}"
    for d in docs
)


    # Call LLM
    result = llm.invoke(
        f"""
        Answer ONLY using context.

        Context:
        {context}

        Question:
        {query}
        """
    )

    # Display
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(result.content)