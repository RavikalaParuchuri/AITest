import streamlit as st 
from pdfminer.high_level import extract_text 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_classic.chains import RetrievalQA 
import tempfile 
import os 
import httpx
import shutil

import tiktoken 
tiktoken_cache_dir = "./token" 
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

client = httpx.Client(verify=False)

# LLM and Embedding setup 
llm = ChatOpenAI( 
    base_url="https://genailab.turl.in", 
    model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="sk-xxxxxxxxx", 
    http_client=client 
)

embedding_model = OpenAIEmbeddings( 
    base_url="https://genailab.url.in", 
    model="azure/genailab-maas-text-embedding-3-large", 
    api_key="sk-xxxxxxxxxxxxxx", 
    http_client=client)

st.set_page_config(page_title="RAG PDF Summarizer") 
st.title("RAG-powered PDF Summarizer")

upload_file = st.file_uploader("Upload a PDF", type="pdf") 
if upload_file: 
    doc_name = upload_file.name
    st.subheader(f"Summary of the {doc_name}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file: 
        temp_file.write(upload_file.read()) 
        temp_file_path = temp_file.name

    # Step 1: Extract text 
    raw_text = extract_text(temp_file_path) 
    
    # Step 2: Chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    chunks = text_splitter.split_text(raw_text)    

# Step 3: Embed and store in Chroma 
    with st.spinner("Indexing document..."): 
        #vectordb = Chroma.from_texts(chunks, embedding_model,persist_directory="./chroma_index")
        # Clear old index if it exists
        if os.path.exists("./chroma_index"):
            shutil.rmtree("./chroma_index")
        vectordb = Chroma.from_texts(chunks, embedding_model,persist_directory=None)  
        vectordb.persist() 

# Step 4: RAG QA Chain 
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    #retriever = vectordb.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)
   
   # Step 5: Ask summarization prompt 
    summary_prompt = "Please summarize this document based on the key topics:" 
    with st.spinner("Running RAG summarization..."): 
        # result = rag_chain.run(summary_prompt) 
        result = rag_chain.invoke(summary_prompt)
        st.markdown(result['result'])
        #st.write(result)