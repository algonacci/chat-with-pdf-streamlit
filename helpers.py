from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain import hub
import os
import uuid

processed_documents = []

def process_document(file):
    global processed_documents
    save_path = save_uploaded_file(file)
    
    if file.type == "application/pdf":
        documents = process_pdf(save_path)
    elif file.type == "text/plain":
        documents = process_txt(save_path)
    else:
        st.error("Unsupported file type.")
        return
    
    processed_documents.extend(documents)
    return f"Processed {file.name}: {len(documents)} pages/sections added."

def save_uploaded_file(uploaded_file):
    os.makedirs("./data", exist_ok=True)
    save_path = f"./data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return save_path

def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()

def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def rag_response(query):
    global processed_documents
    if not processed_documents:
        return "Please upload and process a document first."
    
    for doc in processed_documents:
        if 'id' not in doc.metadata:
            doc.metadata['id'] = str(uuid.uuid4())
    
    vectorstore = Chroma.from_documents(documents=processed_documents, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.stream(query)
