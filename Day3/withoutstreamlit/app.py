import os
import streamlit as st
import fitz
import tempfile
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔐 Set Gemini API key
genai.configure(api_key="AIzaSyCp8H9Ihvgujw76b56eIVQOAK8Jr92YBpo")

# Load PDFs and extract text
def load_papers(pdf_paths):
    documents = []
    for path in pdf_paths:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        documents.append(Document(page_content=text, metadata={"source": os.path.basename(path)}))
    return documents

# Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# Build FAISS vector store
def build_vector_store(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding=embedder)
    return db

# Use Gemini Flash to answer questions
def gemini_answer(query, context_docs):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Use the following context to answer the question.

Context:
{context_text}

Question: {query}
Answer:"""
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("📄 PDF Question Answering with Gemini + FAISS")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        # Save to temp files
        temp_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)

        raw_docs = load_papers(temp_paths)
        chunks = split_documents(raw_docs)
        vectordb = build_vector_store(chunks)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        st.success("PDFs processed and indexed successfully!")

    query = st.text_input("Ask a question about the documents:")

    if query:
        relevant_docs = retriever.get_relevant_documents(query)
        answer = gemini_answer(query, relevant_docs)
        st.subheader("💬 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for doc in relevant_docs:
            st.markdown(f"- `{doc.metadata['source']}`")
