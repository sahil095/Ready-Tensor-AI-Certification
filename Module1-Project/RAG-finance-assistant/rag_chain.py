import os
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

# VECTORSTORE_PATH = "data/faiss_index"
VECTORSTORE_PATH = "data/chroma_index"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Chroma loads (and persists) using the persist_directory argument
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

def get_retriever(vectorstore, top_k=5):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

def build_rag_chain(retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    system_prompt = (
        "You are a financial analyst assistant. Use the provided context from SEC filings (10-K, 10-Q) and earnings transcripts "
        "to answer questions factually and concisely. If the answer is not in the context, say 'Not found in filings.'"
    )
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="System: " + system_prompt + "\n\nContext:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )
    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt
    )
    chain = create_retrieval_chain(
        retriever,
        combine_docs_chain,
    )
    return chain
