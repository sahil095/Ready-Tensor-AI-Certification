import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data/filings"
VECTORSTORE_PATH = "data/chroma_index"

def ingest_pdfs(data_dir=DATA_DIR, vectorstore_path=VECTORSTORE_PATH):
    all_docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            docs = loader.load()
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )
    # vectorstore.persist()
    print(f"Chroma vectorstore persisted to {vectorstore_path}")

if __name__ == "__main__":
    ingest_pdfs()
