import streamlit as st
from rag_chain import load_vectorstore, get_retriever, build_rag_chain

st.set_page_config(page_title="RAG Financial Research Assistant", layout="wide")

st.title("💹 RAG-Based Financial Research Assistant")
st.write("Analyze SEC filings (10-K, 10-Q), earnings calls, and extract insights using RAG & open LLMs.")

@st.cache_resource
def setup_chain():
    vs = load_vectorstore()
    retriever = get_retriever(vs)
    chain = build_rag_chain(retriever)
    return chain

chain = setup_chain()

query = st.text_area(
    "Ask a financial question (e.g., What were the main risk factors disclosed by Amazon in the latest 10-K?):",
    height=100
)

if st.button("Get Answer") and query.strip():
    with st.spinner("Searching filings and generating answer..."):
        result = chain.invoke({"input": query})
        answer = result.get('result') or result.get('answer') or ''
        sources = result.get('source_documents') or result.get('context') or []
    st.markdown(f"**Answer:**\n\n{answer}")
    with st.expander("Show Context Chunks"):
        for i, doc in enumerate(sources):
            st.markdown(f"**Source {i+1}:**")
            st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))

st.markdown("---")
st.subheader("Sample Use Cases")
st.write("""
- Revenue breakdowns by segment
- Year-over-year comparison of key metrics
- Management's discussion of risks
- Summary of cash flow statements
- Sentiment of earnings call Q&A
- ...and 20+ more!
""")
