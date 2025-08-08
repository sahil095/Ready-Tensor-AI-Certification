# RAG-Based Financial Research Assistant for Amazon

## Overview

The **RAG-Based Financial Research Assistant** is an AI-powered, production-ready tool designed for financial analysts, investors, and business leaders. It leverages Retrieval-Augmented Generation (RAG) using Chroma vector indexing, open-source sentence-transformer embeddings, and Llama 3 LLM (via Groq) to answer complex natural language questions about Amazon's SEC filings (10-K, 10-Q) and earnings call transcripts.  
With a fast Streamlit interface, the assistant provides reliable, context-grounded answers and cites relevant source document snippets—saving hours of manual search.

**Key Features:**
- Contextual Q&A over Amazon’s annual, quarterly, and earnings reports
- Answers are grounded in actual filings and transcripts, with cited sources
- Local vector index for privacy and reproducibility
- Easy-to-use Streamlit UI

---

## Installation

1. **Clone this repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your Groq API key:**
    - Create a `.env` file in the project root:
      ```
      GROQ_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
      ```
    - [Get a Groq API key here](https://console.groq.com/keys)

4. **Download and organize Amazon filings and transcripts:**
    - Place Amazon 10-K, 10-Q, and earnings call PDF files in `data/filings/`
    - [SEC EDGAR Search for AMZN filings](https://www.sec.gov/edgar/browse/?CIK=AMZN)

5. **Build the vector index from your documents:**
    ```bash
    python ingest.py
    ```

---

## Usage

1. **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Interact with the assistant:**
    - Enter your financial or business question about Amazon in the text box.
    - Click "Get Answer."
    - The assistant retrieves relevant context and generates a grounded answer.
    - Expand the "Show Context Chunks" section to see original document sources.

---

## Example Questions to Ask

Here are several example questions you can ask (provided you have relevant filings and transcripts indexed):

- What was Amazon's total revenue in 2023?
- How much did Amazon spend on research and development last year?
- What are the major risk factors disclosed in Amazon's most recent 10-K?
- Provide a breakdown of Amazon’s revenue by business segment.
- How did Amazon describe the competitive landscape in its 2023 10-K?
- What were the total operating expenses for Amazon in 2023?
- Summarize Amazon’s management discussion about AWS in the latest annual report.
- What guidance did management provide for the next quarter in the latest earnings call?
- How did the CFO respond to questions about advertising revenue?
- Was there any mention of Prime Video strategy in the latest earnings call transcript?

---

## Support and Customization

- **Add filings for any company:** Place new PDF filings in `data/filings/` and rerun `python ingest.py`
- **Tune prompt or retrieval:** Edit the prompt in `rag_chain.py` for your business context.
- **All document indexes and embeddings are stored locally in `data/chroma_index/`** for security and reproducibility.

---

## License

This project is open-source and free for research and educational use.  
For production or commercial deployments, please ensure compliance with data privacy regulations and review relevant LLM provider terms.

---

*Built using LangChain, Chroma, HuggingFace, Groq, and Streamlit.*
