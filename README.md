https://researchpaper-rag-chatbot.streamlit.app/

# Mini RAG Studio

A lightweight Retrieval-Augmented Generation (RAG) app for asking questions over a single PDF document. Upload a document, build a vector index, and query it in natural language — with every answer traceable back to the exact source passages.

Built as part of C-DAC Mohali summer training (ML / GenAI / LLMs).

## Features

- PDF upload and ingestion
- Configurable chunking (chunk size, overlap)
- Choice of vector store: FAISS or ChromaDB
- Retrieval-augmented Q&A powered by Google Gemini
- Adjustable Top-K retrieval
- Full chat history within a session
- Retrieved passages shown alongside each answer, with source page references
- Response time reporting per query

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Google Gemini |
| Embeddings | HuggingFace |
| Vector Store | FAISS / ChromaDB |

## Project Structure

```
Mini_RAG_Chatbot/
├── app.py                  # Streamlit entry point
├── data/                   # Uploaded PDFs (created at runtime)
├── database/               # Persisted vector store (faiss_db / chroma_db)
├── utils/
│   ├── loader.py            # PDFLoader
│   ├── splitter.py          # DocumentSplitter
│   ├── embeddings.py        # EmbeddingModel
│   ├── vectorstore.py       # VectorStore
│   └── rag_chain.py         # RAGChain
├── requirements.txt
└── .env                     # API keys (not committed)
```

## Setup

1. **Clone and enter the project**

   ```bash
   cd Mini_RAG_Chatbot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv rag_env
   rag_env\Scripts\activate      # Windows
   source rag_env/bin/activate   # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**

   Create a `.env` file in the project root:

   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`.

## Usage

1. Upload a PDF from the sidebar.
2. Choose a vector database (FAISS or ChromaDB) and adjust chunking / retrieval settings if needed.
3. Click **Build Database**.
4. Once the database is ready, type a question and click **Ask**.
5. Review the answer, the retrieved passages, and the source page numbers.
6. Use **Clear Chat** to reset the conversation, or **Delete Database** to remove the current index and start over with a new document.

## Notes

- Only one PDF is held at a time — uploading a new file replaces the previous one.
- Rebuilding the database is required after changing chunk size, overlap, or the source PDF.

## Author

Ipshita Bhardwaj