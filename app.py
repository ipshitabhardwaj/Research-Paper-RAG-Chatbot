import time
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag_chain import RAGChain

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Mini RAG Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mini RAG Studio")
st.caption("LangChain + Google Gemini + FAISS + ChromaDB")

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.header("Configuration")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

vector_db = st.sidebar.radio(
    "Vector Database",
    ["FAISS", "ChromaDB"]
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    500,
    2000,
    1000,
    100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    500,
    200,
    50
)

top_k = st.sidebar.slider(
    "Top K Results",
    1,
    10,
    3
)

build_db = st.sidebar.button(
    "🚀 Build Database",
    use_container_width=True
)

clear_chat = st.sidebar.button(
    "🧹 Clear Chat",
    use_container_width=True
)

delete_db = st.sidebar.button(
    "🗑 Delete Database",
    use_container_width=True
)

# -----------------------------------------------------
# Session State
# -----------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# -----------------------------------------------------
# Save Uploaded PDF
# -----------------------------------------------------

if uploaded_pdf:

    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    # Delete previous PDFs
    for file in data_folder.glob("*.pdf"):
        file.unlink()

    pdf_path = data_folder / uploaded_pdf.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    st.sidebar.success("PDF Uploaded Successfully")

# -----------------------------------------------------
# Delete Database
# -----------------------------------------------------

if delete_db:

    shutil.rmtree("database/faiss_db", ignore_errors=True)
    shutil.rmtree("database/chroma_db", ignore_errors=True)

    st.session_state.rag = None
    st.session_state.database_ready = False
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.retrieved_docs = []

    st.sidebar.success("Database Deleted")

# -----------------------------------------------------
# Clear Chat
# -----------------------------------------------------

if clear_chat:

    st.session_state.messages = []

# -----------------------------------------------------
# Build Vector Database
# -----------------------------------------------------

if build_db:

    pdf_files = list(Path("data").glob("*.pdf"))

    if len(pdf_files) == 0:

        st.error("Please upload a PDF first.")

        st.stop()

    pdf_path = pdf_files[0]

    with st.spinner("Loading PDF..."):

        loader = PDFLoader(str(pdf_path))
        documents = loader.load()

    with st.spinner("Splitting document..."):

        splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split(documents)

    with st.spinner("Loading Embedding Model..."):

        embedding = EmbeddingModel().get_embedding_model()

    with st.spinner("Creating Vector Database..."):

        db = VectorStore(embedding).create(
            chunks,
            db_type=vector_db.lower()
        )

    rag = RAGChain(db)

    st.session_state.rag = rag
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.database_ready = True
    st.session_state.pdf_name = pdf_path.name

    st.success("✅ Database Built Successfully")

# -----------------------------------------------------
# Document Statistics
# -----------------------------------------------------

if st.session_state.database_ready:

    st.divider()

    st.subheader("📊 Document Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Pages",
        len(st.session_state.documents)
    )

    col2.metric(
        "Chunks",
        len(st.session_state.chunks)
    )

    col3.metric(
        "Database",
        vector_db
    )

    col4.metric(
        "Top K",
        top_k
    )

    st.info(f"Current PDF : {st.session_state.pdf_name}")

# -----------------------------------------------------
# Chat Interface
# -----------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions")

question = st.text_input(
    "Enter your question about the document"
)

if st.button("Ask"):

    if not st.session_state.database_ready:

        st.warning("Please build the Vector Database first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Gemini is thinking..."):

            start = time.time()

            answer, docs = st.session_state.rag.ask(
                question,
                k=top_k
            )

            end = time.time()

        st.session_state.retrieved_docs = docs

        st.session_state.messages.append(
            ("User", question)
        )

        st.session_state.messages.append(
            ("Assistant", answer)
        )

        st.success("Answer")

        st.write(answer)

        st.caption(
            f"Response Time : {end-start:.2f} seconds"
        )

# -----------------------------------------------------
# Chat History
# -----------------------------------------------------

if len(st.session_state.messages) > 0:

    st.divider()

    st.subheader("💬 Conversation")

    for role, message in st.session_state.messages:

        if role == "User":

            st.markdown(
                f"**🧑 You :** {message}"
            )

        else:

            st.markdown(
                f"**🤖 Gemini :** {message}"
            )

# -----------------------------------------------------
# Retrieved Chunks
# -----------------------------------------------------

if len(st.session_state.retrieved_docs) > 0:

    st.divider()

    st.subheader("📚 Retrieved Chunks")

    for i, doc in enumerate(
        st.session_state.retrieved_docs,
        start=1
    ):

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        with st.expander(
            f"Chunk {i} (Page {page})"
        ):

            st.write(doc.page_content)

# -----------------------------------------------------
# Source Pages
# -----------------------------------------------------

if len(st.session_state.retrieved_docs) > 0:

    pages = sorted(
        {
            str(
                doc.metadata.get(
                    "page",
                    "Unknown"
                )
            )
            for doc in st.session_state.retrieved_docs
        }
    )

    st.success(
        "Source Pages : "
        + ", ".join(pages)
    )

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.divider()

st.markdown(
"""
### 🚀 Mini RAG Studio

Built with

- LangChain
- Google Gemini
- HuggingFace Embeddings
- FAISS
- ChromaDB
- Streamlit
"""
)