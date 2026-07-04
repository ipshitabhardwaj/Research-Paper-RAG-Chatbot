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
    page_icon="📖",
    layout="wide"
)

# -----------------------------------------------------
# Visual identity — "The Reading Room"
# A card-catalog / archive aesthetic: chunks read like
# index cards, sources read like library stamps.
# -----------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #EDEBE1;
    --paper-card: #FAF9F3;
    --ink: #21262D;
    --ink-soft: #5B6169;
    --teal: #2B5959;
    --teal-deep: #17302F;
    --ochre: #B8823C;
    --line: #D6D1C0;
}

/* base */
.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--teal-deep) !important; }
p, span, label, div { color: var(--ink); }

/* hero */
.rr-hero {
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 2px solid var(--teal-deep);
    margin-bottom: 1.6rem;
}
.rr-hero .rr-title {
    font-family: 'Fraunces', serif;
    font-size: 2.3rem;
    font-weight: 600;
    color: var(--teal-deep);
    letter-spacing: -0.01em;
    margin-bottom: 0.15rem;
}
.rr-hero .rr-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ochre);
}

/* section labels, used in place of default st.subheader visuals */
.rr-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
    border-left: 3px solid var(--ochre);
    padding-left: 0.5rem;
    margin: 0.3rem 0 0.9rem 0;
}

/* sidebar = the "requisition desk" */
[data-testid="stSidebar"] {
    background: var(--teal-deep);
}
[data-testid="stSidebar"] * { color: #EDEBE1 !important; }
[data-testid="stSidebar"] .rr-label { color: #D9C79E; border-left-color: #D9C79E; }
[data-testid="stSidebar"] hr { border-color: rgba(237,235,225,0.15); }

/* inputs */
.stTextInput > div > div > input,
[data-testid="stSidebar"] .stFileUploader section {
    background: var(--paper-card);
    border: 1px solid var(--line);
    border-radius: 3px;
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] .stFileUploader section * { color: var(--ink) !important; }

/* sliders / radio accent */
[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] { background-color: #D9C79E !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--ochre) !important; }

/* buttons — mono, uppercase, no default rounded-pill AI look */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 3px;
    border: 1px solid var(--teal);
    background: var(--paper-card);
    color: var(--teal-deep);
    padding: 0.55rem 1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--teal-deep);
    color: var(--paper-card);
    border-color: var(--teal-deep);
}
/* build database = primary action, stamped in ochre */
[data-testid="stSidebar"] .stButton:nth-of-type(1) button {
    background: var(--ochre);
    color: var(--teal-deep);
    border-color: var(--ochre);
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton:nth-of-type(1) button:hover {
    background: #A17233;
}
/* delete database = quiet warning, not a scary red block */
[data-testid="stSidebar"] .stButton:nth-of-type(3) button {
    background: transparent;
    border: 1px dashed rgba(237,235,225,0.4);
    color: #EDEBE1;
}

/* metric cards */
[data-testid="stMetric"] {
    background: var(--paper-card);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0.7rem 0.8rem 0.5rem 0.8rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-soft) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    color: var(--teal-deep) !important;
}

/* chat bubbles */
.rr-msg { display: flex; margin: 0.5rem 0; }
.rr-msg-user { justify-content: flex-end; }
.rr-msg-assistant { justify-content: flex-start; }
.rr-bubble {
    max-width: 72%;
    padding: 0.65rem 0.95rem;
    border-radius: 4px;
    font-size: 0.93rem;
    line-height: 1.45;
}
.rr-bubble-user {
    background: var(--teal-deep);
    color: #F4F2E9;
    border-top-right-radius: 0;
}
.rr-bubble-assistant {
    background: var(--paper-card);
    border: 1px solid var(--line);
    border-left: 3px solid var(--ochre);
    color: var(--ink);
    border-top-left-radius: 0;
}
.rr-role {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.6;
    display: block;
    margin-bottom: 0.2rem;
}

/* retrieved chunks = index cards */
[data-testid="stExpander"] {
    border: 1px dashed var(--line) !important;
    border-radius: 3px !important;
    background: var(--paper-card) !important;
}
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--teal-deep) !important;
}

/* source page stamps */
.rr-stamp-row { margin-top: 0.6rem; }
.rr-stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: var(--ochre);
    border: 1.5px solid var(--ochre);
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    margin: 0 0.3rem 0.3rem 0;
    transform: rotate(-1.5deg);
}

/* footer / colophon */
.rr-colophon {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    color: var(--ink-soft);
    padding: 1rem 0 0.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Hero
# -----------------------------------------------------

st.markdown("""
<div class="rr-hero">
    <div class="rr-title">Mini RAG Studio</div>
    <div class="rr-sub">LangChain · Gemini · FAISS / ChromaDB</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.markdown('<div class="rr-label">Source Document</div>', unsafe_allow_html=True)

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="rr-label">Retrieval Settings</div>', unsafe_allow_html=True)

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

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

build_db = st.sidebar.button(
    "Build Database",
    use_container_width=True
)

clear_chat = st.sidebar.button(
    "Clear Chat",
    use_container_width=True
)

delete_db = st.sidebar.button(
    "Delete Database",
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

    st.sidebar.success("PDF received")

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

    st.sidebar.success("Database cleared")

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

    with st.spinner("Loading embedding model..."):

        embedding = EmbeddingModel().get_embedding_model()

    with st.spinner("Creating vector database..."):

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

    st.success("Database built")

# -----------------------------------------------------
# Document Statistics
# -----------------------------------------------------

if st.session_state.database_ready:

    st.markdown('<div class="rr-label">Catalog Record</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Pages", len(st.session_state.documents))
    col2.metric("Chunks", len(st.session_state.chunks))
    col3.metric("Database", vector_db)
    col4.metric("Top K", top_k)

    st.caption(f"Current document — {st.session_state.pdf_name}")

# -----------------------------------------------------
# Chat Interface
# -----------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="rr-label">Ask a Question</div>', unsafe_allow_html=True)

question = st.text_input(
    "Enter your question about the document",
    label_visibility="collapsed",
    placeholder="What does the document say about..."
)

if st.button("Ask", type="primary"):

    if not st.session_state.database_ready:

        st.warning("Please build the database first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Consulting the archive..."):

            start = time.time()

            answer, docs = st.session_state.rag.ask(
                question,
                k=top_k
            )

            end = time.time()

        st.session_state.retrieved_docs = docs

        st.session_state.messages.append(("User", question))
        st.session_state.messages.append(("Assistant", answer))

        st.caption(f"Answered in {end - start:.2f}s")

# -----------------------------------------------------
# Chat History
# -----------------------------------------------------

if len(st.session_state.messages) > 0:

    st.markdown('<div class="rr-label">Conversation</div>', unsafe_allow_html=True)

    for role, message in st.session_state.messages:

        if role == "User":
            st.markdown(f"""
            <div class="rr-msg rr-msg-user">
                <div class="rr-bubble rr-bubble-user">
                    <span class="rr-role">You</span>{message}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rr-msg rr-msg-assistant">
                <div class="rr-bubble rr-bubble-assistant">
                    <span class="rr-role">Gemini</span>{message}
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------
# Retrieved Chunks
# -----------------------------------------------------

if len(st.session_state.retrieved_docs) > 0:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="rr-label">Retrieved Passages</div>', unsafe_allow_html=True)

    for i, doc in enumerate(st.session_state.retrieved_docs, start=1):

        page = doc.metadata.get("page", "Unknown")

        with st.expander(f"Card {i:02d}  ·  Page {page}"):
            st.write(doc.page_content)

# -----------------------------------------------------
# Source Pages
# -----------------------------------------------------

if len(st.session_state.retrieved_docs) > 0:

    pages = sorted(
        {str(doc.metadata.get("page", "Unknown")) for doc in st.session_state.retrieved_docs}
    )

    stamps = "".join(f'<span class="rr-stamp">p. {p}</span>' for p in pages)

    st.markdown(f"""
    <div class="rr-stamp-row">{stamps}</div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.markdown("""
<div class="rr-colophon">
    Mini RAG Studio &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; Gemini &nbsp;·&nbsp;
    HuggingFace Embeddings &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)