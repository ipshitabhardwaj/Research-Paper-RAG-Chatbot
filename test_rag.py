from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore
from utils.rag_chain import RAGChain

# ------------------------
# Settings
# ------------------------

VECTOR_DB = "faiss"      # change to chroma if required

# ------------------------
# Load PDF
# ------------------------

pdf = list(Path("data").glob("*.pdf"))[0]

loader = PDFLoader(str(pdf))
documents = loader.load()

# ------------------------
# Split
# ------------------------

chunks = DocumentSplitter().split(documents)

# ------------------------
# Embedding
# ------------------------

embedding = EmbeddingModel().get_embedding_model()

# ------------------------
# Vector DB
# ------------------------

vector_db = VectorStore(embedding).create(
    chunks,
    db_type=VECTOR_DB
)

# ------------------------
# RAG
# ------------------------

rag = RAGChain(vector_db)

question = input("\nAsk your question : ")

answer, docs = rag.ask(question)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(answer)

print("\n" + "=" * 80)
print("SOURCE DOCUMENTS")
print("=" * 80)

for i, doc in enumerate(docs, 1):
    page = doc.metadata.get("page", "Unknown")
    print(f"\nSource {i} (Page {page})")
    print("-" * 60)
    print(doc.page_content[:300])