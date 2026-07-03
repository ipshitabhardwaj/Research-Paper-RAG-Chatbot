from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore


# -------------------------
# Load PDF
# -------------------------

pdf_file = list(Path("data").glob("*.pdf"))[0]

loader = PDFLoader(str(pdf_file))

documents = loader.load()


# -------------------------
# Split
# -------------------------

splitter = DocumentSplitter()

chunks = splitter.split(documents)


# -------------------------
# Embeddings
# -------------------------

embedding = EmbeddingModel()

embedding_model = embedding.get_embedding_model()


# -------------------------
# Create FAISS
# -------------------------

faiss = VectorStore(embedding_model)

vector_db = faiss.create_faiss(chunks)

print("\nFAISS Created Successfully\n")


# -------------------------
# Test Search
# -------------------------

query = "What is self attention?"

results = vector_db.similarity_search(
    query,
    k=3
)

print("=" * 70)

print("Top Retrieved Chunks")

print("=" * 70)

for i, doc in enumerate(results, 1):

    print(f"\nChunk {i}\n")

    print(doc.page_content[:400])

    print("\n")