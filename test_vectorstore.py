from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter
from utils.embeddings import EmbeddingModel
from utils.vectorstore import VectorStore

# Change this to "chroma" to test ChromaDB
VECTOR_DB = "faiss"

pdf = list(Path("data").glob("*.pdf"))[0]

loader = PDFLoader(str(pdf))
documents = loader.load()

splitter = DocumentSplitter()
chunks = splitter.split(documents)

embedding = EmbeddingModel().get_embedding_model()

vector_db = VectorStore(embedding)

db = vector_db.create(chunks, VECTOR_DB)

query = "Explain self attention"

results = db.similarity_search(query, k=3)

print("\nUsing :", VECTOR_DB.upper())

for i, doc in enumerate(results, start=1):
    print("=" * 70)
    print(f"Chunk {i}")
    print("=" * 70)
    print(doc.page_content[:400])