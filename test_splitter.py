from pathlib import Path

from utils.loader import PDFLoader
from utils.splitter import DocumentSplitter

# Find the first PDF in the data folder
pdf_folder = Path("data")
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF found in the data folder.")

pdf_path = pdf_files[0]

print(f"\nLoading PDF: {pdf_path.name}\n")

loader = PDFLoader(str(pdf_path))
documents = loader.load()

splitter = DocumentSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split(documents)

print(f"\nTotal Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content[:500])

print("\nMetadata:\n")
print(chunks[0].metadata)