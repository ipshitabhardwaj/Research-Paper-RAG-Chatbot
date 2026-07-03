from pathlib import Path
from utils.loader import PDFLoader

# Find all PDF files in the data folder
pdf_files = list(Path("data").glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF file found inside the 'data' folder.")

# Use the first PDF found
pdf_path = pdf_files[0]

print(f"Loading PDF: {pdf_path.name}")

loader = PDFLoader(str(pdf_path))

documents = loader.load()

print("\nFirst 1000 characters:\n")
print(documents[0].page_content[:1000])