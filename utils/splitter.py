"""
splitter.py
-----------
Splits documents into smaller overlapping chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:

    def __init__(self, chunk_size=1000, chunk_overlap=200):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents):

        chunks = self.text_splitter.split_documents(documents)

        print("=" * 60)
        print(f"Total Chunks Created : {len(chunks)}")
        print("=" * 60)

        return chunks