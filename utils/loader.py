"""
loader.py
-----------
Loads PDF documents using LangChain's PyPDFLoader.
"""

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    """
    Class to load PDF documents.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self):
        """
        Load the PDF.

        Returns:
            list[Document]
        """
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print("=" * 60)
        print(f"Loaded {len(documents)} pages.")
        print("=" * 60)

        return documents