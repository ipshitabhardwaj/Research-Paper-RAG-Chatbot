"""
embeddings.py
----------------
Creates embedding models.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def get_embedding_model(self):
        return self.model