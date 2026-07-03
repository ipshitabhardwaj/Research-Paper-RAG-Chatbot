import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma


class VectorStore:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    # ----------------------------
    # FAISS
    # ----------------------------

    def create_faiss(self, chunks):

        db = FAISS.from_documents(
            chunks,
            self.embedding_model
        )

        db.save_local("database/faiss_db")

        return db

    def load_faiss(self):

        return FAISS.load_local(
            "database/faiss_db",
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    # ----------------------------
    # ChromaDB
    # ----------------------------

    def create_chroma(self, chunks):

        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory="database/chroma_db"
        )

        return db

    def load_chroma(self):

        return Chroma(
            persist_directory="database/chroma_db",
            embedding_function=self.embedding_model
        )

    # ----------------------------
    # Generic Interface
    # ----------------------------

    def create(self, chunks, db_type="faiss"):

        if db_type.lower() == "faiss":
            return self.create_faiss(chunks)

        elif db_type.lower() == "chroma":
            return self.create_chroma(chunks)

        else:
            raise ValueError("Unsupported database")

    def load(self, db_type="faiss"):

        if db_type.lower() == "faiss":
            return self.load_faiss()

        elif db_type.lower() == "chroma":
            return self.load_chroma()

        else:
            raise ValueError("Unsupported database")