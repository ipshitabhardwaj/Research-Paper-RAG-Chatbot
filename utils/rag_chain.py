"""
rag_chain.py
------------------------------------
RAG Pipeline using Gemini + LangChain
"""

import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class RAGChain:

    def __init__(self, vector_db):

        self.vector_db = vector_db

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant.

Answer ONLY using the context provided below.

If the answer is not present in the context, reply:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
        )

    def ask(self, question, k=3):

        docs = self.vector_db.similarity_search(question,k=k)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return response.content, docs