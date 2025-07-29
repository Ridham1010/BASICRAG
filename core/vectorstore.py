from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import os

class LocalEmbeddingFunction(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def save_to_faiss(
    text_chunks: List[str],
    model,
    index_path: str = "faiss_index"
):   
    embedding_function = LocalEmbeddingFunction(model)
    db = FAISS.from_texts(texts=text_chunks, embedding=embedding_function)
    db.save_local(index_path)

def load_faiss(embedding_model: Embeddings, path: str = "faiss_index") -> FAISS:
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

