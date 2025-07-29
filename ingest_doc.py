import os
from sentence_transformers import SentenceTransformer
from core.splitter import split_text
from core.vectorstore import save_to_faiss
DATA_TXT_PATH = 'data.txt'
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_text_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    print(f"Attempting to ingest data from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        if not full_text.strip():
            print(f"Warning: The file '{file_path}' is empty or contains only whitespace. Nothing to ingest.")
            return
        chunks = split_text(full_text)
        print(f"Text successfully read and split into {len(chunks)} chunks.")
        save_to_faiss(chunks, embedding_model)
        print(f"Successfully processed and stored data from '{file_path}' in the vector database.")

    except UnicodeDecodeError:
        print(f"Error: Could not decode '{file_path}'. Please ensure it is a plain text file and encoded in UTF-8.")
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}")

if __name__ == "__main__":
    ingest_text_file(DATA_TXT_PATH)