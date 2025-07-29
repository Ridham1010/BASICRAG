import pypdf
from core.splitter import split_text
from core.vectorstore import save_to_faiss
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
PDF_PATH = 'CN-2023-0012.pdf'
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    chunks = split_text(text)
    save_to_faiss(chunks, embedding_model)
    print(f"Successfully processed and stored data from '{pdf_path}' in the vector database.")

extract_text_from_pdf(PDF_PATH)

    
