# BASICRAG: Data Processing and Vector Storage Project

## Description

BASICRAG is a Python project designed for efficient data processing and storage, with a focus on enabling Retrieval-Augmented Generation (RAG) applications. It provides tools for splitting large documents into smaller chunks and managing vector embeddings for semantic search and retrieval. The core components are the `splitter.py` and `vectorstore.py` modules, which work together to prepare and store data for use in downstream tasks such as question answering, summarization, and information retrieval.
The `ingest_doc.py` and `ingest_pdf.py` files split your txt and pdf files using `splitter.py` and `vectorstore.py` modules and `main.py` gets the response from LLM on questions asked from your data. 

-   **Document Splitting:** The `splitter.py` module contains logic for breaking down large documents into manageable segments based on various criteria like sentence boundaries, paragraph breaks, or fixed chunk sizes. The `ingest_doc.py` manages your txt files and `ingest_pdf.py` manages your pdf files.
-   **Vector Storage:** The `vectorstore.py` module provides functionality for storing and retrieving vector embeddings of text chunks. It uses the FAISS database for storing vector embeddings.
-   **LLM:** The `main.py` module provides response from the LLM using the top k chunks retrieved.

## Usage
Download the dependencies required for running the code or you can also create a virtual environment using conda or pip.
Keep the files you want to research about in the root directory and update the path of the files in these variables:
- DATA_TXT_PATH in `ingest_doc.py`
- PDF_PATH in `ingest_pdf.py`

## Contact

For questions or inquiries, please contact:ridhamshah1002@gmail.com,+919429646285