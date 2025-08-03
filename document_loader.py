"""load pdf files and convert to vectors and store them in faiss index"""

import os

from langchain_community.document_loaders import PyPDFLoader

from embeddings import get_embeddings, get_vector_store, load_vector_store


def load_pdf_files():
    """Load PDF files from a folder and convert them to vectors."""
    folder = "documents"  # Folder containing PDF files
    all_docs = os.listdir(folder)  # List all files in the folder

    # Try to load existing vector store first, create new one if it doesn't exist
    try:
        vector_store = load_vector_store()  # Load existing vector store
    except (FileNotFoundError, Exception):
        embeddings = get_embeddings()  # Get the embeddings model
        vector_store = get_vector_store(embeddings)  # Create new FAISS vector store

    for doc in all_docs:
        """  Load each PDF file and add its content to the vector store."""
        file_path = os.path.join(folder, doc)
        pdf_data = PyPDFLoader(file_path)
        documents = pdf_data.load()
        vector_store.add_documents(documents)

    vector_store.save_local("context_index")


def load_given_pdf(pdf_file):
    """Load a given PDF file and add it to the vector store."""
    vector_store = load_vector_store()  # Load the FAISS vector store

    pdf_data = PyPDFLoader(pdf_file)
    documents = pdf_data.load()
    vector_store.add_documents(documents)

    vector_store.save_local("context_index")


if __name__ == "__main__":
    """Run the script to load PDF files and store them in the vector store."""
    load_pdf_files()
