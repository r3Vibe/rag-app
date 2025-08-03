"""
Document loader module for PDF processing and vector store management.

This module handles loading PDF documents, extracting their text content,
converting them to embeddings, and storing them in a FAISS vector store
for efficient retrieval. It supports both batch processing of entire
directories and individual file processing.
"""

import os

from langchain_community.document_loaders import PyPDFLoader

from embeddings import get_embeddings, get_vector_store, load_vector_store


def load_pdf_files():
    """
    Load all PDF files from the documents folder and add them to the vector store.

    This function performs batch processing of all PDF files in the "documents"
    directory. It either loads an existing vector store or creates a new one,
    then processes each PDF file by extracting text, creating embeddings,
    and adding the content to the searchable index.

    The function handles the complete pipeline:
    1. List all files in the documents folder
    2. Load or create a vector store
    3. Process each PDF file using PyPDFLoader
    4. Extract text content and split into manageable chunks
    5. Generate embeddings and add to the vector store
    6. Save the updated vector store to disk

    Raises:
        FileNotFoundError: If the documents folder doesn't exist.
        Exception: If PDF loading fails or vector store operations fail.

    Note:
        This function will overwrite any existing vector store with the
        complete set of documents from the folder.
    """
    print("Processing PDF files...")
    folder = "documents"  # Directory containing PDF files to process
    all_docs = os.listdir(folder)  # Get list of all files in the directory

    # Try to load existing vector store, create new one if it doesn't exist
    try:
        vector_store = load_vector_store()  # Attempt to load existing index
    except (FileNotFoundError, Exception):
        # Create new vector store if loading fails
        embeddings = get_embeddings()  # Initialize embeddings model
        vector_store = get_vector_store(embeddings)  # Create empty FAISS store

    # Process each document in the folder
    for doc in all_docs:
        """Load each PDF file and add its content to the vector store."""
        print(f"Processing {doc}...")
        file_path = os.path.join(folder, doc)  # Construct full file path
        pdf_data = PyPDFLoader(file_path)  # Initialize PDF loader
        documents = pdf_data.load()  # Extract text content and metadata
        vector_store.add_documents(documents)  # Add document chunks to vector store
    print("All documents processed. Saving vector store...")
    # Persist the updated vector store to disk
    vector_store.save_local("context_index")
    print("Vector store saved successfully.")


def load_given_pdf(pdf_file):
    """
    Load a specific PDF file and add it to the existing vector store.

    This function processes a single PDF file and integrates it into the
    existing knowledge base. It's designed for dynamic document addition,
    such as when users upload new files through the web interface.

    The process involves:
    1. Loading the existing vector store from disk
    2. Processing the PDF file to extract text content
    3. Converting text to embeddings and adding to the index
    4. Saving the updated vector store back to disk

    Args:
        pdf_file (str): Full path to the PDF file to be processed.
                       The file must exist and be readable.

    Raises:
        FileNotFoundError: If the specified PDF file doesn't exist or
                          if no existing vector store is found.
        Exception: If PDF processing fails or vector store operations fail.

    Note:
        This function assumes an existing vector store exists. If no vector
        store is found, it will raise an exception rather than creating a new one.
    """
    # Load the existing FAISS vector store from disk
    vector_store = load_vector_store()

    # Process the PDF file
    pdf_data = PyPDFLoader(pdf_file)  # Initialize loader for the specific file
    documents = pdf_data.load()  # Extract text content and metadata
    vector_store.add_documents(documents)  # Add new document chunks to existing store

    # Save the updated vector store with the new document
    vector_store.save_local("context_index")


if __name__ == "__main__":
    """
    Execute batch processing of PDF files when script is run directly.
    
    This entry point allows the script to be run standalone for initial
    setup or batch reprocessing of all documents in the documents folder.
    It will create a complete vector store index from all available PDFs.
    
    Usage:
        python document_loader.py
    
    This is useful for:
    - Initial setup of the knowledge base
    - Reprocessing all documents after changes
    - Rebuilding the index from scratch
    """
    load_pdf_files()
