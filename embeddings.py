"""
Embeddings module for text vectorization and vector store management.

This module handles the creation and management of text embeddings using HuggingFace
transformers and FAISS vector store for efficient similarity search. It provides
functions to initialize embeddings models, create vector stores, and load existing
vector stores from disk.
"""

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Initialize and return a HuggingFace embeddings model.

    Creates a sentence transformer model optimized for semantic similarity tasks.
    The chosen model (all-mpnet-base-v2) provides high-quality embeddings with
    768 dimensions and is well-suited for document retrieval tasks.

    Returns:
        HuggingFaceEmbeddings: A configured embeddings model that can convert
                              text into dense vector representations.

    Note:
        The model will be downloaded from HuggingFace Hub on first use and
        cached locally for subsequent calls.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # High-quality multilingual model
    )

    return embeddings


def get_vector_store(embeddings):
    """
    Create and return a new FAISS vector store for similarity search.

    Initializes an empty FAISS index with L2 (Euclidean) distance metric.
    The vector store provides efficient similarity search capabilities for
    large collections of document embeddings.

    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings model to use for
                                           converting text to vectors.

    Returns:
        FAISS: A new, empty vector store ready for document insertion.
              Supports adding documents, similarity search, and persistence.

    Note:
        The index dimension is automatically determined by embedding a test query
        to ensure consistency with the chosen embeddings model.
    """
    # Determine embedding dimension by testing with a sample query
    embedding_dimension = len(embeddings.embed_query("hello world"))

    # Create a FAISS index using L2 (Euclidean) distance metric
    index = faiss.IndexFlatL2(embedding_dimension)

    # Initialize the FAISS vector store with required components
    vector_store = FAISS(
        embedding_function=embeddings,  # Function to convert text to vectors
        index=index,  # FAISS index for similarity search
        docstore=InMemoryDocstore(),  # In-memory storage for document content
        index_to_docstore_id={},  # Mapping between index positions and document IDs
    )

    return vector_store


def load_vector_store():
    """
    Load an existing vector store from local disk storage.

    Restores a previously saved FAISS vector store from the "context_index"
    directory. This allows for persistence of document embeddings across
    application restarts.

    Returns:
        FAISS: The loaded vector store containing previously indexed documents
               and their embeddings, ready for similarity search operations.

    Raises:
        FileNotFoundError: If no saved vector store exists at the expected path.
        Exception: If the vector store files are corrupted or incompatible.

    Note:
        Uses allow_dangerous_deserialization=True to load pickle files.
        This is acceptable for local development but should be used cautiously
        in production environments.
    """
    vector_store = FAISS.load_local(
        "context_index",  # Directory containing saved index
        get_embeddings(),  # Embeddings model (must match saved version)
        allow_dangerous_deserialization=True,  # Allow pickle deserialization
    )
    return vector_store
