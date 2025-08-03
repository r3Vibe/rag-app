"""
Query module for semantic search and document retrieval.

This module provides functionality to search through the indexed document
collection using semantic similarity. It interfaces with the FAISS vector
store to find the most relevant document chunks based on user queries.
"""

from embeddings import load_vector_store


def query_vector_store(query):
    """
    Search the vector store for documents most relevant to the given query.

    This function performs semantic similarity search using the FAISS vector store.
    It converts the user's query into an embedding vector and finds the most
    similar document chunks in the knowledge base.

    The search process:
    1. Load the existing vector store from disk
    2. Convert the query text to an embedding vector
    3. Perform cosine similarity search against all indexed documents
    4. Return the top-k most similar document chunks

    Args:
        query (str): The user's question or search query. This will be
                    converted to an embedding for similarity comparison.

    Returns:
        List[Document]: A list of the most relevant document chunks, each
                       containing:
                       - page_content: The actual text content
                       - metadata: Document metadata (filename, page number, etc.)

                       Results are ordered by similarity score (highest first).

    Raises:
        FileNotFoundError: If no vector store exists (no documents have been indexed).
        Exception: If the vector store is corrupted or query processing fails.

    Note:
        The function returns the top 3 most similar documents by default.
        This provides a good balance between relevance and context length
        for most question-answering scenarios.
    """
    # Load the FAISS vector store containing document embeddings
    vector_store = load_vector_store()

    # Perform similarity search and return top 3 most relevant chunks
    results = vector_store.similarity_search(query, k=3)

    return results  # Return the search results as a list of Document objects
