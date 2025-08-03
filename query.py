"""take user query and return relevant documents from the vector store"""

from embeddings import load_vector_store


def query_vector_store(query):
    """Query the vector store for relevant documents."""
    vector_store = load_vector_store()  # Load the FAISS vector store
    results = vector_store.similarity_search(query, k=3)  # Perform similarity search
    return results  # Return the search results
