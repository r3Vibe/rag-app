"""embeddings module"""

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """Get HuggingFace embeddings model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings


def get_vector_store(embeddings):
    """Get FAISS vector store."""
    # Create a FAISS index with the dimension of the embeddings
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store


def load_vector_store():
    """Load the vector store from a local file."""
    vector_store = FAISS.load_local(
        "context_index", get_embeddings(), allow_dangerous_deserialization=True
    )
    return vector_store
