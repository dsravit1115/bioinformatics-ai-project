from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from langchain.schema import Document

def generate_embeddings() -> SentenceTransformerEmbeddings:
    """
    Initialize SentenceTransformer embeddings.
    """
    model_name = "all-MiniLM-L6-v2"  # You can change this to a different huggingface model
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    return embeddings

def create_vectorstore(documents: List[Document], embeddings: SentenceTransformerEmbeddings) -> FAISS:
    """
    Creates a FAISS vector store from document chunks and embeddings.
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def save_vectorstore(vectorstore: FAISS, path: str):
    """
    Saves the FAISS vector store to disk.
    """
    vectorstore.save_local(path)

def load_vectorstore(path: str, embeddings: SentenceTransformerEmbeddings) -> FAISS:
    """
    Loads an existing FAISS vector store from disk.
    """
    vectorstore = FAISS.load_local(path, embeddings)
    return vectorstore

# Example usage (for testing independently)
if __name__ == "__main__":
    from data_ingestion import load_documents, split_documents
    
    docs = load_documents("data/sample_bio_docs/")
    chunks = split_documents(docs)
    
    embed_model = generate_embeddings()
    vs = create_vectorstore(chunks, embed_model)
    
    save_vectorstore(vs, "vectorstore/faiss_index")
    print("Vectorstore saved successfully!")
