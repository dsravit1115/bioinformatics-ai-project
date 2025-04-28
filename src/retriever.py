from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

def load_vectorstore(path: str, embedding_model: SentenceTransformerEmbeddings) -> FAISS:
    """
    Loads the FAISS vectorstore from a given path.
    """
    vectorstore = FAISS.load_local(path, embedding_model)
    return vectorstore

def retrieve_relevant_documents(query: str, vectorstore: FAISS, k: int = 4):
    """
    Given a user query, retrieves top-k most relevant documents from vectorstore.
    """
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# Example usage (for independent testing)
if __name__ == "__main__":
    from embedding_generation import generate_embeddings

    embedding_model = generate_embeddings()
    vs = load_vectorstore("vectorstore/faiss_index", embedding_model)

    query = "What are the genetic markers associated with breast cancer?"
    results = retrieve_relevant_documents(query, vs, k=3)

    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---\n")
        print(doc.page_content)
