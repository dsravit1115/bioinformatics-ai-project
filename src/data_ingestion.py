import os
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents(directory: str) -> List[Document]:
    """
    Loads all .txt documents from a directory and returns a list of Document objects.
    """
    documents = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
    
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits documents into smaller chunks for embedding generation.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

# Example usage (for testing this file directly)
if __name__ == "__main__":
    docs = load_documents("data/sample_bio_docs/")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Split into {len(chunks)} chunks")
