import os
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    """
    Initializes the OpenAI GPT-4 LLM model.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,
        openai_api_key=openai_api_key
    )
    return llm

def generate_answer(query: str, retrieved_docs: List[Document], llm) -> str:
    """
    Generates an answer from the LLM based on user query and retrieved context.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_template = """
You are a helpful bioinformatics research assistant. Based on the context below, answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format_messages(context=context, question=query)

    response = llm(final_prompt)
    return response.content

# Example usage (for testing independently)
if __name__ == "__main__":
    from retriever import retrieve_relevant_documents, load_vectorstore
    from embedding_generation import generate_embeddings

    query = "Explain the role of BRCA1 gene in cancer development."

    embedding_model = generate_embeddings()
    vs = load_vectorstore("vectorstore/faiss_index", embedding_model)

    retrieved_docs = retrieve_relevant_documents(query, vs, k=3)

    llm = initialize_llm()
    answer = generate_answer(query, retrieved_docs, llm)

    print("\nGenerated Answer:\n")
    print(answer)
