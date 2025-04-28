import streamlit as st
from embedding_generation import generate_embeddings
from retriever import load_vectorstore, retrieve_relevant_documents
from llm_response import initialize_llm, generate_answer

# Initialize models
embedding_model = generate_embeddings()
vectorstore = load_vectorstore("vectorstore/faiss_index", embedding_model)
llm = initialize_llm()

# Streamlit app layout
st.title("Bioinformatics RAG Assistant")
st.subheader("Ask your bio research questions!")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        with st.spinner('Retrieving information...'):
            retrieved_docs = retrieve_relevant_documents(query, vectorstore, k=4)
            answer = generate_answer(query, retrieved_docs, llm)

        st.success("Answer generated:")
        st.write(answer)
    else:
        st.warning("Please enter a question to proceed.")
