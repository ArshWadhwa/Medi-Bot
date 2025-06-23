import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint


# Load environment variables from .env file
load_dotenv()

# Path to local FAISS DB
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the FAISS vector store with caching
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load the HuggingFace LLM
def load_LLM(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm

# Create a custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Main Streamlit app
def main():
    st.title("🩺 Medical Q&A Chatbot")

    if 'message' not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask a medical question based on your uploaded documents...")

    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.message.append({'role': 'user', 'content': prompt})

        # Custom prompt template for medical domain
        CUSTOM_PROMPT_TEMPLATE = """You are a helpful medical assistant. Your task is to answer questions based *only* on the provided context from medical documents.

**Instructions:**
1.  Read the context carefully. It contains excerpts from medical texts.
2.  Answer the user's question using **only** the information found in the context.
3.  If the context contains information relevant to the question, synthesize it into a clear and helpful answer.
4.  If the context does **not** contain enough information to fully answer the question, do not invent a complete answer. Instead, state what you *can* find in the documents and mention that a comprehensive answer is not available. For example: "Based on the documents, I can tell you that [summary of findings], but a complete guide on how to cure cancer is not provided."
5.  Do not use any of your own general knowledge. Stick strictly to the provided text.
6. Provide the page number of source as well.

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        if not HF_TOKEN:
            st.error("HuggingFace API token not found. Set HF_TOKEN environment variable.")
            return

        try:
            vector_store = get_vectorstore()

            if vector_store is None:
                st.error("❌ Failed to load the vector store.")
                return

            llm = load_LLM(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response.get("source_documents", [])

            result_to_show = result + "\n\n---\n📄 **Source Documents**:\n" + "\n\n".join(
                [doc.page_content for doc in source_documents]
            )

            with st.chat_message('assistant'):
                st.markdown(result_to_show)

            st.session_state.message.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
