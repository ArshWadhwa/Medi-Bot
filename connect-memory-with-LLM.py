import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# setUp LLM

# connect LLM with FAISS
HF_TOKEN = os.environ.get("HF_TOKEN")

print("✅ Token loaded successfully!")

HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_LLM(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=512
        )
        print("✅ Using HuggingFace Endpoint")
        return llm
    except Exception as e:
        print(f"❌ Error with HuggingFace Endpoint: {e}")
        print("💡 You can also use a local model or OpenAI instead")
        return None

# create chain

# Custom Medical Prompt Template
DB_FAISS_PATH="vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """You are a medical assistant that ONLY answers questions based on the provided context from medical documents. 

IMPORTANT RULES:
1. ONLY use information from the provided context to answer
2. If the question cannot be answered using the provided context, say: "I cannot answer this question based on the available medical documents. Please ask a question related to the medical information in the documents."
3. Do NOT use any external knowledge or general medical information
4. Do NOT make up or infer information not explicitly stated in the context
5. If the context doesn't contain enough information, say so clearly

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

# Load database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

# create QA chain
llm = load_LLM(HUGGINGFACE_REPO_ID)

if llm is None:
    print("❌ Failed to load LLM. Please check your token and try again.")
    exit(1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# now invoke
user_query=input("Write query here: ")
response=qa_chain.invoke({'query': user_query})
print("Result:", response["result"])
print("Source docs:", response["source_documents"])
