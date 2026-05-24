from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA
from pinecone import Pinecone
from src.prompt import *
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

# embeddings = download_hugging_face_embeddings()

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "medical-chatbot"

# docsearch = PineconeVectorStore.from_texts(
#     [t.page_content for t in text_chunks],
#     embedding=embeddings,
#     index_name=index_name
# )

# PROMPT = PromptTemplate(
#     template=TEMMPLATE,
#     input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

# llm = CTransformers(
#     model="meta-llama/Meta-Llama-3-8B-Instruct",
#     model_type="llama",
#     config={
#         "max_new_tokens": 512,
#         "temperature": 0.8
#     }
# )

# qa = RetrievalQA.from_chain_type(
#     LLm = llm,
#     chain_type = "stuff",
#     retriever = docsearch.as_retriever(search_kwargs = {'K': 2}),
#     return_source_documents = True,
#     chain_type_kwargs = chain_type_kwargs
# )

# @app.route("/")
# def index():
#     return render_template('chat.html')

# if __name__ == "__main__":
#     app.run(debug = True)

# from flask import Flask, render_template, request, jsonify
# from src.helper import download_hugging_face_embeddings
# from src.prompt import *

# from dotenv import load_dotenv
# import os

# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.llms import CTransformers
# # from langchain.chains import RetrievalQA
# from langchain_core.prompts import PromptTemplate

# Initialize Flask App
app = Flask(__name__)

# Load Environment Variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Download Embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Connect Existing Pinecone Index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Prompt Template
# PROMPT = PromptTemplate(
#     template=TEMPLATE,
#     input_variables=["context", "question"]
# )

PROMPT = PromptTemplate(
    template=TEMMPLATE,
    input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

chain_type_kwargs = {
    "prompt": PROMPT
}

# Load LLM
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.8
    }
)

# Create QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        search_kwargs={"k": 2}
    ),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Home Route
@app.route("/")
def index():
    return render_template("chat.html")


# Chat Route
@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = "msg"
    print(input)

    result = qa.invoke({"query": msg})

    # print(result)

    return str(result["result"])


# Run App
if __name__ == "__main__":
    app.run(debug=True)