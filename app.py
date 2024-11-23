from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template
from langchain.prompts import *
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone

import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)  
index_name = "medical-chatbot"
index = pc.Index(index_name) 
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}
vectordb = PineconeStore(index, embeddings,"text").as_retriever(search_kwargs={"k": 2})
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    user_input = request.form["msg"]
    response = qa({"query": user_input})
    return str(response['result'])


if __name__ == "__main__":
    app.run(debug=True, port=4000)