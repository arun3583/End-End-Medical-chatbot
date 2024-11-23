import os

from dotenv import load_dotenv
from pinecone import Pinecone
from src.helper import download_hugging_face_embeddings, load_pdf, text_split

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)  
index_name = "medical-chatbot"
index = pc.Index(index_name)

# uploading embeded data to pinecone

for i, t in zip(range(len(text_chunks)), text_chunks):
    query_result = embeddings.embed_query(t.page_content)
    index.upsert(
    vectors=[
        {
            "id": str(i),  # Convert i to a string
            "values": query_result, 
            "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
        }
    ],
    namespace="real" 
    )
    index.describe_index_stats()


