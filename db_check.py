import google.generativeai as genai
from dotenv import load_dotenv
import os

import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)

for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# Fetching chromadb.Client()
chroma_client = chromadb.PersistentClient(path="chroma_collections")
collections_list = chroma_client.list_collections()
print("List of collections:")

for collection in collections_list:
  print(collection)
  #print(collection.count())
  #chroma_client.delete_collection(name=collection.name)
  #collection.delete(ids = ["1"])
  print(pd.DataFrame(collection.peek(collection.count())))

#db = chroma_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())