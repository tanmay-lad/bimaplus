import textwrap
import chromadb
import numpy as np
import pandas as pd
import streamlit as st

import google.generativeai as genai
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
import os

from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
from PyPDF2 import PdfReader

# Function to extract text from PDF files
def get_pdf_text(pdf_files):
    documents, file_names = [], []
    for pdf_file in pdf_files:
        doc_text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            doc_text += page.extract_text()
        documents.append(doc_text)
        file_names.append(pdf_file.name)
    return [documents, file_names]

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004' #['models/embedding-001', 'models/text-embedding-004']
        title = "Custom query"
        return genai.embed_content(model=model,
                                    content=input,
                                    task_type="retrieval_document",
                                    title=title)["embedding"]

def update_db(documents, name, chroma_client, file_names):
  #chroma_client = chromadb.Client()
  db = chroma_client.get_or_create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
  db_size = db.count()

  for i, d in enumerate(documents):
    file_name = file_names[i]
    db.add(
      documents=d,
      ids=file_name
    )
  return db

def main():
    # Grab an API key
    load_dotenv()
    API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=API_KEY)

    # File uploader for admin to upload PDF files
    st.title("Chatbot Admin Panel")
    pdf_files = st.file_uploader("Upload PDF files for training", accept_multiple_files=True)

    if pdf_files:
        if st.button("Train Embedding Model"):
            st.write("Processing PDF files...")
            [documents, file_names] = get_pdf_text(pdf_files)
            print(file_names)

            # Set up the DB
            chroma_client = chromadb.PersistentClient(path="chroma_collections")
            db = update_db(documents, "PDF_files", chroma_client, file_names)
            #print(pd.DataFrame(db.peek(len(documents))))

            st.success("Embedding model trained and database updated.")

    # Sidebar to show progress
    st.sidebar.title("Training Progress")
    if pdf_files:
        st.sidebar.text("Training in progress...")
    else:
        st.sidebar.text("No PDF files uploaded.")

if __name__ == "__main__":
    main()