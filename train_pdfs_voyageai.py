import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import time
import os

# Set the API key for Voyage AI
import voyageai
voyageai.api_key="pa-j-zRJc8C_z7Cj971b1J07L5fukvsJ_PAjI8XnSmwCqU"

# File path for saving and loading Faiss index
FAISS_INDEX_PATH = "faiss_index"

# Function to extract text from PDF files
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to train embedding model and create vector store
def train_embedding_model(text_chunks, embeddings):
    st.write("Training embedding model...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.write("Training complete!")
    return vectorstore

def main():
    load_dotenv()

    st.title("Chatbot Admin Panel")

    # File uploader for admin to upload PDF files
    pdf_files = st.file_uploader("Upload PDF files for training", accept_multiple_files=True)

    if pdf_files:
        if st.button("Train Embedding Model"):
            st.write("Processing PDF files...")
            raw_text = get_pdf_text(pdf_files)
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=8000, chunk_overlap=1200, length_function=len)
            text_chunks = text_splitter.split_text(raw_text)

            # Train embedding model and create vector store for the new data
            embeddings = VoyageAIEmbeddings(batch_size=8, model="voyage-lite-02-instruct") # ["voyage-lite-02-instruct","voyage-large-2"]
            new_db = train_embedding_model(text_chunks, embeddings)
            
            # Check if the local Faiss index file exists
            if os.path.exists(FAISS_INDEX_PATH):
                # Load the local Faiss index
                saved_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                # Merge the new index with the existing one
                saved_db.merge_from(new_db)
                # Save the merged index back to the local storage
                saved_db.save_local(folder_path=FAISS_INDEX_PATH)
            else:
                # Save the new index to the local storage
                new_db.save_local(folder_path=FAISS_INDEX_PATH)

            st.success("Embedding model trained and Faiss index updated.")

    st.sidebar.title("Training Progress")
    if pdf_files:
        st.sidebar.text("Training in progress...")
    else:
        st.sidebar.text("No PDF files uploaded.")

if __name__ == "__main__":
    main()