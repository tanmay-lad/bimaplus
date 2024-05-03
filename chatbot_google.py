import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
import streamlit as st
import textwrap
import logging
from htmlTemplates import css, bot_template, user_template

import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import os

from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004' #['models/embedding-001', 'models/text-embedding-004']
        title = "Custom query"
        return genai.embed_content(model=model,
                                    content=input,
                                    task_type="retrieval_document",
                                    title=title)["embedding"]

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt 

def setup_sidebar():
    sideb = st.sidebar
    query_1 = sideb.button("Key factors to consider while buying an insurance policy")
    query_2 = sideb.button("Compare health insurance companies")
    query_3 = sideb.button("Know more about tax benefits")
    policy_features = sideb.selectbox(
        "Learn more about key policy features",
        ("Claim settlement ratio", "Co-payment", "Room rent", "Restoration benefits"),
        index=None,
        placeholder="Select from below...",
    )
    sideb.write("TLDR! Policy documents are lengthy! Don't worry, BimaPlus chatbot reads through 100s of policy pages to \
                answer your questions.")
    
    company, policy = None, None
    company = sideb.selectbox(
        "Select the insurance company",
        ("SBI General Insurance", "HDFC ERGO"),
        index=None,
        placeholder="Select from below...",
    )
    #sideb.write(company)
    policy_list = {}
    policy_list["SBI General Insurance"] = ("Arogya Supreme", "Super Health")
    policy_list["HDFC ERGO"] = ("Optima Secure", "Optima Super Secure")
    if company:
        policy = sideb.selectbox(
            "Select the the insurance policy",
            policy_list[company],
            index=None,
            placeholder="Select from below...",
        )

def main():
    # Grab an API key
    load_dotenv()
    API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=API_KEY)

    # Fetching chromadb.Client()
    chroma_client = chromadb.PersistentClient(path="chroma_collections")
    collection_name = "PDF_files"
    db = chroma_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())

    # Set page configuration
    st.set_page_config(page_title="BimaPlus", page_icon=":insurance:")
    
    # Load custom CSS from file
    with open("header_style.css", "r") as f:
        header_css = f.read()
    
    # Initiate a chat session
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    chat = model.start_chat(history=[])
    
    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []
    display_messages = st.session_state["display_messages"]

    # Display fixed header
    header = st.container()
    header.title("Bimaplus: Find the best insurance for you")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    setup_sidebar()

    # Initialize session state for chat history if it doesn't exist
    if display_messages:
        for item in display_messages:
            role, parts = item.values()
            if role == "user":
                st.chat_message("user").markdown(parts[0])
            elif role == "model":
                st.chat_message("assistant").markdown(parts[0])

    # To ensure same passage in not passed every time
    old_passage = ""

    # Display fixed user input box at the bottom
    query = st.chat_input("Ask a Question from the PDF Files")
    answer = None

    # Chatbot conversation
    if query:
        st.chat_message("user").markdown(query)
        answer_area = st.chat_message("assistant").markdown("...")
        
        # Perform embedding search and generate prompt
        passage = get_relevant_passage(query, db)
        Markdown(passage)
        prompt = make_prompt(query, passage)
        Markdown(prompt)

        # Appending user inputs to chat history
        chat.history.append({'role':'user',
                             'parts':[query]})
        display_messages.append({'role':'user',
                         'parts':[query]})

        # Generate response from the model
        answer = model.generate_content(prompt)
        #if old_passage is not passage:
        #    answer = chat.send_message(prompt)
        #else:
        #    answer = chat.send_message(query)

        answer_area.markdown(answer.text)
        #Markdown(answer.text)
        old_passage = passage


        # Appending chatbot answers to chat history
        chat.history.append({'role':'model',
                             'parts':[answer.text]})
        display_messages.append({'role':'model',
                         'parts':[answer.text]})

if __name__ == '__main__':
    main()