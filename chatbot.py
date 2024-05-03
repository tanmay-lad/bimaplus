__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

def get_relevant_passage(query, db, user_selection):
    filenames = {'key_factors': "Health insurance basics.pdf", 
                 'get_feature_details': "Health insurance basics.pdf", 
                 'compare_insurers': "Health insurance basics.pdf", 
                 'claim_ratios': "Health insurance basics.pdf", 
                 'tax_benefits': "Health insurance basics.pdf", 
                 'select_policy_document': f"{user_selection['company']}_{user_selection['policy']}.pdf"
                 }
    
    if st.session_state.button_clicked == 'general':
        passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
    else:
        filename = filenames[st.session_state.button_clicked]
        print(filename)
        passage = db.get(ids=[filename])['documents'][0]
    
    return passage

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative Health Insurance bot that answers questions using text \
  from the reference passage included below. \
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
    # Setting up sidebar
    sideb = st.sidebar
    
    # Defining user_selection
    user_selection = {}
    
    user_selection['key_factors'] = sideb.button("Key factors to consider while buying an insurance policy")
    #sideb.write(user_selection['key_factors'])
    
    user_selection['policy_feature'] = sideb.selectbox(
        "Learn more about key policy features",
        ("Co-payment", "Room rent", "Restoration benefits", "Disease wise sublimits", "Free health checkups", 
         "Waiting periods", "Pre and post hopitalization care", "No claim bonus", "OPD benefits", "Daycare treatment"
         ),
        index=None,
        placeholder="Select from below...",
    )
    user_selection['get_feature_details'] = sideb.button("Get feature details")
    if user_selection['get_feature_details']:
        if not user_selection['policy_feature']:
            sideb.write("Please select policy feature to proceed")
    #sideb.write(user_selection['get_feature_details'])
    #sideb.write(user_selection['policy_feature'])
    
    user_selection['compare_insurers'] = sideb.button("Compare insurance companies")
    #sideb.write(user_selection['compare_insurers'])

    user_selection['claim_ratios'] = sideb.button("Claim settlement ratio vs Incurred claim ratio")
    #sideb.write(user_selection['claim_ratios'])
    
    user_selection['tax_benefits'] = sideb.button("Know more about tax benefits")
    #sideb.write(user_selection['tax_benefits'])
    
    user_selection['policy'] = None
    sideb.write("TLDR! Policy documents are lengthy! Don't worry, BimaPlus chatbot reads through 100s of policy pages to \
                answer your questions.")
    user_selection['company'] = sideb.selectbox(
        "Select insurance company",
        ("SBI", "HDFC"),
        index=None,
        placeholder="Select from below...",
    )
    #sideb.write(user_selection['company'])
    policy_list = {}
    policy_list["SBI"] = ("Arogya Supreme", "Super Health")
    policy_list["HDFC"] = ("Optima Secure", )
    if user_selection['company']:
        user_selection['policy'] = sideb.selectbox(
            "Select company policy",
            policy_list[user_selection['company']],
            index=None,
            placeholder="Select from below...",
        )
    #sideb.write(user_selection['policy'])
    user_selection['select_policy_document'] = sideb.button("Search policy document")
    if user_selection['select_policy_document']:
        if not (user_selection['company'] and user_selection['policy']):
            sideb.write("Please select insurance company and policy to proceed")
    #sideb.write(user_selection['select_policy_document'])

    sideb.write("Don't know what you are looking for? BimaPlus chatbot will guide you with the help of our rich database.")
    user_selection['ask_me_anything'] = sideb.button("Ask me anything")

    return user_selection

def get_relevant_query(user_selection):
    query = None
    
    if(user_selection['key_factors']):
        query = "Briefly explain the key factors to consider while buying an insurance policy. \
                Please also rate each factor on the scale from 1 to 10 - 1 being must not have and 10 being miust have feature."
        st.session_state.button_clicked = 'key_factors'
    
    if(user_selection['get_feature_details'] and user_selection['policy_feature']):
        query = f"Explain in detail about {user_selection['policy_feature']} feature while buying an insurance policy. \
                Please include an explanation, precautions, sample example, recommendation or best practices regading this feature \
                and criticality of this feature."
        st.session_state.button_clicked = 'get_feature_details'
    
    if(user_selection['compare_insurers']):
        query = "Share the comparison of multiple insurance companies across different parameters in table format. \
                Please also include a column with desired values for each parameter. \
                At the end, give top 3 insurance companies based on the analysis."
        st.session_state.button_clicked = 'compare_insurers'

    if(user_selection['claim_ratios']):
        query = "Explain in brief the concepts and difference between them: claim settlement ratio and incurred claim ratio. \
                Please give a list of 5-10 top companies with historical ratios in table format. \
                Also talk about the ideal ratio values while buying an insurance policy."
        st.session_state.button_clicked = 'claim_ratios'
    
    if(user_selection['tax_benefits']):
        query = "Explain in detail about the tax benefits under section 80D while buying an insurance policy. \
                Please include an explanation, precautions, sample example, and recommendation or best practices."
        st.session_state.button_clicked = 'tax_benefits'

    if (user_selection['select_policy_document'] and user_selection['policy']):
        query = f"Briefly mention the key features or highlights of {user_selection['company']} {user_selection['policy']} policy? \
                Also include a table with columns having policy variants if given and rows having following key features: \
                Co-payment, restrictions on room rent, disease wise sublimits, waiting periods, pre-post hospitalization care, \
                restoration benefits, daycare treatment, no claim bonus, free health checkups, OPD benefits, etc. \
                Please be very brief in the table - mention numbers wherever applicable, \
                in addition, also mention in another column on the ideal values for each of these features."
        st.session_state.button_clicked = 'select_policy_document'

    if(user_selection['ask_me_anything']):
        query = "Briefly explain the key factors to consider while buying an insurance policy. \
                Please also rate each factor on the scale from 1 to 10 - 1 being must not have and 10 being miust have feature."
        st.session_state.button_clicked = 'general'

    return query

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
    
    # Initiate a chat session
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    chat = model.start_chat(history=[])
    
    if "display_messages" not in st.session_state:
        st.session_state["display_messages"] = []
    display_messages = st.session_state["display_messages"]

    # Define persisting variable button_clicked to ensure user can ask multiple questions to the same file
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = 'general'
    print(st.session_state.button_clicked)

    # Display fixed header
    header = st.container()
    header.title("Bimaplus: Find the best insurance for you")
    
    # Initialize session state for chat history if it doesn't exist
    if display_messages:
        for item in display_messages:
            role, parts = item.values()
            if role == "user":
                st.chat_message("user").markdown(parts[0])
            elif role == "model":
                st.chat_message("assistant").markdown(parts[0])

    # Get query from user_selection
    user_selection = setup_sidebar()
    query_selection = get_relevant_query(user_selection)

    # Display fixed user input box at the bottom
    query_input = st.chat_input("Ask a Question from the PDF Files")
    
    # Query based on user selection from given options or user question input
    query, answer = None, None
    #old_passage = ""
    if query_selection:
        query = query_selection
    elif query_input:
        query = query_input

    # Chatbot conversation
    if query:
        st.chat_message("user").markdown(query)
        answer_area = st.chat_message("assistant").markdown("...")
        
        # Perform embedding search and generate prompt
        print(st.session_state.button_clicked)
        passage = get_relevant_passage(query, db, user_selection)
        Markdown(passage)
        prompt = make_prompt(query, passage)
        Markdown(prompt)

        # Generate response from the model
        answer = model.generate_content(prompt)
        #if old_passage is not passage:
        #    answer = chat.send_message(prompt)
        #else:
        #    answer = chat.send_message(query)

        answer_area.markdown(answer.text)
        #old_passage = passage

        # Appending values to chat history
        chat.history.append({'role':'user',
                             'parts':[query]})
        chat.history.append({'role':'model',
                             'parts':[answer.text]})
        
        # Appending values to chat history
        display_messages.append({'role':'user',
                         'parts':[query]})
        display_messages.append({'role':'model',
                         'parts':[answer.text]})

if __name__ == '__main__':
    main()