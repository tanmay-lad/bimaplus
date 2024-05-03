import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from htmlTemplates import css, bot_template, user_template

def user_input(user_question):
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = VoyageAIEmbeddings(batch_size=128, model="voyage-lite-02-instruct")  # ["voyage-lite-02-instruct","voyage-large-2"]

    # Load a FAISS vector database from a local file
    saved_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search in the vector database based on the user question
    docs = saved_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Append user question and bot response to conversation history
    st.session_state.conversation_history.append({"user": user_question, "bot": response["output_text"]})

    # Print the response to the console
    print(response)

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    #st.write("Reply: ", response["output_text"])

def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def main():
    load_dotenv()

    # Load custom CSS from file
    with open("header_style.css", "r") as f:
        header_css = f.read()
    
    # Set the API key for Voyage AI
    import voyageai
    voyageai.api_key="pa-j-zRJc8C_z7Cj971b1J07L5fukvsJ_PAjI8XnSmwCqU"

    # Set the API key for OpenAI
    import openai
    openai.api_key = "sk-proj-ezbVvEfeV7ku40fV7XOyT3BlbkFJToo7Jnsq9gBnrKJFxBE5"
    
    # Initialize a session state to store conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Set page configuration
    st.set_page_config(page_title="BimaPlus", page_icon=":insurance:")
    
    # Display fixed header
    #st.title("Chat with PDF using Gemini")
    header = st.container()
    header.title("Bimaplus: Find the best insurance for you")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Display fixed user input box at the bottom
    user_question = st.chat_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)

    # Show conversation history
    for interaction in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(interaction["user"])
        with st.chat_message("assistant"):
            st.write(interaction["bot"])
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()