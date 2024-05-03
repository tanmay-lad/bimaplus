import streamlit as st
import faiss
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Call set_page_config() as the first Streamlit command in your script
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
st.write(css, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = VoyageAIEmbeddings(batch_size=64, model="voyage-lite-02-instruct")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, conversation_chain):
    # Get response from the conversation chain
    response = conversation_chain.invoke({'question': user_question})
    chat_history = response['chat_history']
    
    # Display conversation messages
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.write("Please upload PDFs first and click 'Process' button.")
        else:
            handle_userinput(user_question, st.session_state.conversation)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                #conversation_chain = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
