from llama_index import download_loader, SimpleDirectoryReader, ServiceContext, GPTSimpleVectorIndex, LLMPredictor, GPTSimpleVectorIndex, PromptHelper
from pathlib import Path
import openai
import os
from langchain import OpenAI
import streamlit as st
from streamlit_chat import message
import time
from langchain.chat_models import ChatOpenAI


os.environ['OPENAI_API_KEY'] = 'super secret key'

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#build index from PDF
def pdf_to_index(pdf_path):
    PDFReader = download_loader('PDFReader')
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    index = GPTSimpleVectorIndex.from_documents(documents)
    index.save_to_disk('./test_chat.json')


#query index using GPT
def query_index(index_path, query_u):
    index = GPTSimpleVectorIndex.load_from_disk(index_path, service_context=service_context)
    response=index.query(query_u)
    st.session_state.past.append(query_u)
    st.session_state.generated.append(response.response)   


def clear_convo():
    st.session_state['past'] = []
    st.session_state['generated'] = []


def get_manual():
    manual = st.sidebar. radio("Choose a manual:", ("Yukon Test PDF", "LLM Research"))
    if manual== "Yukon Test":
        return "./test.json"


if __name__ == '__main__':
    st.set_page_config(page_title='PDF ChatBot', page_icon=':robot_face: ') 
    st.sidebar.title('Available manuals')
    clear_button = st.sidebar.button("Clear Conversation", key="clear") 

    if clear_button:
        clear_convo()

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'manual' not in st.session_state:
        st.session_state['manual'] = []

    with st.form(key='my_form', clear_on_submit=True):
        user_input= st.text_area("You:", key="input", height=75) 
        submit_button =st.form_submit_button(label="Submit")

    if user_input and submit_button:
        query_index(index_path='./test_chat.json',query_u=user_input)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1): 
            message(st.session_state['generated'][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + "user")
    
    st.sidebar. radio("Choose a manual:", ("Yukon Test PDF", "LLM Research"))