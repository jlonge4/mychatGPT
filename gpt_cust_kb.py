from llama_index import download_loader, SimpleDirectoryReader, ServiceContext, GPTSimpleVectorIndex, LLMPredictor, GPTSimpleVectorIndex, PromptHelper
from pathlib import Path
import openai
import os
from langchain import OpenAI
import streamlit as st
from streamlit_chat import message
import time
from langchain.chat_models import ChatOpenAI
import glob

index_files = []

def set_openai_api_key():
    api_key = st.sidebar.text_input("OpenAI API key", type="default")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

#select PDF
def upload_pdf():
    pdf_file = st.sidebar.file_uploader("Upload PDF file")
    if pdf_file is not None:
        pdf_name = pdf_file.name
        pdf_path = f"PDFs/{pdf_name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.success(f"{pdf_name} successfully uploaded.")
        pdf_to_index(pdf_path, pdf_name)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#build index from PDF
def pdf_to_index(pdf_path, pdf_name):
    PDFReader = download_loader('PDFReader')
    loader = PDFReader()
    documents = loader.load_data(pdf_path)
    index = GPTSimpleVectorIndex.from_documents(documents)
    index_path = f"PDFs/{pdf_name[:-4]}.json"
    index.save_to_disk(index_path)
    global index_files
    index_files=glob.glob('/PDFs/*.json')


#query index using GPT
def query_index(index_path, query_u):
    index = GPTSimpleVectorIndex.load_from_disk(os.path.join(dir_path, index_path), service_context=service_context)
    response=index.query(query_u)
    st.session_state.past.append(query_u)
    st.session_state.generated.append(response.response)   


def clear_convo():
    st.session_state['past'] = []
    st.session_state['generated'] = []


if __name__ == '__main__':
    st.set_page_config(page_title='PDF ChatBot', page_icon=':robot_face: ') 
    st.sidebar.title('Available manuals')
    clear_button = st.sidebar.button("Clear Conversation", key="clear") 

    dir_path = Path("PDFs/")
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)) and path.endswith('json'):
            index_files.append(path)

    #index_files= glob.glob('\PDFs\*json')
    selected_manual = st.sidebar.radio("Choose a PDF:", index_files)

    set_openai_api_key()
    upload_pdf()

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
        query_index(selected_manual, query_u=user_input)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1): 
            message(st.session_state['generated'][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + "user")
