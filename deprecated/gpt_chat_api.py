from llama_index import download_loader, SimpleDirectoryReader, ServiceContext, LLMPredictor, GPTVectorStoreIndex, PromptHelper, StorageContext, load_index_from_storage
from pathlib import Path
import openai
import os
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI
import uvicorn 


key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = key
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

PATH_TO_DOCS = 'DOCS_DIR'
PATH_TO_PDFS = 'PDFS_DIR'
PATH_TO_INDEXES = 'GPT_INDEXES'
if not os.path.exists(PATH_TO_INDEXES):
        os.makedirs(PATH_TO_INDEXES)

#build index from PD
def pdf_to_index(file):
    if not os.path.exists(PATH_TO_INDEXES):
        os.makedirs(PATH_TO_INDEXES)

        pdf_path=f'{PATH_TO_PDFS}/{file}'
        save_path=f'{PATH_TO_INDEXES}/{file}'
    else:
        pdf_path=f'{PATH_TO_PDFS}/{file}'
        save_path=f'{PATH_TO_INDEXES}/{file}'

    PDFReader = download_loader('PDFReader')
    loader = PDFReader()
    documents = loader.load_data(file=Path(pdf_path))
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=save_path)


#build index from docx
def docx_to_index(file): 
    if not os.path.exists(PATH_TO_INDEXES):
        os.makedirs(PATH_TO_INDEXES)
        docx_path=f'{PATH_TO_DOCS}/{file}'
        save_path=f'{PATH_TO_INDEXES}/{file}'
    else:
        docx_path=f'{PATH_TO_DOCS}/{file}'
        save_path=f'{PATH_TO_INDEXES}/{file}' 

    DocxReader = download_loader("DocxReader")
    loader = DocxReader()
    documents = loader.load_data(file=Path(docx_path))
    index = GPTVectorStoreIndex.from_documents(documents) 
    index.storage_context.persist(persist_dir=save_path) 


#query index using GPT
def query_index(index_to_use, query_u):
    storage_context = StorageContext.from_defaults(persist_dir=f"{PATH_TO_INDEXES}/{index_to_use}")
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine()
    response_q = query_engine.query(query_u)
    response = {"indexed_doc": f"{index_to_use}", 
                "response": f"{response_q}",
                "status": "completed"}
    return response


app = FastAPI()

@app.post("/filename/{file_name}")   
async def create_index(
     file_name: str
):
    file_name = file_name.replace('"', '')
    extension = file_name[-4:]
    print(file_name)
        
    match extension:
        case 'docx':
            docx_to_index(file=file_name)
            
        case '.pdf':
            pdf_to_index(file=file_name)
    
    response = {"indexed_doc": f"{file_name}", "status": "completed"}
    return response, 200


@app.get("/filename/{file_name}/query/{query_string}")
async def query_doc(
    file_name: str, query_string: str
):
    file_name = file_name.replace('"', '')
    response = query_index(index_to_use=file_name, query_u=query_string)
    return response, 200

if __name__ == '__main__':
    uvicorn.run("gpt_chat_api:app",
                host="0.0.0.0",
                port=8432,
                reload=True,
                )
    
    