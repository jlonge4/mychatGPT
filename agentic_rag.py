import streamlit as st
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.utils import Secret
from pathlib import Path
import openai
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
import concurrent.futures
import os
from utils.custom_converters import DocxToTextConverter


@st.cache_resource()
def get_doc_store():
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return document_store


def write_documents(file):
    pipeline = Pipeline()

    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    else:
        pipeline.add_component("converter", PyPDFToDocument())

    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter", DocumentSplitter(split_by="word", split_length=350)
    )
    pipeline.add_component(
        "embedder", OpenAIDocumentEmbedder(api_key=Secret.from_token(openai.api_key))
    )
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder.documents", "writer")

    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    pipeline.run({"converter": {"sources": [Path(file_path)]}})
    st.success("Indexed Document!")


def chunk_documents(file):
    pipeline = Pipeline()
    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    else:
        pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter", DocumentSplitter(split_by="word", split_length=3000)
    )

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    file_path = os.path.join("uploads", file.name)
    docs = pipeline.run({"converter": {"sources": [file_path]}})
    return [d.content for d in docs["splitter"]["documents"]]


def query_pipeline(query):
    query_pipeline = Pipeline()
    query_pipeline.add_component(
        "text_embedder", OpenAITextEmbedder(Secret.from_token(openai.api_key))
    )
    query_pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4)
    )
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    result = query_pipeline.run({"text_embedder": {"text": query}})
    return result["retriever"]["documents"]


def query_router(query):
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional decision making query router bot for a chatbot system that decides whether a user's query requires a summary, 
    a retrieval of extra information from a vector database, or a simple greeting/gratitude/salutation response. If the query
    requires a summary, you will reply with only "(1)". If the query requires extra information, you will reply with only "(2)".
    If the query requires a simple greeting/gratitude/salutation/ or an answer to a follow up question based on conversation history 
    response, you will reply with only "(3)"."""

    instruction = f"""You are given a user's query in the <query> field. You are responsible for routing the query to the appropriate
    choice as described in the system response. <query>{query}</query> You are also given the history of the conversation in the <history>{st.session_state.messages}</history> field."""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response


def map_summarizer(query, chunk):
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional corpus summarizer for a chatbot system. 
    You are responsible for summarizing a chunk of text according to a user's query."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input
    using the provided chunk in the <chunk> tags: <query>{query}</query>\n <chunk>{chunk}</chunk>"""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    print("chunk_summarizer")
    response = generator.run(messages)
    return response


def reduce_summarizer(query, analyses):
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional corpus summarizer for a chatbot system. 
    You are responsible for summarizing a list of summaries according to a user's query."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input
    using the provided list of summaries in the <chunk> tags: <query>{query}</query>\n <chunk>{analyses}</chunk>"""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    print("chunk_summarizer")
    response = generator.run(messages)
    return response


def simple_responder(query):
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional greeting/gratitude/salutation/ follow up responder for a chatbot system. 
    You are responsible for responding politely to a user's query."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input: <query>{query}</query>"""

    messages = []
    history = st.session_state.messages
    messages.append(ChatMessage.from_system(system))
    for i in range(0, len(history) - 1, 2):
        messages.append(ChatMessage.from_user(history[i]["content"]))
        messages.append(ChatMessage.from_assistant(history[i + 1]["content"]))
    messages.append(ChatMessage.from_user(instruction))
    print("simple_responder")
    response = generator.run(messages)
    return response


def summary_tool(query, file):
    chunks = chunk_documents(file)
    # write async function to call chat generator using concurrent futures
    futures = []
    analyses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for chunk in chunks:
            futures.append(executor.submit(map_summarizer, query, chunk))
        for future in concurrent.futures.as_completed(futures):
            analyses.append(future.result())
        return reduce_summarizer(query, analyses)


def context_tool(query):
    context = query_pipeline(query)
    context = [c.content for c in context]
    generator = OpenAIChatGenerator(
        api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo"
    )

    system = """You are a professional Q/A responder for a chatbot system. 
    You are responsible for responding to a user query using ONLY the context provided within the <context> tags."""

    instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input using only the context
    in the <context> field:\n <query>{query}</query>\n <context>{context}</context>"""

    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response


class RAGAgent:
    def __init__(self):
        self.loops = 0

    def invoke_agent(self, query, file):
        intent = query_router(query)["replies"][0].content.strip()

        if intent == "(1)":
            st.success("Retrieving Summary...")
            response = summary_tool(query, file)["replies"][0].content
        elif intent == "(2)":
            st.success("Retrieving Context...")
            response = context_tool(query)["replies"][0].content
        elif intent == "(3)":
            response = simple_responder(query)["replies"][0].content
        return response


def clear_convo():
    st.session_state["messages"] = []


def init():
    st.set_page_config(page_title="GPT RAG", page_icon=":robot_face: ")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


if __name__ == "__main__":
    init()

    document_store = get_doc_store()

    agent = RAGAgent()

    # streamlit components
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    openai.api_key = api_key
    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear", on_click=clear_convo
    )

    file = st.file_uploader("Choose a file to index...", type=["docx", "pdf", "txt"])
    clicked = st.button("Upload File", key="Upload")
    if file and clicked:
        with st.spinner("Wait for it..."):
            write_documents(file)

    user_input = st.chat_input("Say something")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        res = agent.invoke_agent(user_input, file)
        st.session_state.messages.append({"role": "assistant", "content": res})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
