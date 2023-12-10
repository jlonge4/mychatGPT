import streamlit as st
from haystack.pipelines import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import (
    EmbeddingRetriever,
    TextConverter,
    FileTypeClassifier,
    PDFToTextConverter,
    MarkdownConverter,
    DocxToTextConverter,
    PreProcessor,
    PromptNode,
)
import os


def get_doc_store():
    try:
        document_store = FAISSDocumentStore.load(
            index_path="my_index.faiss", config_path="my_config.json"
        )
    except Exception:
        document_store = FAISSDocumentStore(embedding_dim=768)
        document_store.save(index_path="my_index.faiss", config_path="my_config.json")
    return document_store


def indexing_pipe(filename):
    document_store = get_doc_store()
    file_type_classifier = FileTypeClassifier()

    text_converter = TextConverter()
    pdf_converter = PDFToTextConverter()
    md_converter = MarkdownConverter()
    docx_converter = DocxToTextConverter()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=300,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/msmarco-bert-base-dot-v5",
        model_format="sentence_transformers",
    )

    # indexing pipeline
    p = Pipeline()
    p.add_node(
        component=file_type_classifier, name="FileTypeClassifier", inputs=["File"]
    )
    p.add_node(
        component=text_converter,
        name="TextConverter",
        inputs=["FileTypeClassifier.output_1"],
    )
    p.add_node(
        component=pdf_converter,
        name="PdfConverter",
        inputs=["FileTypeClassifier.output_2"],
    )
    p.add_node(
        component=md_converter,
        name="MarkdownConverter",
        inputs=["FileTypeClassifier.output_3"],
    )
    p.add_node(
        component=docx_converter,
        name="DocxConverter",
        inputs=["FileTypeClassifier.output_4"],
    )
    p.add_node(
        component=preprocessor,
        name="Preprocessor",
        inputs=["TextConverter", "PdfConverter", "MarkdownConverter", "DocxConverter"],
    )
    p.add_node(component=retriever, name="Retriever", inputs=["Preprocessor"])
    p.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])

    os.makedirs("uploads", exist_ok=True)
    # Save the file to disk
    file_path = os.path.join("uploads", filename.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    # Run pipeline on document and add metadata to include doc name
    p.run(
        file_paths=["uploads/{0}".format(filename.name)],
        meta={"document_name": filename.name},
    )

    # Once documents are ran through the pipeline, use this to add embeddings to the datastore
    document_store.save(index_path="my_index.faiss", config_path="my_config.json")
    print(
        f"Docs match embedding count: {document_store.get_document_count() == document_store.get_embedding_count()}"
    )


def rag_pipeline(query):
    key = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_API_KEY'] = key
    document_store = get_doc_store()
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        top_k=5,
    )
    prompt_node = PromptNode(
        model_name_or_path="gpt-3.5-turbo",
        default_prompt_template="deepset/question-answering",
        api_key=key,
        max_length = 250,
    )

    p = Pipeline()
    p.add_node(component=retriever, name="EmbeddingRetriever", inputs=["Query"])
    p.add_node(
        component=prompt_node, name="QAPromptNode", inputs=["EmbeddingRetriever"]
    )
    res = p.run(query=query)
    return res


def clear_convo():
    st.session_state["messages"] = []


def init():
    st.set_page_config(page_title="GPT RAG", page_icon=":robot_face: ")
    st.sidebar.title("Available Indexes")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


if __name__ == "__main__":
    init()

    clear_button = st.sidebar.button(
        "Clear Conversation", key="clear", on_click=clear_convo
    )
    file = st.file_uploader("Choose a file to index...", type=["docx", "pdf", "txt"])
    clicked = st.button("Upload File", key="Upload")
    if file and clicked:
        with st.spinner("Wait for it..."):
            document_store = indexing_pipe(file)
        st.success("Indexed {0}!".format(file.name))

    user_input = st.chat_input("Say something")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        res = rag_pipeline(user_input)
        st.session_state.messages.append({"role": "assistant", "content": res["results"][0]})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
