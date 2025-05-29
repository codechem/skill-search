import os
from chromadb import HttpClient
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI
import streamlit as st
import pandas as pd
from pymupdf import Document


def get_cv_rankings(cv_collection, job_listing: str, n_results) -> dict:
    cv_results = cv_collection.query(
        query_texts=[job_listing],
        n_results=n_results,
    )
    filename_by_document = {
        filename: document
        for filename, document in zip(cv_results["ids"][0], cv_results["documents"][0])
    }
    initial_prompt = f"""
    Rank the following CVs based on their relevance to the job requirements.
    Job requirements are:
    {job_listing}
    The CVs are:{filename_by_document}
    Provide the rankings by using the filename of the CVs.
    Provide a short explanation for the ranking.
    Provide the rankings and explanation in the following format:
    {{
        filename1:explanation1,
        filename2:explanation2,
        filename3:explanation3,
    }}
    Don't provide any other information.
    Don't provide any other text.
    """
    openai_client = OpenAI()
    response = openai_client.responses.create(model="gpt-4.1-mini", input=initial_prompt)
    dict_response = eval(response.output_text)
    return dict_response


def process_uploaded_file(uploaded_file: bytes):
    pdf_doc = Document(stream=uploaded_file)
    text = "".join(page.get_text() for page in pdf_doc)
    return text.replace("⎽", "").replace("\\n", "").strip()


def get_cv_collection():
    chromadb_client = HttpClient(host=os.environ["DB_HOST"], port=8000)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )
    cv_collection = chromadb_client.get_or_create_collection(
        name="cvs",
        embedding_function=openai_ef,
    )
    return cv_collection

cv_collection = get_cv_collection()

st.title("Skill Search CV Ranking System")

tab1, tab2 = st.tabs(["CV Rankings", "Upload CVs"])

with tab1:
    st.header("Get CV Rankings")
    user_input = st.text_area("Enter your input here:", height=200)

    number = st.number_input(
        "Number of cvs (1 to 10):", min_value=1, max_value=10, step=1, value=5
    )

    output_text = f"You entered {len(user_input)} characters and picked number {number}."
    if st.button("Get CV Rankings"):
        cv_rankings = get_cv_rankings(cv_collection, user_input, n_results=number)
        st.json(cv_rankings)
with tab2:
    st.header("Upload your CVs to the vector database")
    uploaded_files = st.file_uploader("Drop your files here or click to upload", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            processed_text = process_uploaded_file(bytes_data)
            cv_collection.add(
                documents=[processed_text],
                metadatas=[{"filename": uploaded_file.name}],
                ids=[uploaded_file.name],
            )
    cvs = cv_collection.get()
    df_cvs = pd.DataFrame({"filename": cvs["ids"]})
    st.data_editor(df_cvs)
    rows_to_delete = st.multiselect(
        "Select cvs to delete by filename", options=df_cvs["filename"]
    )
    st.button(
        "Delete selected cvs",
        on_click=lambda: cv_collection.delete(ids=rows_to_delete)
    )
