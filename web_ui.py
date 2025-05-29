import streamlit as st
import pandas as pd

from skill_search.rag import get_cv_rankings, process_uploaded_file, get_cv_collection


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
