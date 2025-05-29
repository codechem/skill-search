from chromadb import HttpClient
from openai import OpenAI
import chromadb.utils.embedding_functions as embedding_functions
import os
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


def get_job_requirements(file_path: str) -> dict:
    with open(file_path, "r") as file:
        job_requirements = file.read()
    return job_requirements

if __name__ == "__main__":
    chromadb_client = HttpClient(host="0.0.0.0", port=8000)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )
    cv_collection = chromadb_client.get_collection("cvs", embedding_function=openai_ef)
    job_req = get_job_requirements("job_requirements/python_job.txt")
    result = cv_collection.query(
        query_texts=[job_req],
        n_results=5,
    )
    filename_by_document = {filename: document for filename, document in zip(result['ids'][0], result['documents'][0])}
    initial_prompt = f"""
    Rank the following CVs based on their relevance to the job requirements.
    Job requirements are:
    {job_req}
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
    print(response.output_text)
