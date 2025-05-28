from chromadb import HttpClient
from openai import OpenAI
import os

def get_job_requirements(file_path: str) -> dict:
    with open(file_path, "r") as file:
        job_requirements = file.read()
    return job_requirements

if __name__ == "__main__":
    chromadb_client = HttpClient(host="0.0.0.0", port=8000)
    cv_collection = chromadb_client.get_collection("cvs")
    job_req = get_job_requirements("job_requirements/react_job.txt")
    result = cv_collection.query(
        query_texts=[job_req],
        n_results=6,
    )
    filename_by_document = {filename: document for filename, document in zip(result['ids'][0], result['documents'][0])}
    initial_prompt = f"""
    Rank the following CVs based on their relevance to the job requirements.
    Job requirements are:
    {job_req}
    The CVs are:{filename_by_document}
    Provide the ranking by using the filename of the CVs.
    Provide a short explanation for the ranking.
    Don't provide any other information.
    Don't provide any other text.
    """
    openai_client = OpenAI()
    response = openai_client.responses.create(model="gpt-4.1", input=initial_prompt)
    print(response.output_text)
