from chromadb import HttpClient

def get_job_requirements(file_path: str) -> dict:
    with open(file_path, "r") as file:
        job_requirements = file.read()
    return job_requirements

if __name__ == "__main__":
    # Initialize the ChromaDB client
    chromadb_client = HttpClient(host="0.0.0.0", port=8000)
    cv_collection = chromadb_client.get_collection("cvs")
    job_req = get_job_requirements("job_requirements/dotnet_job.txt")
    result = cv_collection.query(
        query_texts=[job_req],
        n_results=8,
    )
    print(result)
