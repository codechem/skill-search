from pymupdf import Document
from chromadb import HttpClient
import os
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm

def extract_text_from_cv(file_path) -> str:
    pdf_doc = Document(file_path)
    return "".join(page.get_text() for page in pdf_doc)


def extract_texts_from_cvs(directory_path: str) -> dict:
    cv_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_cv(file_path)
            cv_texts[filename] = text.replace("⎽", "").replace("\n", "").strip()
    return cv_texts

def add_cvs_to_chromadb(cv_texts: dict, collection) -> None:
    for filename, text in tqdm(cv_texts.items(), desc="Adding CVs to ChromaDB"):
        collection.add(
            documents=[text], metadatas=[{"filename": filename}], ids=[filename]
        )


if __name__ == "__main__":
    chromadb_client = HttpClient(host="0.0.0.0", port=8000)
    job_listings = extract_texts_from_cvs("tests/cvs")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )
    cv_collection = chromadb_client.get_or_create_collection(
        name="cvs",
        embedding_function=openai_ef,
    )
    add_cvs_to_chromadb(job_listings, cv_collection)
    print("CVs added to ChromaDB successfully.")
