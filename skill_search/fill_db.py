from pymupdf import Document
from chromadb import HttpClient
import os


def extract_text_from_cv(file_path) -> str:
    pdf_doc = Document(file_path)
    return "".join(page.get_text() for page in pdf_doc)


def extract_texts_from_cvs(directory_path: str) -> dict:
    cv_texts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_cv(file_path)
            cv_texts[filename] = text
    return cv_texts

def add_cvs_to_chromadb(cv_texts: dict, collection) -> None:
    for filename, text in cv_texts.items():
        collection.add(
            documents=[text], metadatas=[{"filename": filename}], ids=[filename]
        )


if __name__ == "__main__":
    chromadb_client = HttpClient(host="0.0.0.0", port=8000)
    cv_collection = chromadb_client.create_collection(
        "cvs", configuration={"space": "cosine"}
    )
    cv_texts = extract_texts_from_cvs("cvs")
    add_cvs_to_chromadb(cv_texts, cv_collection)
    print("CVs added to ChromaDB successfully.")
