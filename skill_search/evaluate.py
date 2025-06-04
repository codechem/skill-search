import os

import tqdm
import pandas as pd
from skill_search.rag import get_cv_rankings, get_cv_collection

def get_job_listings(job_listing_dir: str):
    job_listings = {}
    for filename in os.listdir(job_listing_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(job_listing_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                job_listings[filename] = text.replace("⎽", "").replace("\n", "").strip()
    return job_listings


def generate_ranking(job_listing: str, cv_collection, n_results=5):
    rankings = get_cv_rankings(cv_collection, job_listing, n_results)
    rankings_filenames = [
        (ranking["filename"], ranking["confidence_score"])
        for ranking in rankings["cv_rankings"]
    ]
    rankings_filenames.sort(key=lambda x: x[1], reverse=True)
    return rankings_filenames


if __name__ == "__main__":
    job_listing_dir = "tests/job_listings"
    job_listings = get_job_listings(job_listing_dir)

    cv_collection = get_cv_collection()
    cv_results = {"job_listing": [], "rankings": [], "confidence_scores": []}
    for filename, job_listing in tqdm.tqdm(job_listings.items()):
        rankings = generate_ranking(job_listing, cv_collection, 5)
        cv_results["job_listing"].append(filename)
        cv_results["rankings"].append(",".join([str(r[0]) for r in rankings]))
        cv_results["confidence_scores"].append(",".join([str(r[1]) for r in rankings]))

    df = pd.DataFrame(cv_results)
    df.to_csv("cv_rankings.csv", index=False)
