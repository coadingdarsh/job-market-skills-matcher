"""Job Recommender System (Skills → Job match)

What it does
------------
Given a dataset of job postings with a `job_skills` column and a file of job-seeker
queries (also with `job_skills`), this script:

1) Cleans + tokenizes skills text (lowercase, remove non-letters, remove stopwords)
2) Builds a TF–IDF vector space over job skills
3) Computes cosine similarity between each user's skills and filtered job postings
4) Outputs the Top-N matches per user query to a CSV

Why this is useful
------------------
For newcomers to a job market, the hardest part is often *translating their existing
experience into the local "skills language" used in postings*. This project helps
reduce that gap by showing which roles best match a person's skills, and what
skills the role expects.

Run:
    python job_recommender_system.py --jobs data/linkedin_job_posts_skills.csv \
      --queries data/test_cases.csv --out outputs/output_job_skills_match.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_stopwords_downloaded() -> None:
    """Ensure NLTK stopwords exist."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def load_custom_stopwords(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        print(f"Warning: Custom stopwords file not found at: {path}. Using NLTK defaults.")
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cleaner(stop_words_set: set[str]):
    """Return a function that cleans and tokenizes a skills string."""

    def clean_tokenize_skills(skills: object) -> str:
        # Handle missing values (NaN / None)
        if skills is None or (isinstance(skills, float) and pd.isna(skills)):
            return ""
        skills_str = str(skills).lower().strip()
        if not skills_str:
            return ""

        # keep letters and spaces only
        skills_str = re.sub(r"[^a-z\s]", " ", skills_str)
        tokens = skills_str.split()

        cleaned_tokens = [t for t in tokens if t not in stop_words_set and len(t) > 1]
        return " ".join(cleaned_tokens)

    return clean_tokenize_skills


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skills-based job recommender (TF-IDF + cosine similarity)")
    p.add_argument("--jobs", required=True, help="Path to jobs CSV (must include job_skills + metadata columns)")
    p.add_argument("--queries", required=True, help="Path to job seeker queries CSV (must include sn + job_skills)")
    p.add_argument("--out", default="outputs/output_job_skills_match.csv", help="Output CSV path")
    p.add_argument("--stopwords", default="stopwords.txt", help="Optional custom stopwords file")
    p.add_argument("--topk", type=int, default=5, help="Top-K jobs to return per query")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    jobs_path = Path(args.jobs)
    queries_path = Path(args.queries)
    out_path = Path(args.out)
    stopwords_path = Path(args.stopwords) if args.stopwords else None

    if not jobs_path.exists():
        print(f"ERROR: Could not find jobs file at {jobs_path}")
        return 1
    if not queries_path.exists():
        print(f"ERROR: Could not find queries file at {queries_path}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_stopwords_downloaded()

    stopwords_nltk = set(stopwords.words("english"))
    stopwords_custom = set(load_custom_stopwords(stopwords_path))
    stop_words_set = stopwords_nltk | stopwords_custom

    clean_tokenize_skills = build_cleaner(stop_words_set)

    print("Loading job data…")
    df_jobs = pd.read_csv(jobs_path)
    if "job_skills" not in df_jobs.columns:
        print("ERROR: jobs CSV must include a 'job_skills' column")
        return 1

    df_jobs = df_jobs.where(pd.notnull(df_jobs), None)
    df_jobs["processed_text"] = df_jobs["job_skills"].apply(clean_tokenize_skills)

    print("Building TF–IDF matrix…")
    tfidf_vectorizer = TfidfVectorizer(min_df=2)
    tfidf_matrix_jobs = tfidf_vectorizer.fit_transform(df_jobs["processed_text"])
    print(f"TF–IDF matrix shape: {tfidf_matrix_jobs.shape}")

    print("Loading user queries…")
    df_queries = pd.read_csv(queries_path)
    df_queries = df_queries.where(pd.notnull(df_queries), None)
    if "job_skills" not in df_queries.columns:
        print("ERROR: queries CSV must include a 'job_skills' column")
        return 1
    if "sn" not in df_queries.columns:
        print("ERROR: queries CSV must include an 'sn' column")
        return 1

    df_queries["processed_text"] = df_queries["job_skills"].apply(clean_tokenize_skills)
    tfidf_matrix_queries = tfidf_vectorizer.transform(df_queries["processed_text"])

    print("Calculating matches…")
    results: list[dict] = []

    # Filter columns are optional; if present, we use them.
    filter_cols = [
        ("search_city", "search_city"),
        ("search_country", "search_country"),
        ("job_level", "job_level"),
        ("job_type", "job_type"),
    ]

    for idx, user_row in df_queries.iterrows():
        user_sn = user_row.get("sn")
        print(f"  • Query: {user_sn}")

        mask = pd.Series([True] * len(df_jobs))
        for user_col, job_col in filter_cols:
            if user_col in df_queries.columns and job_col in df_jobs.columns:
                val = user_row.get(user_col)
                if val is not None and str(val).strip() != "":
                    mask &= (df_jobs[job_col] == val)

        candidate_jobs = df_jobs[mask]
        if candidate_jobs.empty:
            continue

        candidate_indices = candidate_jobs.index
        candidate_vectors = tfidf_matrix_jobs[candidate_indices]
        user_vector = tfidf_matrix_queries[idx]

        similarity_scores = cosine_similarity(user_vector, candidate_vectors).flatten()
        ranked_candidates = candidate_jobs.copy()
        ranked_candidates["similarity_score"] = similarity_scores

        top_k = ranked_candidates.sort_values(by="similarity_score", ascending=False).head(args.topk)

        for rank, (_, job_row) in enumerate(top_k.iterrows(), start=1):
            results.append(
                {
                    "query_sn": user_sn,
                    "users_skills": user_row.get("job_skills", ""),
                    "job_suitability_rank": rank,
                    "job_link_id": job_row.get("job_link", ""),
                    "job_title": job_row.get("job_title", ""),
                    "company": job_row.get("company", ""),
                    "job_location": job_row.get("job_location", ""),
                    "job_level": job_row.get("job_level", ""),
                    "job_type": job_row.get("job_type", ""),
                    "job_skills": job_row.get("job_skills", ""),
                    "similarity_score": round(float(job_row.get("similarity_score", 0.0)), 4),
                }
            )

    if not results:
        print("No matches found.")
        return 0

    print(f"Writing results → {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
