import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend_top_jobs(resume_text, job_csv_path, top_n=3):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, job_csv_path)

    jobs = pd.read_csv(csv_path)

    corpus = [resume_text] + jobs["Job Description"].tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:]
    )[0]

    jobs["Match Score"] = similarity_scores * 100
    top_jobs = jobs.sort_values(
        by="Match Score", ascending=False
    ).head(top_n)

    return top_jobs


def extract_skills(resume_text):
    skill_set = [
        "python", "java", "c", "c++", "machine learning",
        "deep learning", "nlp", "sql", "html", "css",
        "javascript", "react", "tensorflow", "pytorch"
    ]

    found_skills = [
        skill for skill in skill_set if skill in resume_text
    ]

    return found_skills
