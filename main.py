# main.py

import os
import logging
import json

from DataPreprocessing.clean_text import clean_extracted_text, preprocess_text
from OCREngine.tesseract_wrapper import perform_ocr
from NLPEngine.extract_entities import extract_entities_from_text
from NLPEngine.global_tfidf import fit_global_tfidf, build_tfidf_with_control
from NLPEngine.skill_matcher import prefilter_resumes
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
resume_folder = "dataset/resumes"
output_path = "results/ranked_resumes.json"
logging.basicConfig(
    filename="results/logs.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
jd_text = """
Job Title: Database / Data-Platform Engineer / Data Scientist

Overview  
Join our data-platform group to design, administer and optimise large-scale Oracle, SQL-Server and PostgreSQL systems in both on-prem and cloud (AWS / Azure) environments. You will work closely with data-science and analytics teams to build ETL pipelines and surface high-quality data to business stakeholders.

Key Responsibilities  
• Install, upgrade and patch Oracle 11g/12c, SQL-Server 2016+ and PostgreSQL clusters  
• Design physical schemas, indexes and partitioning strategies for high-volume OLTP & OLAP workloads  
• Develop PL/SQL, T-SQL and Python utilities for data extraction, cleansing and migration  
• Implement backup, fail-over and disaster-recovery procedures; automate with Bash/PowerShell & Cron/Azure DevOps  
• Monitor performance (AWR, OEM, SolarWinds, Query Store); tune SQL and server parameters  
• Build ELT/ETL pipelines with SQL, Python (Pandas) and Apache Spark/Hadoop; publish data sets to BI tools  
• Collaborate with data-scientists to productionise predictive models and dashboards  
• Write clear run-books and mentor junior DBAs

Required Skills  
• 3 + yrs administering Oracle **or** SQL-Server **and** at least one NoSQL store (MongoDB, Redis)  
• Strong SQL tuning & index-design chops; ability to read **EXPLAIN** plans  
• Scripting in Python or Bash; experience with **Git** and CI/CD  
• Familiarity with AWS RDS / Azure SQL or containerised DB images (Docker)  
• Excellent communication—can translate performance metrics into business impact

Nice-to-Have  
• Tableau/PowerBI, Airflow, Kafka  
• Experience migrating on-prem DBs to cloud  
• Certification: Oracle OCP, Microsoft DP-300, AWS DVA

Education  
BS in Computer Science, Information Systems or related discipline (or equivalent experience)
"""

# Must-have keywords (for pre-filtering)
required_keywords = ["SQL", "Python", "engineer", "database"]
# JD important keywords (for TF-IDF matching)
jd_keywords = [
    "oracle", "sql-server", "postgresql", "aws", "azure", "etl", "pl/sql", "t-sql",
    "python", "bash", "git", "ci/cd", "spark", "hadoop", "airflow", "docker", "mongodb", "redis"
]


def main():

    # Load and prepare all resumes first for global TF-IDF
    all_resume_texts = []
    raw_resume_texts = []
    filenames = []
    for filename in os.listdir(resume_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(resume_folder, filename)
            raw_text, ocr_confidence = perform_ocr(file_path)
            cleaned_text = clean_extracted_text(raw_text)
            lemmatized_text = preprocess_text(cleaned_text)
            all_resume_texts.append(lemmatized_text)
            raw_resume_texts.append(raw_text)
            filenames.append(filename)
            print(f"{filename} Appended successfully!")

     # Pre-filter resumes based on required keywords
    filtered_resumes, filtered_indexes = prefilter_resumes(all_resume_texts, required_keywords)

    tfidf_matrix, vectorizer = build_tfidf_with_control(preprocess_text(jd_text), filtered_resumes, jd_keywords)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    jd_vector_array = jd_vector.toarray()
    similarities = cosine_similarity(resume_vectors, jd_vector_array)

    results = []
    # Now process each file individually
    for idx, similarity in enumerate(similarities):
        original_index = filtered_indexes[idx]
        filename = filenames[original_index]
        logging.info(f"Processing: {filename}")

        raw_text = raw_resume_texts[original_index]

        logging.info(f"Extracted Text:\n{raw_text}")
        logging.info(f"Average OCR Confidence: {ocr_confidence:.2f}%")

        entities = extract_entities_from_text(raw_text)
        matched_keywords = [word for word in all_resume_texts[original_index].split() if word in jd_keywords]
        
        logging.info("Extracted Entities:", entities)

        result = {
            "filename": filename,
            "name": entities.get("name"),
            "email": entities.get("email"),
            "phone": entities.get("phone"),
            "score": round(float(similarity[0]), 4),
            "ocr_avg_confidence": round(float(ocr_confidence), 2),
            "matched_keywords": matched_keywords[:10],
            "tfidf_vocab_size": len(vectorizer.vocabulary_),
        }
        results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        logging.info("Result processed successfully.")

    print("✅ Screening complete! Results saved in results/ranked_resumes.json")

if __name__ == "__main__":
    main()
