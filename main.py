# main.py

import os
import logging
import json

from DataPreprocessing.clean_text import clean_extracted_text, preprocess_text
from OCREngine.tesseract_wrapper import perform_ocr, basic_spell_check
from NLPEngine.extract_entities import extract_entities_from_text
from NLPEngine.global_tfidf import fit_global_tfidf, transform_with_global_tfidf

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

def main():

    # Load and prepare all resumes first for global TF-IDF
    all_resume_texts = []
    filenames = []
    for filename in os.listdir(resume_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(resume_folder, filename)
            raw_text, ocr_confidence = perform_ocr(file_path)
            cleaned_text = clean_extracted_text(raw_text)
            lemmatized_text = preprocess_text(cleaned_text)
            all_resume_texts.append(lemmatized_text)
            filenames.append(filename)
            print(f"{filename} Appended successfully!")

    # Fit global TF-IDF
    fit_global_tfidf(all_resume_texts + [preprocess_text(jd_text)])

    # Now process each file individually
    for idx, filename in enumerate(filenames):
        file_path = os.path.join(resume_folder, filename)
        logging.info(f"Processing: {filename}")
        
        try:
            raw_text, ocr_confidence = perform_ocr(file_path)
            clean_text = clean_extracted_text(raw_text)

            logging.info(f"Extracted Text:\n{raw_text}")
            logging.info(f"Average OCR Confidence: {ocr_confidence:.2f}%")
            
            if ocr_confidence < 0.40:
                clean_text = basic_spell_check(clean_text)

            lemmatized_text = preprocess_text(clean_text)
            entities = extract_entities_from_text(raw_text)
            logging.info("Extracted Entities:", entities)
            resume_vec = transform_with_global_tfidf(lemmatized_text)
            jd_vec = transform_with_global_tfidf(preprocess_text(jd_text))

            from sklearn.metrics.pairwise import cosine_similarity
            score = cosine_similarity(resume_vec, jd_vec)[0][0]

            matched_keywords = [word for word in lemmatized_text.split() if word in preprocess_text(jd_text)]

            result = {
                "filename": filename,
                "name": entities["name"],
                "email": entities["email"],
                "phone": entities["phone"],
                "score": round(float(score), 4),
                "ocr_avg_confidence": round(float(ocr_confidence), 2),
                "matched_keywords": matched_keywords[:10],
                "tfidf_vocab_size": len(resume_vec.toarray()[0])
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            logging.info("Result processed successfully.")

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    print("✅ Screening complete! Results saved in results/ranked_resumes.json")

if __name__ == "__main__":
    main()
