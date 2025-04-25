# main.py

import os
import sys
from DataPreprocessing.clean_text import clean_extracted_text, preprocess_text
from OCREngine.easyocr_wrapper import extract_text_from_image
from NLPEngine.extract_entities import extract_entities
from NLPEngine.skill_matcher import match_resume_with_jd
import json

# Folder paths
image_path = "dataset/resumes/Image_2.jpg"
# log_path = "results/logs.txt"
# sys.stdout = open(log_path, "w")  # Send all prints to logs.txt

jd_text = """
Job Summary:

We are seeking a skilled and passionate Software Developer with experience in building and maintaining IT applications, The ideal candidate will be responsible for designing, developing, testing, and maintaining high-performance, secure, and reliable software solutions tailored to different environments, ensuring security compliance.

Key Responsibilities:
Design and develop application (real-time video capture, image processing, and diagnostics tools) using Java, Python and C++.
Collaborate with experts, architecture engineers, and other developers to define software requirements.
Write reusable, testable, and efficient code.
Collaborate with cross-functional teams including front-end developers, DevOps, and product managers.
Integrate user-facing elements with server-side logic.
Implement data storage solutions (e.g., PostgreSQL, MongoDB, Redis).
Develop RESTful APIs or microservices architecture.
Participate in code reviews, architecture discussions, and continuous improvement initiatives.
Write unit and integration tests to ensure code quality.
Troubleshoot and debug software issues in development and production environments.
Maintain documentation for code, architecture, and user manuals.
Stay up to date with latest trending technologies, compliance requirements, and software development best practices.

Required Skills & Qualifications:
Good to have years of experience in Softawre development.
Strong understanding of different frameworks and Programming language.
Good knowledge of SQL and NoSQL databases.
Experience working with APIs and third-party integrations.
Familiarity with version control systems like Git.
Understanding of software development best practices, including Agile/Scrum.
Good problem-solving and communication skills.
Basic understanding of data structures and algorithms.

Nice to Have:
Knowledge of front-end technologies.
Exposure to containerization tools like Docker.
Experience with CI/CD pipelines is a plus.
Experience with cloud platforms like AWS, GCP, or Azure.
Familiarity with asynchronous programming, Celery, or message brokers like RabbitMQ/Kafka.

Education:
Bachelor's degree in Computer Science, Engineering, or related field (or equivalent work experience).

"""
# Folder with multiple image resumes
resume_folder = "dataset/resumes"

def main():

    output_path = "results/ranked_resumes.json"
    if not os.path.exists("results"):
        os.makedirs("results")

    for filename in os.listdir(resume_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(resume_folder, filename)
            print(f"📄 Processing: {filename}")
            try:
                # Extract and clean text
                raw_text, ocr_confidence, ocr_word_count = extract_text_from_image(image_path)

                clean_text = clean_extracted_text(raw_text)  # Just light cleanup (linebreaks, tabs)

                # Save full raw for Flair/regex entity extraction
                entities = extract_entities(raw_text)

                # Use clean text ONLY for vectorization
                lemmatized_text = preprocess_text(clean_text)

                score, matched_keywords, vocab = match_resume_with_jd(lemmatized_text, preprocess_text(jd_text))

                # Append result
                print("🧾 Raw OCR Text:")
                print(raw_text[:500])
                result = {
                    "filename": filename,
                    "name": entities["name"],
                    "email": entities["email"],
                    "phone": entities["phone"],
                    "score": score,
                    "ocr_word_count": ocr_word_count,
                    "ocr_avg_confidence": ocr_confidence,
                    "matched_keywords": matched_keywords[:10],  # log top matches
                    "tfidf_vocab_size": len(vocab)
                }

                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

    print("✅ Screening complete! Results saved in results/ranked_resumes.json")

if __name__ == "__main__":
    main()
