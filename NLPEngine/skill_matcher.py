from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_resume_with_jd(resume_text, jd_text):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

    vectors = vectorizer.fit_transform([resume_text, jd_text])

    # Cosine similarity
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Feature terms used
    feature_names = vectorizer.get_feature_names_out()
    
    # Intersect words from JD and resume
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    common_words = resume_words.intersection(jd_words)

    # Vocabulary overlap with JD
    matched_keywords = [term for term in feature_names if term in common_words]

    return round(score, 2), matched_keywords, list(feature_names)
