# global_tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer

global_vectorizer = None

def fit_global_tfidf(texts):
    global global_vectorizer
    global_vectorizer = TfidfVectorizer()
    global_vectorizer.fit(texts)
    return global_vectorizer.get_feature_names_out()

def transform_with_global_tfidf(text):
    if global_vectorizer is None:
        raise Exception("Global TF-IDF vectorizer not fitted. Call fit_global_tfidf() first.")
    return global_vectorizer.transform([text])
