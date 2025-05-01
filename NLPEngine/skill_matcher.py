# skill_matcher.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import numpy as np

def expand_synonyms(skill_list):
    expanded = set(skill_list)
    for word in skill_list:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name())
    return list(expanded)

def match_resume_with_jd(resume_text, jd_text):
    skills = jd_text.split()
    expanded_skills = expand_synonyms(skills)
    
    documents = [resume_text, " ".join(expanded_skills)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]
