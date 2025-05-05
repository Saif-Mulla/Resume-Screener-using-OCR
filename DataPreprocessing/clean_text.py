# clean_text.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_extracted_text(text):
    text = re.sub(r'\n+', ' ', text)    # Remove newlines
    text = re.sub(r'\s+', ' ', text)     # Normalize spaces
    text = re.sub(r'[^\w\s@.-]', '', text) # Remove special chars except email related
    return text.strip()

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

