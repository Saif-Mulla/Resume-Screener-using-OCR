import re
import spacy

def clean_extracted_text(text):
    # ✅ Minimal cleanup — preserve word tokens
    text = text.replace('\n', ' ').replace('\r', ' ')

    # ⚠️ Don't remove punctuation critical for email or names
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)             # Normalize whitespace
    return text.strip()


nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # ✅ Clean out special characters and numbers
    text = re.sub(r"\s+", " ", text)               # ✅ Normalize spaces

    doc = nlp(text.lower())                        # ✅ SpaCy parses text into tokens
    lemmatized = " ".join([
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha     # ✅ Removes stopwords & non-words
    ])
    return lemmatized