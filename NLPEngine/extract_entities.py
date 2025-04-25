import re
from flair.data import Sentence
from flair.models import SequenceTagger

def extract_email_and_phone(text):
    # Fix broken @ and . that OCR commonly splits
    text = text.replace(" @ ", "@").replace(" . ", ".").replace(" dot ", ".")
    text = text.replace(" at ", "@").replace("(at)", "@").replace(" ", "")

    # Email regex (simple + tolerant)
    email_matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = email_matches[0] if email_matches else ""

    # Phone regex (with OCR context tolerance)
    phone_matches = re.findall(r"[\+(\d]{0,4}[\d\s\-().]{9,}", text)
    phone = ""

    # Filter only digit-rich results with length ~10-15
    for match in phone_matches:
        digits = re.sub(r"\D", "", match)
        if 10 <= len(digits) <= 15:
            phone = digits
            break

    return email, phone

# Load the NER model (once globally)
flair_ner_tagger = SequenceTagger.load("ner")

def extract_name_using_flair(text):
    # Convert text into Flair sentence
    sentence = Sentence(text)

    try:
        # Predict named entities
        flair_ner_tagger.predict(sentence)

        # Loop through entities and return first PERSON entity
        for entity in sentence.get_spans('ner'):
            if entity.get_label("ner").value == "PER":
                return entity.text.strip()
    except Exception as e:
        print("❌ Flair NER error:", e)

    return ""

def extract_entities(text):
    name = extract_name_using_flair(text)  # You’ll use Flair here
    email, phone = extract_email_and_phone(text)
    
    return {
        "name": name,
        "email": email,
        "phone": phone
    }
