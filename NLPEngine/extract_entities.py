# --- Enhanced Entity Extraction (ML + Rules Hybrid) ---

import re
import spacy
from typing import Dict

# Load spaCy's English NER model (smaller model for speed, can upgrade to 'en_core_web_trf' for best results)
nlp = spacy.load("en_core_web_sm")

# Regular expressions for emails and phone numbers
EMAIL_REGEX = r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,6}"
PHONE_REGEX = r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{3,4}(?:[\s-]?\d{3,4})?"


def extract_entities_from_text(text: str) -> Dict[str, str]:
    """
    Extracts Name, Email, and Phone from a text block.
    Hybrid Approach: ML (SpaCy NER) + Rule-Based (Regex)
    """
    extracted_data = {"name": "", "email": "", "phone": ""}

    # 1. Extract Email
    email_match = re.search(EMAIL_REGEX, text)
    if email_match:
        extracted_data["email"] = email_match.group()

    # 2. Extract Phone
    phone_match = re.search(PHONE_REGEX, text)
    if phone_match:
        phone_number = re.sub(r"[^\d]", "", phone_match.group())  # Keep only digits
        if 8 <= len(phone_number) <= 15:
            extracted_data["phone"] = phone_number

    # 3. Extract Name
    # Process text with SpaCy
    doc = nlp(text)
    candidate_names = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    if candidate_names:
        # Pick the first reasonably looking name
        selected_name = candidate_names[0]
        # Avoid cases where email or random words are picked as name
        if len(selected_name.split()) <= 5 and not any(char.isdigit() for char in selected_name):
            extracted_data["name"] = selected_name

    return extracted_data

