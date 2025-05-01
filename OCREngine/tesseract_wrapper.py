# ocr_module.py

import pytesseract
import cv2
import numpy as np
from spellchecker import SpellChecker  # This works fine with pyspellchecker


# Configure path to tesseract executable if needed (Windows example)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

spell = SpellChecker()

def perform_ocr(image_path, retries=2, confidence_threshold=40):
    """
    Perform OCR on the image, retrying if confidence is low.
    """
    img = cv2.imread(image_path)

    # Preprocessing for better OCR
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR options
    custom_config = r'--oem 3 --psm 6'  # LSTM OCR Engine, Assume a block of text
    data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config=custom_config)

    text = ' '.join(data['text'])
    avg_confidence = np.mean([conf for conf in data['conf'] if conf != -1])

    attempt = 0
    while avg_confidence < confidence_threshold and attempt < retries:
        print(f"Low confidence ({avg_confidence}). Retrying OCR with adaptive threshold...")
        # Adaptive threshold to retry
        img_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        data = pytesseract.image_to_data(img_adapt, output_type=pytesseract.Output.DICT, config=custom_config)
        text = ' '.join(data['text'])
        avg_confidence = np.mean([conf for conf in data['conf'] if conf != -1])
        attempt += 1

    cleaned_text = basic_spell_check(text)
    return cleaned_text, avg_confidence

def basic_spell_check(text):
    """
    Selectively apply spell checking to fix OCR mistakes (only for alphabetic words).
    """
    corrected_words = []
    words = text.split()
    for word in words:
        if word.isalpha() and len(word) > 3:  # Avoid touching numbers, short words
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word if corrected_word else word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

