from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    # Uses Tesseract OCR to extract text from a given image.
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return ""