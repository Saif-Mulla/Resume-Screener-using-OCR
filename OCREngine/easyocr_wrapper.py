import easyocr

reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    try:
        results = reader.readtext(image_path, detail=1)  # detail=1 returns boxes + text + confidence
        extracted_text = " ".join([text for _, text, _ in results])
        confidences = [conf for _, _, conf in results]
        average_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        word_count = len(results)

        return extracted_text, average_conf, word_count
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return "", 0.0, 0
