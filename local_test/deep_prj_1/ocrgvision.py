from google.cloud import vision

def ocr_google_vision(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    texts = response.text_annotations
    if texts:
        return texts[0].description  # 첫 번째가 전체 텍스트
    return None

print(ocr_google_vision("sample.jpg"))
