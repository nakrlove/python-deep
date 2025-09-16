# ===============================
# M1 Mac용 영수증 OCR + 총액 추출 예제
# ===============================

import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import os
# os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
# -------------------------------
# 1. Tesseract 경로 지정 (M1 Mac Homebrew)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# -------------------------------
# 2. 이미지 열기 (HEIC/JPG 자동 처리)
# -------------------------------
def open_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".heic":
        # HEIC → JPG 변환 후 열기
        jpg_path = file_path.replace(".heic", ".jpg")
        os.system(f"sips -s format jpeg {file_path} --out {jpg_path}")
        return Image.open(jpg_path)
    else:
        return Image.open(file_path)

# -------------------------------
# 3. 이미지 전처리 (OCR 정확도 향상)
# -------------------------------
def preprocess_image(pil_image):
    # PIL -> OpenCV
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # 그레이스케일
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    gray = cv2.medianBlur(gray, 3)
    
    # 이진화
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 다시 PIL로 변환
    return Image.fromarray(thresh)

# -------------------------------
# 4. OCR 수행 (한국어+영어)
# -------------------------------
def ocr_image(pil_image):
    return pytesseract.image_to_string(pil_image, lang="kor+eng")

# -------------------------------
# 5. 총액 추출
# -------------------------------
def extract_total(text):
    # "합계" / "총액" / "Total" 키워드 탐색
    match = re.search(r"(합계|총액|Total)\s*[:\-]?\s*([\d,]+)", text)
    if match:
        return int(match.group(2).replace(",", ""))
    return None

# -------------------------------
# 6. 실행 예제
# -------------------------------
if __name__ == "__main__":
    image_path = "./local_test/deep-prj-1/receipt_cafe_001.jpg"  # HEIC 가능
    img = open_image(image_path)
    preprocessed_img = preprocess_image(img)
    text = ocr_image(preprocessed_img)
    
    print("📄 OCR 결과:\n")
    print(text)
    
    total = extract_total(text)
    print(f"\n💰 총액: {total if total else '추출 실패'}")
