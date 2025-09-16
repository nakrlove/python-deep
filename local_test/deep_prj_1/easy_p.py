import os
from PIL import Image
import numpy as np
import cv2
import easyocr
from transformers import pipeline
import torch

# -------------------------
# 1️⃣ 이미지 전처리 및 OCR
# -------------------------
def preprocess_image(image_path):
    # PIL로 이미지 열기 (HEIC 등도 가능)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # OpenCV 전처리: 그레이스케일 + 노이즈 제거 + 이진화
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return img_thresh

def ocr_extract(image_np):
    reader = easyocr.Reader(['ko','en'])
    results = reader.readtext(image_np, detail=0)
    return "\n".join(results)

# -------------------------
# 2️⃣ LLM 후처리
# -------------------------
def llm_postprocess(raw_text, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",  # GPU 자동 할당 (없으면 CPU)
        torch_dtype="auto"
    )

    prompt = f"""
다음은 영수증 OCR 결과입니다. 글자가 깨져있고 숫자 오류가 있을 수 있습니다.
이를 기반으로 영수증을 JSON 형태로 정리해주세요.

규칙:
- JSON만 출력
- 포함 항목: 가게 이름, 주소, 일시, 품목(이름, 수량, 단가, 금액), 합계, 결제정보, 포인트 적립 여부
- 숫자는 영수증 맥락에 맞게 교정

OCR 원본:
{raw_text}
"""

    outputs = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        eos_token_id=generator.tokenizer.eos_token_id
    )
    return outputs[0]["generated_text"]

# -------------------------
# 3️⃣ 실행부
# -------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "IMG_2709.jpg")  # HEIC 포함
    print("이미지 경로:", image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")

    print("📌 Step1: 이미지 전처리 및 OCR 실행 중...")
    img_preprocessed = preprocess_image(image_path)
    raw_text = ocr_extract(img_preprocessed)
    print("OCR 결과 미리보기:\n", raw_text[:200], "...\n")

    print("📌 Step2: Hugging Face LLM 후처리 중...")
    structured_data = llm_postprocess(raw_text, model_name="mistralai/Mistral-7B-Instruct-v0.2")
    
    print("\n✅ 결과(JSON):")
    print(structured_data)
