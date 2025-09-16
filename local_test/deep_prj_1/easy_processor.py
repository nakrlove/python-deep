import easyocr
# import openai
import json

# 🔑 OpenAI API 키 설정
openai.api_key = "325495978822"

# 1. EasyOCR로 텍스트 추출
def ocr_extract(image_path):
    reader = easyocr.Reader(['ko', 'en'])
    results = reader.readtext(image_path, detail=0)  # detail=0 → 텍스트만 추출
    return "\n".join(results)

# 2. GPT로 보정 & JSON 구조화
def gpt_postprocess(raw_text):
    prompt = f"""
    아래는 영수증 OCR 결과입니다. 글자가 깨져있고 숫자 오류가 있을 수 있습니다.
    이를 기반으로 영수증을 JSON 형태로 정리해주세요.
    
    규칙:
    - 가게 이름, 주소, 일시, 품목(이름, 수량, 단가, 금액), 합계, 결제정보, 포인트 적립 여부를 포함
    - 숫자는 영수증 맥락에 맞게 교정
    - JSON만 출력

    OCR 원본:
    {raw_text}
    """
    print(prompt)
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o-mini",  # 가볍고 빠른 모델 추천
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.2  # 안정적 출력
    # )

    # return response.choices[0].message["content"]

# 3. 실행
if __name__ == "__main__":
    # image_path = "IMG_2709.jpg"  # OCR할 영수증 이미지 경로
    image_path = "/Users/nakrlove/Desktop/dev/python-deep/local_test/deep-prj-1/IMG_2709.jpg"
    print("📌 Step1: OCR 실행 중...")
    raw_text = ocr_extract(image_path)
    print(raw_text[:300])  # 일부 미리보기

    print("\n📌 Step2: GPT 후처리 중...")
    structured_data = gpt_postprocess(raw_text)

    # JSON 저장
    with open("receipt.json", "w", encoding="utf-8") as f:
        f.write(structured_data)

    print("\n✅ receipt.json 파일 생성 완료")
