# receipt_ocr.py
import sys
import re
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pytesseract

# (필요시) Tesseract 경로 지정 (Windows 등)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image_for_ocr(image_path):
    """OpenCV 전처리: 그레이스케일, 블러, 어댑티브 쓰레숄드"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # resize if very large (선택)
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000.0 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold (글자가 선명해지도록)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # optionally invert if background is dark
    white_ratio = np.mean(th == 255)
    if white_ratio < 0.5:
        th = cv2.bitwise_not(th)
    return th

def ocr_image(img_array):
    """pytesseract로 OCR 수행 (한국어 + 영어)"""
    # lang='kor+eng' : 한국어와 영어 함께 읽기
    custom_config = r'--oem 3 --psm 6'  # OEM/PSM 설정은 상황에 따라 조정
    text = pytesseract.image_to_string(img_array, lang='kor+eng', config=custom_config)
    return text

def parse_receipt_text(text):
    """영수증에서 핵심정보(상점, 날짜, 품목, 합계 등) 간단 추출"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    result = {}

    # 상점명(맨 위에 나온 큰 글씨 추정) — 첫 줄을 상점명으로
    result['store'] = lines[0] if lines else ''

    # 날짜/시간 찾기 (예: 2025-09-09 22:34)
    date_match = re.search(r'(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*\d{1,2}[:.]\d{2})', joined)
    if date_match:
        result['datetime'] = date_match.group(1)
    else:
        # 다른 포맷 시도
        dm2 = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', joined)
        result['datetime'] = dm2.group(1) if dm2 else ''

    # 합계 금액 찾기 (합계:, 합 계, Total 등)
    total_match = re.search(r'합\s*계[:\s]*([0-9,]+)\s*원?', joined)
    if not total_match:
        total_match = re.search(r'Total[:\s]*([0-9,]+)', joined, re.IGNORECASE)
    result['total'] = total_match.group(1) if total_match else ''

    # 결제수단(카드 등)
    pay_match = re.search(r'(BC|우리카드|카드)[^\n]*([0-9,]+원?)', joined)
    if pay_match:
        result['payment'] = pay_match.group(0)
    else:
        # 단순히 "결제금액" 있는지 확인
        pay2 = re.search(r'결제금액[:\s]*([0-9,]+)', joined)
        result['payment'] = pay2.group(0) if pay2 else ''

    # 항목(상품 코드/단가/수량/금액) 단순 파싱 시도:
    items = []
    # 항목으로 보이는 줄: 숫자+공백+이름 ... 금액 형태
    # ex) "002 롯데초코파이480G 4,570 1 4,570"
    item_line_regex = re.compile(r'^\d{3}\s+(.+?)\s+([0-9,]{2,})\s+(\d+)\s+([0-9,]{2,})$')
    for ln in lines:
        m = item_line_regex.match(ln)
        if m:
            name = m.group(1).strip()
            price = m.group(2).replace(',', '')
            qty = m.group(3)
            amount = m.group(4).replace(',', '')
            items.append({'name': name, 'unit_price': int(price), 'qty': int(qty), 'amount': int(amount)})
    # fallback: 간단히 "가격"이 포함된 줄에서 추출
    if not items:
        # find lines containing '원' and a number
        for ln in lines:
            if re.search(r'\d{1,3}(,\d{3})*원?', ln):
                # 간단히 추가 (정교한 파싱은 실제 포맷에 맞춰 튜닝 필요)
                items.append({'raw': ln})
    result['items'] = items
    result['raw_text'] = joined
    return result

def pretty_print(parsed):
    print("\n=== 추출 결과 ===\n")
    print(f"상점: {parsed.get('store')}")
    print(f"일시: {parsed.get('datetime')}")
    print()
    print("구매 항목:")
    if parsed.get('items'):
        for it in parsed['items']:
            if 'name' in it:
                print(f" - {it['name']}  단가: {it['unit_price']}  수량: {it['qty']}  금액: {it['amount']}")
            else:
                print(f" - {it.get('raw')}")
    else:
        print(" - 항목을 찾지 못했습니다.")
    print()
    print(f"합계: {parsed.get('total')}")
    print(f"결제정보: {parsed.get('payment')}")
    print("\n[원문 OCR 텍스트]\n")
    print(parsed.get('raw_text')[:2000])  # 길면 일부만 출력

def main(image_path):
    img_pre = preprocess_image_for_ocr(image_path)
    # PIL 이미지로 변환해 pytesseract에 전달 (opencv -> PIL)
    pil_img = Image.fromarray(img_pre)
    text = ocr_image(pil_img)
    parsed = parse_receipt_text(text)
    pretty_print(parsed)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python receipt_ocr.py /path/to/receipt.jpg")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    main(image_path)
