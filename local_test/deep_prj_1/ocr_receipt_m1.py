"""
ocr_receipt_m1.py
M1 Mac용 영수증 OCR 파이프라인:
 - HEIC 자동 변환 (sips)
 - 전처리: resize, deskew, denoise, adaptive threshold, morphology
 - OCR: Tesseract (kor+eng) 기본, easyocr는 optional fallback
 - 총액(합계) 추출: 다양한 정규식 시도
 - 한 폴더의 이미지 일괄 처리 + 결과 CSV 출력

Usage:
 python ocr_receipt_m1.py --input data/receipts --output results.csv
"""

import os
import sys
import argparse
import subprocess
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import pytesseract
import re
import csv
from io import BytesIO

# ---------------------------
# 환경 설정 (M1 Homebrew 경로)
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
# Tesseract 데이터 경로가 특이하면 설정 (권장)
os.environ.setdefault("TESSDATA_PREFIX", "/opt/homebrew/share/")

# ---------------------------
# Optional: easyocr fallback (설치되어 있으면 사용)
# ---------------------------
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # EasyOCR reader를 lazy-init (언어: ko, en)
    _easyocr_reader = None
    def get_easyocr_reader():
        global _easyocr_reader
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['ko','en'], gpu=False)  # GPU/MPS 설정은 별도
        return _easyocr_reader
except Exception:
    EASYOCR_AVAILABLE = False

# ---------------------------
# 유틸: HEIC -> JPG 변환 (macOS sips 사용)
# ---------------------------
def convert_heic_to_jpg(path):
    base, ext = os.path.splitext(path)
    if ext.lower() == ".heic":
        jpg_path = base + ".jpg"
        cmd = ["sips", "-s", "format", "jpeg", path, "--out", jpg_path]
        subprocess.run(cmd, check=True)
        return jpg_path
    return path

# ---------------------------
# 이미지 로드(HEIC 처리 포함)
# ---------------------------
def load_image(path):
    path = convert_heic_to_jpg(path)
    img = Image.open(path)
    # 일부 이미지는 RGBA일 수 있으니 RGB로 변환
    if img.mode not in ("RGB","L"):
        img = img.convert("RGB")
    return img

# ---------------------------
# 전처리: 고해상도 리사이즈, deskew, denoise, adaptive threshold
# ---------------------------
def preprocess_for_ocr(pil_img, upscale=2, apply_deskew=True):
    # 1) 해상도 업스케일 (작은 글씨 보강)
    w, h = pil_img.size
    pil_img = pil_img.resize((w*upscale, h*upscale), Image.LANCZOS)

    # 2) 회색화
    gray = pil_img.convert("L")

    # 3) 노이즈 제거 (필터)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    # 4) OpenCV로 변환
    arr = np.array(gray)
    # 5) GaussianBlur로 추가 노이즈 제거
    arr = cv2.GaussianBlur(arr, (3,3), 0)

    # 6) Adaptive Threshold (조명 균일화에 강함)
    th = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)

    # 7) Morphological ops로 글자 연결/잡음 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8) deskew (기울기 보정)
    if apply_deskew:
        coords = np.column_stack(np.where(th > 0))
        if coords.size != 0:
            angle = cv2.minAreaRect(coords)[-1]
            # adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            # rotate image to deskew
            (h2, w2) = th.shape[:2]
            center = (w2 // 2, h2 // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            th = cv2.warpAffine(th, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(th)

# ---------------------------
# OCR with pytesseract
# ---------------------------
def ocr_with_tesseract(pil_img, lang="kor+eng", config="--psm 6"):
    # config: psm 6 = Assume a single uniform block of text
    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        return text
    except Exception as e:
        print("Tesseract OCR error:", e, file=sys.stderr)
        return ""

# ---------------------------
# OCR with EasyOCR fallback (optional)
# ---------------------------
def ocr_with_easyocr(pil_img):
    if not EASYOCR_AVAILABLE:
        return ""
    reader = get_easyocr_reader()
    arr = np.array(pil_img.convert("RGB"))
    results = reader.readtext(arr)
    # results: list of (bbox, text, confidence)
    texts = [r[1] for r in results]
    return "\n".join(texts)

# ---------------------------
# 총액(합계) 추출: 다양한 패턴 시도
# ---------------------------
TOTAL_PATTERNS = [
    r"(합계|총액|총 금액|결제금액)\s*[:：]?\s*([\d.,]+)\s*(원|KRW)?",
    r"(TOTAL|Total|AMOUNT DUE|AMOUNT)\s*[:\-]?\s*([\d,\.]+)",
    r"([\d,]{3,}\s*원)",   # '10,300원' 같은 패턴
    r"([\d]{3,}\,[\d]{3})", # '1,000,000' 등
]

def extract_total_from_text(text):
    if not text or len(text.strip()) == 0:
        return None
    # Normalize wide spaces and odd chars
    s = text.replace("\xa0", " ").replace("\u200b"," ").replace(" ", " ")
    # Try each pattern
    for pat in TOTAL_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            # capture group possibly in group 1 or 2
            for i in range(1, m.lastindex+1 if m.lastindex else 2):
                grp = m.group(i)
                if grp:
                    # strip non-digit except , and .
                    num = re.sub(r"[^\d.,]", "", grp)
                    # prefer integer style (comma) over decimal
                    num = num.replace(",", "")
                    if num.isdigit():
                        return int(num)
                    # if num has dot, try parse float then int
                    try:
                        f = float(num)
                        return int(round(f))
                    except:
                        continue
    return None

# ---------------------------
# 전체 처리: 한 파일 처리
# ---------------------------
def process_file(path, debug=False):
    try:
        img = load_image(path)
    except Exception as e:
        return {"file": path, "error": f"load_error: {e}", "text": "", "total": None}

    pre = preprocess_for_ocr(img, upscale=2, apply_deskew=True)
    # First try tesseract with kor+eng
    text = ocr_with_tesseract(pre, lang="kor+eng", config="--psm 6")
    total = extract_total_from_text(text)

    # If no total found or text seems gibberish, try variations:
    if (total is None) or (len(text.strip()) < 10):
        # try different psm values or Tesseract with digits-only whitelist
        tcfg = "--psm 6 -c tessedit_char_whitelist=0123456789,.,,원,만원,총합계,합계,Total,:" 
        alt_text = ocr_with_tesseract(pre, lang="kor+eng", config=tcfg)
        alt_total = extract_total_from_text(alt_text)
        if alt_total and total is None:
            total = alt_total
            text = text + "\n\n[alt]\n" + alt_text

    # If still nothing and EasyOCR available, try EasyOCR
    if (total is None or len(text.strip()) < 10) and EASYOCR_AVAILABLE:
        try:
            e_text = ocr_with_easyocr(pre)
            e_total = extract_total_from_text(e_text)
            if e_total:
                total = e_total
            # append to text for review
            text = text + "\n\n[EasyOCR]\n" + e_text
        except Exception as e:
            # EasyOCR may raise on missing torch/mps etc.
            text = text + f"\n\n[EasyOCR error: {e}]"

    # Final fallback: search raw image for numbers via OCR digits-only
    if total is None:
        tcfg_digits = "--psm 6 -c tessedit_char_whitelist=0123456789,"
        digits_text = ocr_with_tesseract(pre, lang="eng", config=tcfg_digits)
        d_total = extract_total_from_text(digits_text)
        if d_total:
            total = d_total
            text = text + "\n\n[digits_ocr]\n" + digits_text

    return {"file": path, "error": None, "text": text, "total": total}

# ---------------------------
# Batch process folder and write CSV
# ---------------------------
def process_folder(input_dir, output_csv, debug=False):
    exts = (".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp")
    rows = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                path = os.path.join(root, fn)
                print("Processing:", path)
                res = process_file(path, debug=debug)
                rows.append(res)
                print(" -> total:", res["total"], " error:", res["error"])
    # write csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file","total","error","ocr_text_snippet"])
        for r in rows:
            snippet = (r["text"] or "").replace("\n", " ")[:300]
            writer.writerow([r["file"], r["total"] or "", r["error"] or "", snippet])
    print("Saved results to", output_csv)

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="M1 OCR receipts (kor+eng) pipeline")
    parser.add_argument("--input", "-i", required=True, help="input folder with images")
    parser.add_argument("--output", "-o", default="ocr_results.csv", help="output CSV")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    process_folder(args.input, args.output, debug=args.debug)

if __name__ == "__main__":
    main()
