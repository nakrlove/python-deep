#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import easyocr
import re
import csv
import argparse
from pathlib import Path
import numpy as np

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser(description="M1 Mac용 영수증 OCR")
parser.add_argument("--input", required=True, help="이미지 파일 또는 폴더 경로")
parser.add_argument("--output", required=True, help="결과 CSV 파일 경로")
args = parser.parse_args()

input_path = Path(args.input)
output_file = Path(args.output)

# ---------------------------
# EasyOCR Reader
# ---------------------------
# M1 CPU/MPS 환경에서 GPU=False 안정적
# EasyOCR Reader 초기화 시 `detect_orientation=True`를 추가하여 기울어진 텍스트 보정 시도
# reader = easyocr.Reader(['ko', 'en'], gpu=False, detect_orientation=True) 
reader = easyocr.Reader(['ko', 'en'], gpu=False) 
# ---------------------------
# 이미지 전처리 함수
# ---------------------------
def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # 1. 원본 이미지 회색조 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 이미지 확대 (OCR 정확도 향상)
    # 텍스트 크기가 작을 경우 2배 또는 3배 확대
    # 이 부분은 영수증 이미지 해상도에 따라 조절 필요
    scaled_gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. 대비 향상 (CLAHE)
    # 어두운 영역과 밝은 영역의 대비를 지역적으로 조절하여 글자를 더 선명하게 만듦
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(scaled_gray)

    # 4. 이진화 (Threshold)
    # Otsu's thresholding을 사용하여 배경과 텍스트를 분리
    _, binary_image = cv2.threshold(enhanced_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 노이즈 제거 (Median Blur)
    # 작은 점 노이즈 제거
    denoised = cv2.medianBlur(binary_image, 3)

    # 6. Unsharp Masking (선명화) - 텍스트 테두리를 더 날카롭게
    # Unsharp Masking은 이미지의 선명도를 높이는 데 사용됩니다.
    # blur = cv2.GaussianBlur(denoised, (0,0), 3) # 이진화된 이미지에 적용하기는 부적합할 수 있음.
    # sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    # 이진화된 이미지에 직접 선명화를 적용하기는 어려우므로, 
    # 원본 이미지에 적용 후 이진화하는 방식이 더 효과적일 수 있습니다.
    # 여기서는 이진화 이후에 적용할 수 있는 방법을 다시 고려하거나, 생략합니다.
    # 현재 denoised 이미지가 OCR에 가장 적합한 형태라고 판단.

    return denoised

# ---------------------------
# OCR 결과에서 영수증 정보 추출 함수
# ---------------------------
def extract_receipt_info(ocr_results):
    total_amount = None
    items = []
    
    # OCR 결과를 텍스트와 바운딩 박스로 분리
    text_blocks = [(bbox, text) for bbox, text, _ in ocr_results]
    full_text = '\n'.join([t for _, t in text_blocks])

    # 1. 합계 금액 추출 (가장 중요)
    # '합계', '총액', '결제금액', 'TOTAL' 등의 키워드를 사용하여 최종 금액 탐색
    # 더 많은 패턴과 키워드 추가
    total_patterns = [
        r'(?:합계|총액|결제금액|TOTAL|Total)\s*[:]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:원|KRW)?',
        r'(\d{1,3}(?:,\d{3})*)\s*(?:원|KRW)?\s*(?:합계|총액|결제금액|TOTAL|Total)',
        r'(\d{1,3}(?:,\d{3})*)\s*원', # 단순 'X원' 패턴
        r'(\d{1,3}(?:,\d{3})*)$' # 줄의 마지막에 있는 숫자
    ]

    # 뒤에서부터 탐색하여 가장 마지막에 나오는 합계 금액을 찾습니다.
    lines = full_text.split('\n')
    for line in reversed(lines):
        line = line.replace(' ', '').strip() # 공백 제거 후 처리
        for pattern in total_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    total_amount = int(match.group(1).replace(',', ''))
                    # 찾았으면 더 이상 탐색하지 않고 반환
                    return total_amount, full_text, items 
                except ValueError:
                    continue # 숫자로 변환 실패 시 다음 패턴 시도

    # 2. 상품명, 단가, 수량, 금액 추출 (선택 사항 - 필요하다면 구현)
    # EasyOCR의 `detail=1` 결과 (바운딩 박스)를 사용하여 아이템 라인 파싱
    # 이 부분은 영수증의 레이아웃이 매우 다양하므로 정교한 로직이 필요합니다.
    # 예시: 항목명, 수량, 단가, 총액이 한 줄에 있는 패턴
    # r'(.+?)\s+(\d+)\s+([\d,]+)\s+([\d,]+)'
    
    # 이 예시에서는 총액 추출에 집중하고, 항목별 추출은 추후 필요시 고도화
    
    return total_amount, full_text, items

# ---------------------------
# 이미지 파일 찾기
# ---------------------------
def get_image_files(path):
    exts = ['.png', '.jpg', '.jpeg', '.heic', '.bmp', '.tiff']
    files = []
    if path.is_dir():
        for ext in exts:
            files.extend(path.glob(f'*{ext}'))
    elif path.is_file():
        if path.suffix.lower() in exts:
            files.append(path)
    return files

# ---------------------------
# OCR 처리 및 결과 저장
# ---------------------------
results = []
image_files = get_image_files(input_path)

if not image_files:
    print("이미지 파일을 찾지 못했습니다.")
    sys.exit(1)

for img_file in image_files:
    print(f"Processing: {img_file}")
    processed_img = preprocess_image(img_file)
    
    if processed_img is None:
        print(f"Failed to read or preprocess image: {img_file}")
        results.append([str(img_file), None, 'Failed to read image', ''])
        continue

    # EasyOCR
    try:
        # detail=1로 설정하여 바운딩 박스 정보를 함께 받음
        ocr_result_detailed = reader.readtext(processed_img, detail=1)
        
        # 추출 함수 호출
        total, full_text_ocr, items = extract_receipt_info(ocr_result_detailed)
        
        # OCR 텍스트 스니펫을 더 길게 저장하여 디버깅 용이
        ocr_text_snippet = full_text_ocr[:500].replace('\n',' ') if full_text_ocr else ''
        
        results.append([str(img_file), total, None, ocr_text_snippet])
        print(f" -> total: {total}  error: None")
    except Exception as e:
        results.append([str(img_file), None, str(e), ''])
        print(f" -> total: None  error: {e}")

# ---------------------------
# CSV 저장
# ---------------------------
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'total', 'error', 'ocr_text_snippet'])
    writer.writerows(results)

print(f"Saved results to {output_file}")