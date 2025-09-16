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
reader = easyocr.Reader(['ko', 'en'], gpu=False)  # M1 CPU/MPS 환경에서 GPU=False 안정적

# ---------------------------
# 이미지 전처리 함수
# ---------------------------
def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 확대
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 이진화 (Threshold)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 노이즈 제거
    processed = cv2.medianBlur(thresh, 3)
    return processed

# ---------------------------
# 합계 금액 추출 함수
# ---------------------------
def extract_total(text):
    # 숫자 추출 (콤마 포함)
    matches = re.findall(r'\d{1,3}(?:,\d{3})*', text.replace(' ', ''))
    if not matches:
        return None

    # 키워드 근처 숫자 우선
    keywords = ['합계', '총액', '금액', '총 금액', '합']
    lines = text.split('\n')
    for line in lines[::-1]:  # 마지막 줄부터 탐색
        if any(k in line for k in keywords):
            line_matches = re.findall(r'\d{1,3}(?:,\d{3})*', line.replace(' ', ''))
            if line_matches:
                return int(line_matches[-1].replace(',', ''))
    # 없으면 마지막 숫자
    return int(matches[-1].replace(',', ''))

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
# OCR 처리
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
        print(f"Failed to read image: {img_file}")
        results.append([str(img_file), None, 'Failed to read image', ''])
        continue

    # EasyOCR
    try:
        ocr_result = reader.readtext(processed_img, detail=0, paragraph=True)
        ocr_text = '\n'.join(ocr_result)
        total = extract_total(ocr_text)
        results.append([str(img_file), total, None, ocr_text[:100].replace('\n',' ')])
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
