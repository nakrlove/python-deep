#!/usr/bin/env python3
"""
receipt_parser.py

Usage:
  python receipt_parser.py --input IMG_2709.jpg --out_csv items.csv --out_summary summary.json --debug debug.png

요약:
- EasyOCR 우선 사용, 없으면 pytesseract로 대체(단, tesseract가 설치되어 있어야 함)
- 이미지 전처리: grayscale -> CLAHE -> denoise -> adaptive thresh -> deskew
- OCR -> bounding boxes -> 행(row) 단위로 병합 -> 각 행에서 숫자 추출 -> 상품/단가/수량/금액 추정
- 합계/면세/부가세 탐색
- 결과: items CSV, summary JSON, (디버깅용) 박스 표시된 이미지
"""
import argparse
import os
import re
import json
from collections import defaultdict
from typing import List, Dict, Tuple
import cv2
import numpy as np
import pandas as pd

# Try easyocr first
try:
    import easyocr
    OCR_ENGINE = "easyocr"
except Exception:
    OCR_ENGINE = None

# Try pytesseract fallback
try:
    import pytesseract
    OCR_ENGINE = OCR_ENGINE or "pytesseract"
except Exception:
    pass

# -------------------------
# Helpers: image preprocess
# -------------------------
def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def deskew_image(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    # Use binary image pixels to compute angle via minAreaRect of nonzero points
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = 255 - bw

    coords = np.column_stack(np.where(bw_inv > 0))
    if len(coords) < 10:
        return gray, 0.0
    rect = cv2.minAreaRect(coords)  # ((cx,cy),(w,h), angle)
    angle = rect[-1]
    # minAreaRect angle convention: if angle < -45 -> angle = -(90 + angle); else -angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # rotate
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def preprocess(img: np.ndarray, debug_save: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (preproc_for_ocr, debug_vis)
    preproc_for_ocr: grayscale or binary image to feed OCR
    debug_vis: color image for visualization
    """
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE (contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # denoise (fastNlMeans)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    # deskew
    gray, angle = deskew_image(gray)

    # adaptive thresh to get readable text blobs for some OCR engines
    # but we will feed original grayscale to EasyOCR (it expects normal image).
    # we still keep a binary for contour detection
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    debug_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if debug_save:
        cv2.imwrite(debug_save, debug_vis)
    return gray, bw

# -------------------------
# OCR wrappers
# -------------------------
def run_easyocr(img: np.ndarray, langs=['ko','en']):
    reader = easyocr.Reader(langs, gpu=False)  # GPU optional
    # easyocr readtext expects color or gray images; pass the (possibly rotated) image
    raw = reader.readtext(img, detail=1)  # returns list of (bbox, text, prob)
    # normalize to our box structure
    boxes = []
    for bbox, text, prob in raw:
        # bbox is list of 4 points [[x,y],...]
        xmin = min([p[0] for p in bbox])
        ymin = min([p[1] for p in bbox])
        xmax = max([p[0] for p in bbox])
        ymax = max([p[1] for p in bbox])
        boxes.append({
            "text": text,
            "conf": float(prob),
            "x1": int(xmin),
            "y1": int(ymin),
            "x2": int(xmax),
            "y2": int(ymax),
            "cx": int((xmin+xmax)/2),
            "cy": int((ymin+ymax)/2),
        })
    return boxes

def run_tesseract(img: np.ndarray):
    # pytesseract.image_to_data
    # ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    custom_oem_psm_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_oem_psm_config, lang='kor+eng')
    n = len(data['text'])
    boxes = []
    for i in range(n):
        text = data['text'][i].strip()
        if text == "":
            continue
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({
            "text": text,
            "conf": conf/100.0,
            "x1": int(x),
            "y1": int(y),
            "x2": int(x+w),
            "y2": int(y+h),
            "cx": int(x + w/2),
            "cy": int(y + h/2),
        })
    return boxes

# -------------------------
# Group boxes into rows
# -------------------------
def group_boxes_to_rows(boxes: List[Dict], row_tol=12) -> List[Dict]:
    """
    boxes: list with keys x1,y1,x2,y2,text,cx,cy
    returns rows: list of {"y": avg_y, "items": [box,...], "line_text": combined}
    row_tol: vertical tolerance in pixels (tweakable)
    """
    if not boxes:
        return []
    # sort by cy (top to bottom)
    boxes_sorted = sorted(boxes, key=lambda b: b['cy'])
    rows = []
    current_row = [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b['cy'] - np.mean([x['cy'] for x in current_row])) <= row_tol:
            current_row.append(b)
        else:
            # finalize current row
            # sort by x
            current_row = sorted(current_row, key=lambda x: x['x1'])
            line_text = " ".join([x['text'] for x in current_row])
            rows.append({
                "y": int(np.mean([x['cy'] for x in current_row])),
                "items": current_row,
                "line_text": line_text
            })
            current_row = [b]
    # last row
    current_row = sorted(current_row, key=lambda x: x['x1'])
    line_text = " ".join([x['text'] for x in current_row])
    rows.append({
        "y": int(np.mean([x['cy'] for x in current_row])),
        "items": current_row,
        "line_text": line_text
    })
    return rows

# -------------------------
# Normalization utilities
# -------------------------
def normalize_text(s: str) -> str:
    s = s.strip()
    # common OCR confusions
    s = s.replace('O', '0').replace('o','0')
    s = s.replace('l', '1').replace('I', '1')
    s = s.replace('S', '5')
    s = re.sub(r'\s+', ' ', s)
    return s

num_re = re.compile(r'(\d{1,3}(?:[,\.\s]\d{3})+|\d{1,9})')

def extract_numbers_from_text(s: str) -> List[str]:
    s = s.replace('·','').replace('•','')
    # unify common separators
    s = s.replace('.', ',')  # temporarily treat dots as comma (will strip later)
    # extract number-like tokens
    found = num_re.findall(s.replace(' ', ''))
    # Clean: remove leading zeros that are not meaningful? keep as is
    cleaned = []
    for f in found:
        f2 = re.sub(r'[^\d]', '', f)  # strip commas/dots/spaces
        if f2 == "":
            continue
        # ignore suspiciously long numeric lines (like barcode) when used as price (length>8)
        cleaned.append(f2)
    return cleaned

# -------------------------
# Row parsing heuristic
# -------------------------
def parse_rows_to_items(rows: List[Dict]) -> List[Dict]:
    items = []
    for r in rows:
        text = normalize_text(r['line_text'])
        nums = extract_numbers_from_text(text)
        # heuristic: if there are >=2 numbers and text contains hangul or alphabets (product name)
        # filter out rows that are purely barcode or header/footer
        # Consider barcode lines: numeric token length >= 8 and mostly on separate line -> skip
        if len(nums) == 0:
            continue
        # If the entire line is a single long number -> likely barcode/footer -> skip
        if len(nums) == 1 and len(nums[0]) >= 8 and not bool(re.search(r'[가-힣A-Za-z]', text)):
            continue
        # Now try to infer unit, qty, amount
        unit = None
        qty = None
        amount = None

        # If >=3 numbers, assume: ... unit, qty, amount (common)
        if len(nums) >= 3:
            unit = int(nums[-3])
            qty = int(nums[-2])
            amount = int(nums[-1])
        elif len(nums) == 2:
            # ambiguous: could be (unit, amount) with qty=1, or (qty, amount) without unit
            # Heuristic: if second >> first*1 => assume first=unit, second=amount
            a = int(nums[0])
            b = int(nums[1])
            # if b ~ a or b > a -> probably unit then qty? use qty=1 assumption
            # Assume qty = 1, unit=a, amount=b if b >= a
            if b >= a:
                unit = a
                qty = 1
                amount = b
            else:
                # otherwise unit missing -> qty=a, amount=b
                unit = None
                qty = a
                amount = b
        else:
            # single number -> amount only, assume qty=1
            amount = int(nums[-1])
            qty = 1
            unit = amount

        # product name: remove numeric tokens from end
        # Try to capture product name from the left side of row text by removing trailing numeric tokens
        # Remove sequences of numeric tokens at end
        prod_text = text
        # remove barcode-like token patterns
        prod_text = re.sub(r'\b\d{6,}\b', '', prod_text)
        # remove trailing number groups
        prod_text = re.sub(r'(\s*\d[\d,.\s]*)+$', '', prod_text).strip()
        # If prod_text is empty, use first text item as product name
        if prod_text == "":
            prod_text = r['items'][0]['text'] if r['items'] else text

        items.append({
            "product": prod_text,
            "unit": int(unit) if unit is not None else None,
            "qty": int(qty) if qty is not None else None,
            "amount": int(amount) if amount is not None else None,
            "raw_line": text,
            "y": r['y']
        })
    return items

# -------------------------
# Totals extraction
# -------------------------
def find_total_lines(rows: List[Dict]) -> Dict[str,int]:
    """
    Search for 합계, 면세, 부가세, 과세, 결제금액 등.
    Returns dict with keys found.
    """
    summary = {}
    keywords = {
        "total": ["합계", "총계", "결제금액", "결제 금액"],
        "tax_free": ["면세"],
        "vat": ["부가세", "부가 세", "부가세:" , "부가"],
        "taxable": ["과세", "과세:"],
    }
    for r in rows:
        text = normalize_text(r['line_text'])
        nums = extract_numbers_from_text(text)
        for key, kws in keywords.items():
            for kw in kws:
                if kw in text:
                    if nums:
                        # choose last numeric token on the line as value
                        val = int(nums[-1])
                        summary[key] = val
                    else:
                        # if no numeric on same line, try following items in nearby rows
                        pass
    # if 결제금액 not found, search for lines containing '결제' or '결제금액'
    # Also handle forms like "합 계 14,990"
    # fallback: look for largest number near bottom of receipt
    if "total" not in summary:
        # pick bottom-most numeric in rows
        bottom_nums = []
        for r in rows:
            nums = extract_numbers_from_text(r['line_text'])
            if nums:
                bottom_nums.append((r['y'], int(nums[-1]), r['line_text']))
        if bottom_nums:
            # choose the numeric from the lowest y (last lines)
            bottom_nums_sorted = sorted(bottom_nums, key=lambda x: x[0], reverse=True)
            summary["total_guess"] = bottom_nums_sorted[0][1]
    return summary

# -------------------------
# Main pipeline
# -------------------------
def parse_receipt_image(img_path: str, debug_prefix: str = None):
    img = load_image(img_path)
    gray, bw = preprocess(img, debug_save=(f"{debug_prefix}_preproc.png" if debug_prefix else None))

    # Choose OCR engine
    if OCR_ENGINE == "easyocr":
        boxes = run_easyocr(img)
    elif OCR_ENGINE == "pytesseract":
        boxes = run_tesseract(img)
    else:
        raise RuntimeError("No OCR engine available. Install easyocr or pytesseract + tesseract.")

    # group boxes to rows
    rows = group_boxes_to_rows(boxes, row_tol=max(12, img.shape[0]//100))

    # parse rows into items
    items = parse_rows_to_items(rows)

    # totals
    summary = find_total_lines(rows)

    # compute sum of parsed item amounts
    parsed_sum = sum([it['amount'] for it in items if it.get('amount')])

    result = {
        "items": items,
        "summary": summary,
        "parsed_sum": parsed_sum,
        "n_boxes": len(boxes),
        "n_rows": len(rows)
    }

    # debug image with bounding boxes and row texts
    if debug_prefix:
        vis = img.copy()
        # draw boxes
        for b in boxes:
            cv2.rectangle(vis, (b['x1'], b['y1']), (b['x2'], b['y2']), (0,255,0), 1)
        # draw row texts
        for r in rows:
            cv2.putText(vis, r['line_text'][:60], (5, r['y']+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(f"{debug_prefix}_annot.png", vis)

    return result

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="input image or folder")
    parser.add_argument("--out_csv", default="items.csv", help="output items CSV")
    parser.add_argument("--out_summary", default="summary.json", help="summary JSON")
    parser.add_argument("--debug", default=None, help="debug output prefix")
    args = parser.parse_args()

    inpath = args.input
    if os.path.isdir(inpath):
        img_files = [os.path.join(inpath, f) for f in os.listdir(inpath) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    else:
        img_files = [inpath]

    all_items = []
    summaries = {}
    for f in img_files:
        try:
            res = parse_receipt_image(f, debug_prefix=(args.debug or "debug"))
        except Exception as e:
            print("Error processing", f, e)
            continue
        # attach filename
        for it in res['items']:
            it['source_file'] = os.path.basename(f)
        all_items.extend(res['items'])
        summaries[os.path.basename(f)] = {
            "parsed_sum": res['parsed_sum'],
            "summary": res['summary'],
            "n_boxes": res['n_boxes'],
            "n_rows": res['n_rows']
        }

    # save items csv
    if all_items:
        df = pd.DataFrame(all_items)
        df.to_csv(args.out_csv, index=False)
        print("Saved items to", args.out_csv)
    else:
        print("No items parsed.")

    # save summary
    with open(args.out_summary, "w", encoding="utf-8") as fp:
        json.dump(summaries, fp, ensure_ascii=False, indent=2)
    print("Saved summary to", args.out_summary)
    print("Finished.")

if __name__ == "__main__":
    main()
