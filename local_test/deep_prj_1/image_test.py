import cv2
image_path = "/Users/nakrlove/Desktop/dev/python-deep/local_test/deep-prj-1/IMG_2709.jpg"
img = cv2.imread(image_path)
print(img)
import sys, os
sys.path.append(f"{os.getcwd()}/local_test/deep_prj_1")
# from ocrgvision import ocr_google_vision
from ocrgvision import ocr_google_vision
ocr_google_vision("./IMG_2709.jpg")


