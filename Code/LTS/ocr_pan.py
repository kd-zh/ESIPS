import cv2
from mmocr.utils.ocr import MMOCR
import utils
from ocr import *
import ocr_tesseract
import ocr_canny
from utils import *
import os
import ocr
import time
import ocr_paddle
def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file)
    
    ocr = MMOCR(det='PANet_IC15',recog = 'SEG',device='cpu')
    results = ocr.readtext(cv_image,details=True)

    res = results[0]
    result= res['result']
    boxes=[]

    for dict in result:
        box =dict['box']
        odd = box[::2]
        even = box[1::2]
        odd.sort()
        even.sort()
        x_min = odd[0]
        x_max = odd[len(odd)-1]
        y_min = even[0]
        y_max = even[len(even) - 1]
        bounds = (x_min, y_min, x_max - x_min, y_max - y_min)
        text = dict['text']
        confidence = dict['text_score']
        tb = OcrResult(bounds, confidence, text)
        boxes.append(tb)

    return cv_image, boxes

def ocr_detect(image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None,ocr_cache_folder=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file)

    ocr = MMOCR(det='PANet_CTW', recog=None)
    results = ocr.readtext(cv_image)
    res = results[0]
    result= res['boundary_result']
    boxes=[]
    for c in result:
        odd = c[:-1:2]
        even = c[1:-1:2]
        odd.sort()
        even.sort()
        x_min = int(odd[0])
        x_max = int(odd[len(odd)-1])
        y_min = int(even[0])
        y_max = int(even[len(even) - 1])
        bounds = (x_min, y_min, x_max - x_min, y_max - y_min)
        boxes.append(bounds)
    
    
    return cv_image, boxes


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    return ocr_paddle.ocr_recognize(image_file, cv_image, debug, timeout, resize_rate)
    
    
