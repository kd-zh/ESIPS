import time

from PIL import Image, ImageStat
from paddleocr import PaddleOCR, draw_ocr

import cv2

import ocr_canny
from ocr import *

pd_gpu = None


def ensure_paddle_gpu():
    global pd_gpu

    if pd_gpu is None:
        
        
        
        pd_gpu = PaddleOCR(lang="en", use_gpu=True)


def paddle_run(pd_ocr, image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    paddle_result = pd_ocr.ocr(cv_image)
    results = []
    for r in paddle_result:
        bounds = get_bounds(r[0])
        text = r[1][0]
        confidence = r[1][1]
        tb = OcrResult(bounds, confidence, text)
        results.append(tb)
    return cv_image, results


def paddle_detect(pd_ocr, image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    paddle_result = pd_ocr.ocr(cv_image, rec=False)
    boxes = []
    for region in paddle_result:
        
        lt, rt, rb, lb = tuple(region)
        x_min = int(min(lt[0], rt[0], rb[0], lb[0]))
        x_max = int(max(lt[0], rt[0], rb[0], lb[0]))
        y_min = int(min(lt[1], rt[1], rb[1], lb[1]))
        y_max = int(max(lt[1], rt[1], rb[1], lb[1]))
        bounds = (x_min, y_min, x_max-x_min, y_max-y_min)
        boxes.append(bounds)
    return cv_image, boxes


def annotate_bounds(img, rects):
    copy = img.copy()
    index = 0
    for region in rects:
        x, y, w, h = region
        color = (0, 0, 255)
        cv2.rectangle(copy, (x, y), (x + w, y + h), color, 1)
        copy = cv2.putText(copy, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        
        index = index+1
    return copy


def paddle_recognize(pd_ocr, image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file)
    paddle_result = pd_ocr.ocr(cv_image, det=False)
    results = []
    for r in paddle_result:
        text, confidence = r
        tb = OcrResult(None, confidence, text)
        results.append(tb)
    return cv_image, results


def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    ensure_paddle_gpu()
    return paddle_run(pd_gpu, image_file, cv_image, debug, timeout, resize_rate)


def ocr_detect(image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    ensure_paddle_gpu()
    return paddle_detect(pd_gpu, image_file, cv_image, tgt_text, is_label, debug, timeout, resize_rate, ocr_cache_folder)


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    ensure_paddle_gpu()
    return paddle_recognize(pd_gpu, image_file, cv_image, debug, timeout, resize_rate)