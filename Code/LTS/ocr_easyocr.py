"""
https://pypi.org/project/easyocr/
https://github.com/JaidedAI/EasyOCR
"""
import time

import easyocr
import cv2
from ocr import *
import ocr_canny
import  ocr_canny


es_gpu = None


def ensure_easyocr_gpu():
    global es_gpu

    if es_gpu is None:
        
        
        es_gpu = easyocr.Reader(['en'], model_storage_directory='~/.EasyOCR/model', gpu=True)



def unify_ocr_results(ocr_results: list):
    text_blocks = []
    for r in ocr_results:
        
        lt, rt, rb, lb = tuple(r[0])
        
        x_min = int(min(lt[0], rt[0], rb[0], lb[0]))
        x_max = int(max(lt[0], rt[0], rb[0], lb[0]))
        y_min = int(min(lt[1], rt[1], rb[1], lb[1]))
        y_max = int(max(lt[1], rt[1], rb[1], lb[1]))
        bounds = (x_min, y_min, x_max - x_min, y_max - y_min)
        tb = OcrResult(bounds, r[2], r[1])
        text_blocks.append(tb)
    return text_blocks


def es_run(es_ocr, image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    results = es_ocr.readtext(cv_image)
    text_blocks = unify_ocr_results(results)
    return cv_image, text_blocks


def es_detect(es_ocr, image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    horizontal_list, free_list = es_ocr.detect(cv_image)
    boxes = []
    for region in horizontal_list[0]:
        x_min, x_max, y_min, y_max = tuple(region)
        bounds = (x_min, y_min, x_max-x_min, y_max-y_min)
        boxes.append(bounds)
    for region in free_list[0]:
        
        lt, rt, rb, lb = tuple(region)
        
        x_min = int(min(lt[0], rt[0], rb[0], lb[0]))
        x_max = int(max(lt[0], rt[0], rb[0], lb[0]))
        y_min = int(min(lt[1], rt[1], rb[1], lb[1]))
        y_max = int(max(lt[1], rt[1], rb[1], lb[1]))
        bounds = (x_min, y_min, x_max-x_min, y_max-y_min)
        boxes.append(bounds)

    return cv_image, boxes


def es_recognize(es_ocr, image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    
    results = es_ocr.recognize(cv_image)
    text_blocks = unify_ocr_results(results)
    return cv_image, text_blocks


def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    ensure_easyocr_gpu()
    return es_run(es_gpu, image_file, cv_image, debug, timeout, resize_rate)


def ocr_detect(image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    ensure_easyocr_gpu()
    return es_detect(es_gpu, image_file, cv_image, tgt_text, is_label, debug, timeout, resize_rate, ocr_cache_folder)


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    ensure_easyocr_gpu()
    return es_recognize(es_gpu, image_file, cv_image, debug, timeout, resize_rate)


