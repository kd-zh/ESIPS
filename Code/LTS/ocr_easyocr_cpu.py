import easyocr
import cv2
from ocr import *
import ocr_easyocr as es


es_cpu = None

def ensure_easyocr_cpu():
    global es_cpu
    if es_cpu is None:
        es_cpu = easyocr.Reader(['en'], model_storage_directory='~/.EasyOCR/model',gpu=False)


def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    ensure_easyocr_cpu()
    return es.es_run(es_cpu, image_file, cv_image, debug, timeout, resize_rate)


def ocr_detect(image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    ensure_easyocr_cpu()
    return es.es_detect(es_cpu, image_file, cv_image, tgt_text, is_label, debug, timeout, resize_rate, ocr_cache_folder)


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    ensure_easyocr_cpu()
    return es.es_recognize(es_cpu, image_file, cv_image, debug, timeout, resize_rate)



