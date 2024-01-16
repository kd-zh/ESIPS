"""
"""
import ocr_easyocr_cpu
import  ocr_canny


def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    return ocr_easyocr_cpu.ocr_run(image_file, cv_image, debug, timeout, resize_rate)


def ocr_detect(image_file, cv_image=None, expected_bounds=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    return ocr_canny.ocr_detect(image_file, cv_image, expected_bounds, tgt_text, is_label, debug, timeout, resize_rate, ocr_cache_folder)


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    return ocr_easyocr_cpu.ocr_recognize(image_file, cv_image, debug, timeout, resize_rate)


