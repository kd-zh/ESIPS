from paddleocr import PaddleOCR
import ocr_paddle as pd


pd_cpu = None


def ensure_paddle_cpu():
    global pd_cpu

    if pd_cpu is None:
        pd_cpu = PaddleOCR(lang="en", use_gpu=False)


def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    ensure_paddle_cpu()
    return pd.paddle_run(pd_cpu, image_file, cv_image, debug, timeout, resize_rate)


def ocr_detect(image_file, cv_image=None, tgt_text=None, is_label=False, debug=False, timeout=20, resize_rate=None, ocr_cache_folder=None):
    ensure_paddle_cpu()
    return pd.paddle_detect(pd_cpu, image_file, cv_image, tgt_text, is_label, debug, timeout, resize_rate, ocr_cache_folder)


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    ensure_paddle_cpu()
    return pd.paddle_recognize(pd_cpu, image_file, cv_image, debug, timeout, resize_rate)
