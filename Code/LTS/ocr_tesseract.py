import ocr_canny

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from pytesseract import image_to_string, image_to_boxes, image_to_data
from tesserocr import PyTessBaseAPI, RIL, PSM, OEM, iterate_level
import cv2
from ocr import *
import platform



def config_tesseract():
    if platform.system().lower() == "windows":

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract"
        )
    elif platform.system().lower() == "linux":
        pass


config_tesseract()

def unify_tesseract_results(results: list):
    text_blocks = []
    for i in range(0, len(results["text"])):

        tl_x = results["left"][i]
        tl_y = results["top"][i]
        width = results["width"][i]
        height = results["height"][i]
        bounds = (tl_x, tl_y, width, height)
        level = results["level"][i]
        conf = results["conf"][i]
        conf = float(conf)
        text = results["text"][i]

        if level == 5 and len(text.strip()) > 0:
            tb = OcrResult(bounds, conf, text)
            text_blocks.append(tb)
    return text_blocks


def ocr_run(image_file, cv_image, expected_bounds, debug=False, timeout=20, resize_rate=None):

    if cv_image is None:
        cv_image = cv2.imread(image_file)
        if expected_bounds is not None:   
            x_min, y_min, width, height = expected_bounds
            x_max = x_min + width
            y_max = y_min + height
            cropped_image = cv_image[y_min:y_max, x_min:x_max]

    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    config_tesseract()

    try:
        results = pytesseract.image_to_data(
            img_rgb, output_type=pytesseract.Output.DICT, timeout=timeout
        )
    except:
        results = {"text": []}

    if debug:

        text = image_to_string(img_rgb)

        if len(text) > 3:
            string = text[:-2]

            text_blocks = unify_tesseract_results(results)
            return cv_image, text_blocks
        else:
            cv_show("cv_image", img_rgb)
            raise Exception("tesseract recognize fail")

    text_blocks = unify_tesseract_results(results)
    return cv_image, text_blocks


def ocr_detect(
    image_file,
    cv_image=None,
    expected_bounds=None,
    tgt_text=None,
    is_label=False,
    debug=False,
    timeout=20,
    resize_rate=None,
    ocr_cache_folder=False,
):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file, resize_rate)

    boxes = []
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    custom_oem_psm_config = r"-l eng --oem 3 "
    results = pytesseract.image_to_data(
        img_rgb, output_type=pytesseract.Output.DICT, timeout=timeout
    )
    t2 = time.time()

    for i in range(0, len(results["text"])):

        tl_x = results["left"][i]
        tl_y = results["top"][i]
        width = results["width"][i]
        height = results["height"][i]
        bounds = (tl_x, tl_y, width, height)
        boxes.append(bounds)

    return cv_image, boxes

def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    if cv_image is None:
        cv_image = read_resized_cv2_image(image_file)

    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    text_blocks = []
    with PyTessBaseAPI(
        psm=PSM.SINGLE_LINE,
        oem=OEM.DEFAULT,
        lang="eng",
    ) as api:
        api.SetImage(img_rgb)
        api.Recognize(timeout * 1000)
        it = api.GetIterator()
        level = RIL.WORD
        for i, w in enumerate(iterate_level(it, level)):
            try:
                text = w.GetUTF8Text(level)
                conf = w.Confidence(level)
                x1, y1, x2, y2 = w.BoundingBox(level)

                bounds = (x1, y1, x2 - x1, y2 - y1)
                conf = float(conf)
                tb = OcrResult(bounds, conf, text)
                text_blocks.append(tb)
            except:

                pass

    return cv_image, text_blocks



