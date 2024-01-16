import sys
import time
import cv2
import os
from enum import Enum, auto
from typing import List
import ocr
import importlib
import Levenshtein
import numpy
import tesserocr

from utils import *
import json
import utils

VERSION = 0.04


class OCR_ALGO(Enum):
    TESSERACT = auto()
    TR = auto()
    PADDLE = auto()
    PADDLE_CPU = auto()
    PADDLE_CANNY_PADDLE_CPU = auto()
    PADDLE_CANNY_PADDLE = auto()
    EasyOCR = auto()
    EasyOCR_CPU = auto()
    EASYOCR_CANNY_EASYOCR = auto()
    EASYOCR_CANNY_EASYOCR_CPU = auto()

    TENCENT = auto()
    CANNY_TESSERACT = auto()
    CANNY_TESSERACT_REVERT = auto()
    CANNY_PADDLE = auto()
    CANNY_PADDLE_REVERT = auto()
    CANNY_PADDLE_CPU = auto()
    CANNY_PADDLE_CPU_REVERT = auto()
    CANNY_EasyOCR = auto()
    CANNY_EasyOCR_CPU_REVERT = auto()
    CANNY_EasyOCR_CPU = auto()
    CANNY_EasyOCR_REVERT = auto()

    EAST_TESSERACT = auto()
    PAN_PADDLE = auto()

    TESSERACT_REVERT = auto()


OCR_MODULE_NAME = {
    OCR_ALGO.TESSERACT: "ocr_tesseract",
    OCR_ALGO.TR: "ocr_tr",
    OCR_ALGO.PADDLE: "ocr_paddle",
    OCR_ALGO.PADDLE_CPU: "ocr_paddle_cpu",
    OCR_ALGO.EasyOCR: "ocr_easyocr",
    OCR_ALGO.EasyOCR_CPU: "ocr_easyocr_cpu",
    OCR_ALGO.TENCENT: "ocr_tencent",
    OCR_ALGO.EAST_TESSERACT: "ocr_east_tesseract",
    OCR_ALGO.PAN_PADDLE: "ocr_pan",
    OCR_ALGO.CANNY_TESSERACT: "ocr_canny",
    OCR_ALGO.CANNY_PADDLE: "ocr_canny_paddle",
    OCR_ALGO.CANNY_PADDLE_CPU: "ocr_canny_paddle_cpu",
    OCR_ALGO.CANNY_EasyOCR: "ocr_canny_easyocr",
    OCR_ALGO.CANNY_EasyOCR_CPU: "ocr_canny_easyocr_cpu",
}

TMP_IMAGE = "tmp.png"

TEXT_MATCH_THRESHOLD = 0.8

OCR_MODULE = {}

CONFIG = None


def __ensure_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = {"MEDIAN_BLUR": True}


def set_config(var, value):
    global CONFIG
    __ensure_config()
    CONFIG[var] = value


def get_config(var):
    global CONFIG
    __ensure_config()

    if var in CONFIG.keys():
        return CONFIG[var]
    else:
        return None


def load_ocr_module(algorithm: OCR_ALGO):
    module_name = OCR_MODULE_NAME[algorithm]
    print("\n====== Load %s ======" % (module_name))
    OCR_MODULE[algorithm] = importlib.import_module(module_name)
    print("====== End %s Loading ======" % (module_name))


def read_resized_cv2_image(image_file, resize_rate=None):
    if resize_rate is None:
        cv_image = cv2.imread(image_file)
    else:
        cv_image = cv2.imread(image_file)
        (h, w) = cv_image.shape[:2]

        dim = (int(w * resize_rate), int(h * resize_rate))
        cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
    return cv_image


def get_resized_image(image_file, resize_rate=None):
    cv_image = read_resized_cv2_image(image_file, resize_rate)
    resized_file = TMP_IMAGE
    cv2.imwrite(resized_file, cv_image)
    return resized_file


class OcrResult:
    def __init__(self, bounds, confidence, text) -> None:
        self.bounds = bounds
        self.confidence = confidence
        self.text = text

    @classmethod
    def from_json(cls, data):
        r = OcrResult(tuple(data["bounds"]), data["confidence"], data["text"])
        return r

    def __str__(self):
        return (
            "("
            + str(self.bounds)
            + ","
            + str(self.text)
            + ","
            + str(self.confidence)
            + ")"
        )


def get_bounds(point_list: list):
    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize
    y_max = 0

    for point in point_list:
        x_min = int(min(x_min, point[0]))
        x_max = int(max(x_max, point[0]))
        y_min = int(min(y_min, point[1]))
        y_max = int(max(y_max, point[1]))

    bounds = (x_min, y_min, x_max - x_min, y_max - y_min)
    return bounds


def annotate_ocr_results(img, ocr_results: List[OcrResult]):
    copy = img.copy()
    for r in ocr_results:
        x, y, w, h = r.bounds
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        copy = cv2.putText(
            copy, r.text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
        )

    return copy


def annotate_ocr_detections(img, ocr_detections):
    copy = img.copy()
    for r in ocr_detections:
        x, y, w, h = r
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return copy


def ocr_run(
    algorithm: OCR_ALGO, image_file, cv_image, debug=False, timeout=20, resize_rate=None
):
    module = OCR_MODULE[algorithm]
    ocr_func = getattr(module, "ocr_run", None)
    results = ocr_func(
        image_file, cv_image, debug=debug, timeout=timeout, resize_rate=resize_rate
    )
    return results


def ocr_detect(
    algorithm: OCR_ALGO,
    image_file,
    cv_image,
    expected_bounds,
    text=None,
    is_label=False,
    debug=False,
    timeout=20,
    resize_rate=1,
    ocr_cache_folder=None,
):
    module = OCR_MODULE[algorithm]
    ocr_func = getattr(module, "ocr_detect", None)
    cv_image, boxes = ocr_func(
        image_file,
        cv_image,
        expected_bounds,
        text,
        is_label,
        debug,
        timeout,
        resize_rate,
        ocr_cache_folder,
    )

    return cv_image, boxes


def ocr_recognize(algorithm: OCR_ALGO, image_file, cv_image, debug=False, timeout=20, resize_rate=None) -> List[OcrResult]:
    module = OCR_MODULE[algorithm]
    ocr_func = getattr(module, 'ocr_recognize', None)
    cv_image, text_blocks = ocr_func(image_file, cv_image, debug, timeout=timeout, resize_rate=resize_rate)

    if len(text_blocks)>0:
        if text_blocks[0].bounds is not None:
            phases = join_ocr_results(cv_image, text_blocks)
            return phases

    return text_blocks


def calc_ocr_text_match_rate(ocr_text, expected):
    if len(expected) == 0 or len(ocr_text) == 0:
        return 0

    diff2 = Levenshtein.distance(expected.lower(), ocr_text.lower()) / len(expected)
    without_case_sim = max(1.0 - diff2, 0)

    return without_case_sim


def is_ocr_ignore_case_text_match(ocr_text, expected) -> bool:
    if len(expected) == 0 or len(ocr_text) == 0:
        return False

    diff = Levenshtein.distance(expected.lower(), ocr_text.lower()) / len(expected)
    text_match_rate = max(1.0 - diff, 0)
    if text_match_rate >= TEXT_MATCH_THRESHOLD:
        return True
    else:
        return False


def collect_matched_ocr_results(
    ocr_results: List[OcrResult], label: str, detected_bounds
) -> List[OcrResult]:
    matched_results = []
    for r in ocr_results:
        text_match_rate = calc_ocr_text_match_rate(r.text, label)
        if text_match_rate >= TEXT_MATCH_THRESHOLD:
            if r.bounds is not None:
                if detected_bounds is not None:
                    x, y, w, h = detected_bounds
                    r.bounds = (
                        r.bounds[0] + x,
                        r.bounds[1] + y,
                        r.bounds[2],
                        r.bounds[3],
                    )
                else:
                    r.bounds = (r.bounds[0], r.bounds[1], r.bounds[2], r.bounds[3])
            else:
                r.bounds = detected_bounds
            r.confidence = r.confidence * text_match_rate
            matched_results.append(r)

    return matched_results


def match_text(
    algorithm: OCR_ALGO,
    image_path,
    text,
    expected_bounds = None,
    cv_image=None,
    debug=False,
    stat={},
    ocr_cache_folder_map=None,
):
    t1 = time.time()
    image, ocr_results = ocr_run(algorithm, image_path, cv_image, expected_bounds)
    ocr_time = time.time() - t1

    t1 = time.time()
    result = collect_matched_ocr_results(ocr_results, text, None)
    print(result[0])
    match_time = time.time() - t1

    stat["image"] = image_path
    stat["ocr_time"] = ocr_time
    stat["match_time"] = match_time
    stat["total_time"] = ocr_time + match_time

    # return image, result, ocr_results
    return result


def match_label(
    algorithm: OCR_ALGO,
    image_path,
    expected_bounds,
    label,
    cv_image=None,
    debug=False,
    stat={},
    check_head_strategy="all",
    revert=True,
    ocr_run_cache_folder_map=None,
    ocr_detect_cache_folder_map=None,
    rank_detected_boxes=True,
):

    t1 = time.time()
    image, text_bounds = ocr_detect(
            algorithm, image_path, cv_image, expected_bounds, label, True, debug
        )
    
    
    t2 = time.time()
    stat["detect_time"] = t2 - t1

    detect_num = len(text_bounds)
    if len(text_bounds) > 0 and isinstance(text_bounds[0], list):

        phrase_num = 0
        rows_of_phrases = text_bounds
        for row in rows_of_phrases:
            phrase_num += len(row)
        detect_num = phrase_num

    stat["detect_num"] = detect_num

    t_rank1 = time.time()
    ranked_text_bounds, discard_boxes = rank_detection_results(
        image_path, image, text_bounds, rank_detected_boxes, label, debug
    )
    t_rank2 = time.time()

    stat["image"] = image_path
    stat["rank_time"] = t_rank2 - t_rank1

    t_rec1 = time.time()
    to_rec = get_to_recognize_text_boxes(ranked_text_bounds, check_head_strategy)

    results = []
    index = -1
    for o in to_rec:
        index = index + 1
        bounds = list(o[0])
        x1, y1, x2, y2 = get_text_box_bounds(image, bounds, algorithm)
        cut_img = image[y1:y2, x1:x2]

        reco_results = ocr_recognize(algorithm, image_path, cut_img, debug)

        matched_results = collect_matched_ocr_results(reco_results, label, bounds)
        results.extend(matched_results)

        if len(results) > 0:
            stat["match_rank"] = index
            stat["match_bounds"] = bounds
            break

    t_rec2 = time.time()

    if "match_rank" not in stat.keys():
        stat["match_rank"] = -1

    stat["recognize_time"] = t_rec2 - t_rec1

    substat = {}
    if len(results) > 0:
        stat["label_detected"] = True
    else:
        stat["label_detected"] = False

        if revert:
            print("revert to match text", file=sys.stderr)
            image, results, ocr_results = match_text(
                algorithm,
                image_path,
                label,
                image,
                debug,
                substat,
                ocr_run_cache_folder_map,
            )
            stat["revert"] = substat

    if len(substat) == 0:
        stat["total_time"] = (
            stat["detect_time"] + stat["rank_time"] + stat["recognize_time"]
        )
    else:
        stat["total_time"] = (
            stat["detect_time"]
            + stat["rank_time"]
            + stat["recognize_time"]
            + substat["total_time"]
        )
    return (image, results, ranked_text_bounds, discard_boxes, text_bounds)


def ocr_detect_and_rank(
    algorithm: OCR_ALGO,
    image_path,
    label,
    cv_image=None,
    debug=False,
    stat={},
    ocr_cache_folder_map=None,
):

    t1 = time.time()
    image, text_bounds = ocr_detect(
        algorithm,
        image_path,
        cv_image,
        label,
        True,
        debug,
        ocr_cache_folder=ocr_cache_folder_map,
    )
    t2 = time.time()

    ranked_text_bounds, discard_boxes = rank_detection_results(
        image_path, image, text_bounds, True, label, debug
    )
    t3 = time.time()

    stat["image"] = image_path
    stat["detect_time"] = t2 - t1
    stat["detect_num"] = len(text_bounds)
    stat["detect_results"] = text_bounds
    stat["rank_time"] = t3 - t2

    return image, ranked_text_bounds


def rank_detection_results(
    image_path, image, text_bounds, rank_detected_boxes, label, debug
):
    discard_boxes = []
    if len(text_bounds) == 0:
        ranked_text_bounds = text_bounds
    else:
        first = text_bounds[0]

        if isinstance(first, list):

            if rank_detected_boxes:
                import ocr_canny

                ranked_text_bounds, discard_boxes = ocr_canny.match_row_of_phrases(
                    image_path, image, text_bounds, label, debug
                )
            else:

                ranked_text_bounds = []
                for t in text_bounds:
                    ranked_text_bounds.append((t.bounds, 1))
        else:
            if rank_detected_boxes:

                import ocr_canny

                ranked_text_bounds, discard_boxes = ocr_canny.match_label_boxes(
                    image_path, image, text_bounds, label, debug
                )
            else:
                ranked_text_bounds = []
                for t in text_bounds:
                    ranked_text_bounds.append((t, 1))

    return ranked_text_bounds, discard_boxes


def get_to_recognize_text_boxes(all_text_boxes, strategy):
    if strategy == "all":
        to_rec = all_text_boxes
    elif strategy == "closest":

        to_rec = []
        if len(all_text_boxes) > 0:
            top_score = all_text_boxes[0][1]
            for c in all_text_boxes:
                box, score = c
                if top_score - score < 0.1 or len(to_rec) < 3:
                    to_rec.append(box)
    elif isinstance(strategy, int):
        to_rec = all_text_boxes[:strategy]
    else:
        K = 20
        to_rec = all_text_boxes[:K]
    return to_rec


BOUNDS_EXPAND_SIZE_FOR_RECOGNITION = 5


def get_text_box_bounds(image, textbox, algorithm):
    length = BOUNDS_EXPAND_SIZE_FOR_RECOGNITION
    x, y, w, h = textbox
    x1 = max(x - length, 0)
    x2 = min(x + w + length, image.shape[1])
    y1 = max(y - length, 0)
    y2 = min(y + h + length, image.shape[0])
    return x1, y1, x2, y2


def height_iou(top1, bottom1, top2, bottom2):

    height_intersection = min(bottom1, bottom2) - max(top1, top2)

    height_union = max(bottom1, bottom2) - min(top1, top2)
    if height_union == 0:
        return height_intersection * 10
    height_IoU = height_intersection / height_union

    return height_IoU


def bounds_union(bounds1, bounds2):
    x = min(bounds1[0], bounds2[0])
    y = min(bounds1[1], bounds2[1])
    right = max(bounds1[0] + bounds1[2], bounds2[0] + bounds2[2])
    bottom = max(bounds1[1] + bounds1[3], bounds2[1] + bounds2[3])
    return (x, y, right - x, bottom - y)


def merge_ocr_results(ocr1, ocr2):
    bounds = bounds_union(ocr1.bounds, ocr2.bounds)
    confidence = min(ocr1.confidence, ocr2.confidence)
    o = OcrResult(bounds, confidence, ocr1.text + " " + ocr2.text)
    return o


def group_ocr_results_by_row(
    image, ocr_results: List[OcrResult], min_height_iou=0.2
) -> list:

    rows = []
    cur_row = None
    row_top = 0
    row_bottom = 0
    for w in ocr_results:
        if type(w) == numpy.ndarray:

            continue
        if type(w) == OcrResult:

            top = w.bounds[1]
            bottom = w.bounds[1] + w.bounds[3]
            if cur_row is not None:
                if height_iou(row_top, row_bottom, top, bottom) > min_height_iou:
                    cur_row.append(w)
                    row_top = min(row_top, top)
                    row_bottom = max(row_bottom, bottom)
                else:
                    cur_row = None

            if cur_row is None:
                cur_row = [w]
                rows.append(cur_row)
                row_top = top
                row_bottom = bottom
        if type(w) == list:
            for x in w:
                top = x.bounds[1]
                bottom = x.bounds[1] + x.bounds[3]
                if cur_row is not None:
                    if height_iou(row_top, row_bottom, top, bottom) > min_height_iou:
                        cur_row.append(x)
                        row_top = min(row_top, top)
                        row_bottom = max(row_bottom, bottom)
                    else:
                        cur_row = None

                if cur_row is None:
                    cur_row = [x]
                    rows.append(cur_row)
                    row_top = top
                    row_bottom = bottom

    return rows


def get_letter_width(w):

    if len(w.text) == 0:
        height = w.bounds[3]
        return height * 2 / 3
    else:
        width1 = w.bounds[2] / len(w.text)
        width2 = w.bounds[3] * 2 / 3
        return max(width1, width2)


def is_word_in_phrase(word, phrase):

    max_y_gap = max(word.bounds[3], phrase.bounds[3]) / 2
    w_top = word.bounds[1]
    p_top = phrase.bounds[1]
    w_bottom = w_top + word.bounds[3]
    p_bottom = p_top + phrase.bounds[3]

    if (abs(w_bottom - p_bottom)) > max_y_gap or abs(w_top - p_top) > max_y_gap:
        return False

    letter_width1 = get_letter_width(phrase)
    letter_width2 = get_letter_width(word)
    max_gap = 2 * min(letter_width1, letter_width2)
    if (word.bounds[0] - (phrase.bounds[0] + phrase.bounds[2])) < max_gap:
        return True

    return False


def join_ocr_results(image, ocr_results: List[OcrResult]) -> List[OcrResult]:
    rows = group_ocr_results_by_row(image, ocr_results, 0.2)

    phases = []
    for row in rows:
        row.sort(key=lambda i: i.bounds[0])

        cur_phase = None
        for w in row:
            if cur_phase is not None:
                if len(cur_phase.text) == 0:
                    continue

                if is_word_in_phrase(w, cur_phase):
                    cur_phase = merge_ocr_results(cur_phase, w)
                else:
                    phases.append(cur_phase)
                    cur_phase = None

            if cur_phase is None:
                cur_phase = w

        if cur_phase is not None:
            phases.append(cur_phase)
    return phases


def group_bounds_by_row(image, bounds: list, min_height_iou=0.2) -> list:

    rows = []
    cur_row = None
    row_top = 0
    row_bottom = 0
    for w in bounds:
        top = w[1]
        bottom = w[1] + w[3]

        if cur_row is not None:
            if height_iou(row_top, row_bottom, top, bottom) > min_height_iou:
                cur_row.append(w)
                row_top = min(row_top, top)
                row_bottom = max(row_bottom, bottom)
            else:
                cur_row = None

        if cur_row is None:
            cur_row = [w]
            rows.append(cur_row)
            row_top = top
            row_bottom = bottom
    return rows


def join_bounds(image, bounds: list) -> list:

    bounds.sort(key=lambda i: (i[1] + i[3]))

    rows = group_bounds_by_row(image, bounds, 0.2)

    phases = []
    for row in rows:
        row.sort(key=lambda i: i[0])

        cur_phase = None
        for w in row:
            if cur_phase is not None:
                max_gap = 2 * cur_phase[3]
                if (w[0] - (cur_phase[0] + cur_phase[2])) < max_gap:
                    cur_phase = bounds_union(cur_phase, w)
                else:
                    phases.append(cur_phase)
                    cur_phase = None

            if cur_phase is None:
                cur_phase = w

        if cur_phase is not None:
            phases.append(cur_phase)

    return phases
