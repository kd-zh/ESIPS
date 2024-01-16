
import cv2
import numpy as np
import time
import os
import sys
from typing import List
from PIL import Image, ImageStat

import shape_wave
from utils import *
import ocr
import ocr_tesseract
import colorsys
import Levenshtein



MIN_CHAR_HEIGHT_WIDTH_RATION = 1
MAX_CHAR_HEIGHT_WIDTH_RATION = 3


CHAR_EASY_CONNECT_FONT_HEIGHT = 14

BAD_SCORE = -10


class LazyColor:
    def __init__(self, color, image=None, contour=None):
        self.image = image
        self.contour = contour
        self.color = color

    def get(self):
        if self.color is None:
            self.color = calc_contour_color(self.image, self.contour)
        return self.color


class Rect:
    def __init__(self, contour, bounds):
        self.contour = contour

        self.contour_id = None
        self.parent_contour_id = None
        self.bounds = bounds

    def top(self):
        return self.bounds[1]

    def bottom(self):
        return self.bounds[1] + self.bounds[3]

    def left(self):
        return self.bounds[0]

    def right(self):
        return self.bounds[0] + self.bounds[2]

    def width(self):
        return self.bounds[2]

    def height(self):
        return self.bounds[3]

    def inside(self, rect):
        if (
            self.left() >= rect.left()
            and self.right() <= rect.right()
            and self.top() >= rect.top()
            and self.bottom() <= rect.bottom()
        ):
            return True
        else:
            return False

    def guess_chars(self, avg_word_rect_height):
        if (
            self.width() < self.height()
            or self.height() > CHAR_EASY_CONNECT_FONT_HEIGHT
        ):

            return 1
        else:

            return self.width() / (avg_word_rect_height * 0.9)


class Row:
    def __init__(self):
        self.rects = []
        self.v_center = 0
        self.avg_height = 0
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None
        self.contour_level_id = None

    @classmethod
    def create_by_rects(cls, rects: List[Rect]):
        row = Row()
        row.rects = rects

        total_v_center = 0
        total_height = 0
        top = None
        bottom = None
        left = None
        right = None
        for r in rects:
            rect_v_center = r.top() + r.height() / 2
            total_v_center += rect_v_center
            total_height += r.height()
            if top is None or r.top() < top:
                top = r.top()
            if bottom is None or r.bottom() > bottom:
                bottom = r.bottom()

        row.top = top
        row.bottom = bottom
        row.v_center = total_v_center / len(rects)
        row.avg_height = total_height / len(rects)
        row.contour_level_id = rects[0].parent_contour_id
        return row


class Word:
    def __init__(self):
        self.rects = []
        self.bounds = None
        self.avg_rect_width = None
        self.avg_rect_height = None
        self.font_color = None
        self.char_count = None

    def left(self):
        return self.bounds[0]

    def top(self):
        return self.bounds[1]

    def bottom(self):
        return self.bounds[1] + self.bounds[3]

    def right(self):
        return self.bounds[0] + self.bounds[2]

    def width(self):
        return self.bounds[2]

    def height(self):
        return self.bounds[3]

    def guessed_char_width(self):

        return min(self.avg_rect_width, self.avg_rect_height * 2 / 3)

    def max_rect_height(self):
        return max(self.rects, key=lambda x: x.height())


class Phrase:
    def __init__(self):
        self.words = []

        self.bounds = None
        self.avg_rect_width = None
        self.avg_rect_height = None
        self.font_color = None
        self.top_wave = None
        self.bottom_wave = None

    def guessed_char_width(self):

        return min(self.avg_rect_width, self.avg_rect_height * 2 / 3)

    def rect_count(self):
        count = 0
        for w in self.words:
            count = count + len(w.rects)
        return count

    def left(self):
        return self.bounds[0]

    def top(self):
        return self.bounds[1]

    def right(self):
        return self.bounds[0] + self.bounds[2]

    def bottom(self):
        return self.bounds[1] + self.bounds[3]

    def width(self):
        return self.bounds[2]

    def height(self):
        return self.bounds[3]

    def max_rect_height(self):
        return max(self.words, key=lambda x: x.max_rect_height())


def annotate_rects(img, rects: List[Rect]):
    copy = img.copy()
    for rect in rects:
        x, y, w, h = rect.bounds

        color = (0, 0, 255)
        cv2.rectangle(copy, (x, y), (x + w, y + h), color, 1)
    return copy


def annotate_bounds(img, rects: list):
    copy = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return copy


def annotate_min_area_rects(img, rects: list):
    copy = img.copy()
    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(copy, [box], 0, (0, 0, 255), 2)
    return copy


def annotate_rows(img, rows: List[Row]):
    copy = img.copy()
    for row in rows:
        rleft = min(row.rects, key=lambda r: r.left())
        rright = max(row.rects, key=lambda r: r.right())
        x = rleft.left()
        y = int(row.v_center - row.avg_height / 2)
        w = rright.right() - x
        h = int(row.avg_height)
        color = (255, 0, 0)
        cv2.rectangle(copy, (x, y), (x + w, y + h), color, 1)

        for rect in row.rects:
            x, y, w, h = rect.bounds

            cv2.rectangle(copy, (x, y), (x + w, y + h), color, 1)
    return copy


def annotate_words(img, rows_of_words: list):
    copy = img.copy()

    for row in rows_of_words:
        for word in row:
            x, y, w, h = word.bounds
            color = (0, 0, 255)

            cv2.rectangle(copy, (x, y), (x + w, y + h), color, 1)
    return copy


def annotate_phrases(img, rows_of_phrases: list):
    copy = img.copy()
    for row in rows_of_phrases:
        for p in row:
            x, y, w, h = p.bounds
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if p.top_wave is not None:
                copy = cv2.putText(
                    copy,
                    p.top_wave,
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
    return copy


def annotate_match_candidates(img, candidate_phrases: list):
    copy = img.copy()
    index = 0
    for c in candidate_phrases:
        phrase = c[0]
        score = c[1]
        x, y, w, h = phrase.bounds
        cv2.rectangle(copy, (x, y), (x + w, y + h), (255, 0, 0), 4)
        copy = cv2.putText(
            copy,
            str(index) + ":" + str(score)[:4],
            (x, y - 2),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
        )
        index = index + 1
    return copy


def ocr_detect(
    image_file,
    cv_image=None,
    expected_bounds=None,
    tgt_text=None,
    is_label=False,
    debug=False,
    timeout=20,
    resize_rate=None,
    ocr_cache_folder=None,
):
    if cv_image is None:
        original_image = cv2.imread(image_file)
        if expected_bounds is not None:   
            x_min, y_min, width, height = expected_bounds
            x_max = x_min + width
            y_max = y_min + height
            original_image = original_image[y_min:y_max, x_min:x_max]
    else:
        original_image = cv_image

    resize_ratenum = 1
    if resize_ratenum != 1:
        (h, w) = original_image.shape[:2]
        dim = (int(w * resize_ratenum), int(h * resize_ratenum))
        cv_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    else:
        cv_image = original_image

    rect_list, canny, non_filtered_rects, min_area_rects = detect_char_rects(
        cv_image, True, debug, resize_ratenum
    )
    rows = group_rects_by_row(cv_image, rect_list)

    rows_of_words = []
    rows_of_phrases = []
    for r in rows:

        words = group_words(cv_image, r.rects)

        phrases = group_phrases(cv_image, words)
        rows_of_words.append(words)

        rows_of_phrases.append(phrases)

    if debug:

        cv2.imwrite(get_dbg_output_file(image_file, "canny.jpg"), canny)
        cv2.imwrite(
            get_dbg_output_file(image_file, "rect.jpg"),
            annotate_rects(original_image, non_filtered_rects),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "filtered_rect.jpg"),
            annotate_rects(original_image, rect_list),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "minarea_rect.jpg"),
            annotate_min_area_rects(cv_image, min_area_rects),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "rows.jpg"),
            annotate_rows(original_image, rows),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "words.jpg"),
            annotate_words(original_image, rows_of_words),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "phrases.jpg"),
            annotate_phrases(original_image, rows_of_phrases),
        )

    return original_image, rows_of_phrases


def match_row_of_phrases(image_file, cv_image, rows_of_phrases, tgt_text, debug):
    candidate_phrases, discard_phrases = guess_matched_phrases(
        cv_image, tgt_text, rows_of_phrases
    )

    boxes = []
    for c in candidate_phrases:
        phrase, score = c
        boxes.append((phrase.bounds, score))

    if debug:
        cv2.imwrite(
            get_dbg_output_file(image_file, "match.jpg"),
            annotate_match_candidates(cv_image, candidate_phrases),
        )
    return boxes, discard_phrases


def match_label_boxes(image_file, cv_image, bound_boxes, tgt_text, debug):

    label_len = len(tgt_text)
    candidates = []
    debug_id = 0
    discard_phrases = []

    for box in bound_boxes:
        x, y, w, h = box

        min_width, max_width = get_phrase_width_range(label_len, h)
        if w < label_len * h / 4:
            discard_phrases.append(box)
            continue
        elif w < min_width:
            bad_score = BAD_SCORE - abs((w / label_len) / h - 0.5)
            candidates.append((box, bad_score))
            continue
        textbox_image = cv_image[y : y + h, x : x + w]
        if not textbox_image.any():
            continue

        result = ocr_get_label_similarity(
            image_file, textbox_image, tgt_text, debug_id, debug
        )
        debug_id = debug_id + 1

        for bounds, score in result:
            new_bounds = (bounds[0] + x, bounds[1] + y, bounds[2], bounds[3])
            candidates.append((new_bounds, score))

    candidates.sort(key=lambda i: i[1], reverse=True)
    return candidates, discard_phrases


def ocr_get_label_similarity(image_file, textbox_image, label, debug_id, debug=False):
    (
        rect_list,
        canny,
        non_filtered_rects,
        min_area_rects,
    ) = detect_char_rects_for_text_line(textbox_image, False, debug)
    rect_list.sort(key=lambda i: i.bounds[0])
    words = group_words(textbox_image, rect_list)
    phrases = group_phrases(textbox_image, words)

    candidate_phrases, discard_phrases = guess_matched_phrases(
        textbox_image, label, [phrases]
    )
    boxes = []
    for c in candidate_phrases:
        phrase, score = c
        boxes.append((phrase.bounds, score))

    if debug:
        cv2.imwrite(get_dbg_output_file(image_file, "canny%d.jpg" % debug_id), canny)

        cv2.imwrite(
            get_dbg_output_file(image_file, "rect%d.jpg" % debug_id),
            annotate_rects(textbox_image, non_filtered_rects),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "filtered_rect%d.jpg" % debug_id),
            annotate_rects(textbox_image, rect_list),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "minarea_rect%d.jpg" % debug_id),
            annotate_min_area_rects(textbox_image, min_area_rects),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "words%d.jpg" % debug_id),
            annotate_words(textbox_image, [words]),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "phrases%d.jpg" % debug_id),
            annotate_phrases(textbox_image, [phrases]),
        )
        cv2.imwrite(
            get_dbg_output_file(image_file, "match%d.jpg" % debug_id),
            annotate_match_candidates(textbox_image, candidate_phrases[0:1]),
        )

    return boxes


CANNY_MIN = 200

CANNY_MAX = 200


def get_canny_thresholds():
    c_min = CANNY_MIN
    c_max = CANNY_MAX

    config_min = ocr.get_config("CANNY_MIN")
    if config_min is not None:
        c_min = config_min
    config_max = ocr.get_config("CANNY_MAX")
    if config_max is not None:
        c_max = config_min
    return c_min, c_max


def detect_char_rects(cv_image, is_full_screen, debug, resize_rate=1):
    if ocr.get_config("MEDIAN_BLUR"):

        base = cv2.medianBlur(cv_image, 3)

    else:
        base = cv_image

    img2 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    c_min, c_max = get_canny_thresholds()
    canny = cv2.Canny(img2, c_min, c_max)

    x = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(x) == 2:

        contours, hierarchy = x
    else:
        i, contours, hierarchy = x

    rect_list = []
    non_filtered_rects = []
    min_area_rects = []

    i = 0
    contours_list = []
    if len(contours) > 0:
        contours_list = [(0, -1, contours[0])]
        processed_contours = set()
        processed_contours.add(0)

    while i < len(contours_list):
        contour_id, parent_contour_id, c = contours_list[i]
        i += 1

        x, y, w, h = cv2.boundingRect(c)
        next_c, prev_c, child_c, parent_c = hierarchy[0][contour_id]
        if next_c != -1 and next_c not in processed_contours:
            processed_contours.add(next_c)
            contours_list.append((next_c, parent_c, contours[next_c]))

        if h >= 20 and w >= 60 and child_c != -1 and child_c not in processed_contours:

            area = cv2.contourArea(c)
            rect_area = w * h
            extent = float(area) / rect_area
            if extent > 0.9:
                processed_contours.add(child_c)
                contours_list.append((child_c, contour_id, contours[child_c]))

        rect = Rect(c, (x, y, w, h))
        rect.contour_id = contour_id
        rect.parent_contour_id = parent_contour_id

        if debug:

            mr = cv2.minAreaRect(c)
            min_area_rects.append(mr)

            non_filtered_rects.append(rect)
            mr_w = mr[1][0]
            mr_h = mr[1][1]
        if is_rect_filtered(cv_image, rect, is_full_screen):
            continue

        rect_list.append(rect)

    return rect_list, canny, non_filtered_rects, min_area_rects


def detect_char_rects_for_text_line(cv_image, is_full_screen, debug):

    canny = cv2.Canny(cv_image, CANNY_MIN, CANNY_MAX)

    x = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(x) == 2:
        contours, hierarchy = x
    else:
        i, contours, hierarchy = x

    rect_list = []
    non_filtered_rects = []
    min_area_rects = []

    i = 0
    while i < len(contours):
        c = contours[i]
        parent_contour_id = -1

        i += 1
        x, y, w, h = cv2.boundingRect(c)

        rect = Rect(c, (x, y, w, h))
        rect.contour_id = i
        rect.parent_contour_id = parent_contour_id

        if debug:

            mr = cv2.minAreaRect(c)
            min_area_rects.append(mr)
            non_filtered_rects.append(rect)

        if is_rect_filtered(cv_image, rect, is_full_screen):
            continue

        rect_list.append(rect)

    return rect_list, canny, non_filtered_rects, min_area_rects



def calc_contour_color(cv_image, contour):
    line_index = contour[:, :, 0]
    line_index = line_index.flatten().tolist()
    row_index = contour[:, :, 1]
    row_index = row_index.flatten().tolist()

    matrix = np.ix_(row_index, line_index)
    b = cv_image[:, :, 0]
    b_arr = b[matrix]
    b_vector = np.diag(b_arr)
    b_vector_mean = np.mean(b_vector)

    g = cv_image[:, :, 1]
    g_arr = g[matrix]
    g_vector = np.diag(g_arr)
    g_vector_mean = np.mean(g_vector)

    r = cv_image[:, :, 2]
    r_arr = r[matrix]
    r_vector = np.diag(r_arr)
    r_vector_mean = np.mean(r_vector)
    return (b_vector_mean, g_vector_mean, r_vector_mean)


def is_rect_filtered(image, rect, is_full_screen):

    width = rect.bounds[2]
    height = rect.bounds[3]

    if height < 5 or height / width > 20:
        return True

    if height > 30 and width / height > 2:
        return True

    if is_full_screen:

        screen_width = image.shape[1]
        if width > 0.25 * screen_width:
            return True

    return False




def find_best_fit_row(rows: List[Row], rect: Rect) -> Row:

    rect_top = rect.bounds[1]
    rect_height = rect.bounds[3]
    rect_bottom = rect_top + rect_height

    for row in reversed(rows):

        row_bottom = row.v_center + row.avg_height / 2
        h_iou = ocr.height_iou(
            row.v_center - row.avg_height / 2, row_bottom, rect_top, rect_bottom
        )
        min_top = row.v_center - 2 * row.avg_height
        max_bottom = row.v_center + 2 * row.avg_height
        in_range = rect_top > min_top and rect_bottom < max_bottom

        if in_range and h_iou > 0 and row.contour_level_id == rect.parent_contour_id:
            return row
        if rect_top - row.v_center > 40 * rect_height:
            break

    return None


def merge_rect_to_row(row, rect):

    row.rects.append(rect)

    rect_count = len(row.rects)
    rect_v_center = rect.bounds[1] + rect.bounds[3] / 2
    row.v_center = (row.v_center * (rect_count - 1) + rect_v_center) / rect_count
    row.avg_height = (row.avg_height * (rect_count - 1) + rect.bounds[3]) / rect_count

    if row.top is None or rect.top() < row.top:
        row.top = rect.top()
    if row.bottom is None or rect.bottom() > row.bottom:
        row.bottom = rect.bottom()


def is_row_overlapped(row1: Row, row2: Row, overlap_rate=0.8):

    if row1.contour_level_id != row2.contour_level_id:
        return False

    top1 = row1.top
    bottom1 = row1.bottom
    top2 = row2.top
    bottom2 = row2.bottom

    y_intersection = min(bottom1, bottom2) - max(top1, top2)
    min_height = min((bottom1 - top1), (bottom2 - top2))

    left1 = row1.rects[0].left()
    right1 = row1.rects[len(row1.rects) - 1].right()
    left2 = row2.rects[0].left()
    right2 = row2.rects[len(row2.rects) - 1].right()
    x_intersection = min(right1, right2) - max(left1, left2)

    y_overlapped = (y_intersection / min_height) >= overlap_rate
    x_overlapped = False
    if x_intersection >= 0:
        x_overlapped = True
    else:
        if left2 > right1:
            gap = left2 - right1
        elif left1 > right2:
            gap = left1 - right2
        else:
            raise Exception("error")

        if gap < 5 * max(row1.avg_height, row2.avg_height):
            x_overlapped = True

    return y_overlapped and x_overlapped


def merge_rows(dest: Row, src: Row):
    size = len(dest.rects) + len(src.rects)
    v_center = (dest.v_center * len(dest.rects) + src.v_center * len(src.rects)) / size
    avg_height = (
        dest.avg_height * len(dest.rects) + src.avg_height * len(src.rects)
    ) / size
    dest.v_center = v_center
    dest.avg_height = avg_height
    dest.top = min(src.top, dest.top)
    dest.bottom = max(src.bottom, dest.bottom)

    dest.rects.extend(src.rects)
    dest.rects.sort(key=lambda i: i.bounds[0])


def group_rects_by_row(image, rects: List[Rect]) -> List[Row]:
    rects.sort(key=lambda i: (i.bounds[1] + i.bounds[3]))

    rows = []
    for rect in rects:

        best_fit_row = find_best_fit_row(rows, rect)
        if best_fit_row is not None:

            merge_rect_to_row(best_fit_row, rect)
        else:

            row = Row()
            row.rects.append(rect)
            bounds = rect.bounds
            row.v_center = bounds[1] + bounds[3] / 2
            row.avg_height = bounds[3]
            row.contour_level_id = rect.parent_contour_id
            row.top = rect.top()
            row.bottom = rect.bottom()

            rows.append(row)

    refined_rows = []
    for row in rows:
        row.rects.sort(key=lambda i: i.bounds[0])
        broken = False
        right_start = 0

        char_width = row.avg_height
        prev_right = None
        for i in range(1, len(row.rects)):
            rect = row.rects[i]
            if prev_right is None:
                prev_right = row.rects[i - 1].right()
            else:
                prev_right = max(prev_right, row.rects[i - 1].right())

            if rect.left() - prev_right > 5 * char_width:
                broken_left = Row.create_by_rects(row.rects[right_start:i])
                refined_rows.append(broken_left)
                broken = True
                right_start = i

        if broken:

            broken_remain = Row.create_by_rects(row.rects[right_start:])
            refined_rows.append(broken_remain)
        else:
            refined_rows.append(row)

    merged_rows = []
    for k in range(0, len(refined_rows)):
        src = refined_rows[k]
        merged = False

        maxgap = 5 * src.avg_height
        src_top = src.v_center - src.avg_height / 2

        for m in range(len(merged_rows) - 1, -1, -1):
            dest = merged_rows[m]

            if src_top - (dest.v_center + dest.avg_height / 2) > maxgap:
                break

            if is_row_overlapped(dest, src):
                merge_rows(dest, src)
                merged = True
                break
        if not merged:
            merged_rows.append(src)

    rows = merged_rows

    return rows


def get_contour_points(contour) -> dict:
    points = {}
    for position in contour:
        [[x, y]] = position
        x_serial = points.get(y)
        if x_serial is None:
            x_serial = []
            points[y] = x_serial
        x_serial.append(x)
    for k, v in points.items():
        v.sort()
    return points


def get_overlap_region(start1, end1, start2, end2):
    if end1 < start2 or start1 > end2:
        return 0, 0
    else:
        return max(start1, start2), min(end1, end2)


def calc_contour_distance(left, right):
    left_points = get_contour_points(left)
    right_points = get_contour_points(right)

    left_ys = left_points.keys()
    l_top = min(left_ys)
    l_bottom = max(left_ys)

    right_ys = right_points.keys()
    r_top = min(right_ys)
    r_bottom = max(right_ys)

    overlap_top, overlap_bottom = get_overlap_region(l_top, l_bottom, r_top, r_bottom)

    gaps = []
    for y in range(overlap_top, overlap_bottom + 1):
        left_x = left_points.get(y)
        right_x = right_points.get(y)
        if left_x is None or right_x is None:
            gaps.append(sys.maxsize)
        else:
            l_r = left_x[len(left_x) - 1]
            r_l = right_x[0]
            gaps.append(r_l - l_r)
    distance = min(gaps)
    return distance

def is_y_overlapped(left, right):
    r_top = right.top()
    r_bottom = right.bottom()
    l_top = left.top()
    l_bottom = left.bottom()

    y_intersection = min(l_bottom, r_bottom) - max(l_top, r_top)
    if y_intersection < min(left.height(), right.height()) / 2:
        return False

    max_y_gap = max(right.height(), left.height()) / 2
    if (abs(l_bottom - r_bottom)) > max_y_gap or abs(l_top - r_top) > max_y_gap:
        return False

    return True


def is_rect_in_word(word: Word, rect: Rect):

    max_char_gap = max(3, word.avg_rect_height / 3)

    use_contour_distance = False
    if not use_contour_distance:
        r_left = rect.bounds[0]
        word_right = word.right()

        if r_left <= word_right:
            return True
        else:
            r_top = rect.top()
            r_bottom = rect.bottom()
            l_top = word.top()
            l_bottom = word.bottom()

            y_intersection = min(l_bottom, r_bottom) - max(l_top, r_top)
            if y_intersection < min(word.height(), rect.height()) / 2:
                return False

            max_y_gap = min(rect.height(), word.height()) / 2
            if (abs(l_bottom - r_bottom)) > max_y_gap or abs(l_top - r_top) > max_y_gap:
                return False
            return (r_left - word_right) < max_char_gap


def is_words_connected(left_word, right_word):

    max_char_gap = max(3, right_word.avg_rect_height / 3)
    rword_left = right_word.left()
    lword_right = left_word.right()

    if rword_left <= lword_right:

        return True
    else:
        r_top = right_word.top()
        r_bottom = right_word.bottom()
        l_top = left_word.top()
        l_bottom = left_word.bottom()

        y_intersection = min(l_bottom, r_bottom) - max(l_top, r_top)
        if y_intersection < min(left_word.height(), right_word.height()) / 2:
            return False

        max_y_gap = max(right_word.height(), left_word.height())
        if (abs(l_bottom - r_bottom)) > max_y_gap or abs(l_top - r_top) > max_y_gap:
            return False

        return (rword_left - lword_right) < max_char_gap


def group_words(image, rects: List[Rect]) -> List[Word]:

    words = []

    cur_word = None
    for rect in rects:

        r = rect.bounds
        r_width = r[2]
        r_height = r[3]

        if cur_word is not None:
            if is_rect_in_word(cur_word, rect):
                rect_count = len(cur_word.rects)
                cur_word.avg_rect_width = (
                    cur_word.avg_rect_width * rect_count + r_width
                ) / (rect_count + 1)
                cur_word.avg_rect_height = (
                    cur_word.avg_rect_height * rect_count + r_height
                ) / (rect_count + 1)
                cur_word.bounds = ocr.bounds_union(cur_word.bounds, r)
                cur_word.rects.append(rect)
            else:
                cur_word = None

        if cur_word is None:
            cur_word = Word()
            cur_word.bounds = r
            cur_word.rects.append(rect)
            cur_word.avg_rect_width = r_width
            cur_word.avg_rect_height = r_height
            words.append(cur_word)

    words2 = []
    last_word = None
    for w in reversed(words):

        if last_word is not None:

            if is_words_connected(w, last_word):

                last_word.avg_rect_width = (
                    last_word.avg_rect_width * len(last_word.rects)
                    + w.avg_rect_width * len(w.rects)
                ) / (len(last_word.rects) + len(w.rects))

                last_word.avg_rect_height = (
                    last_word.avg_rect_height * len(last_word.rects)
                    + w.avg_rect_height * len(w.rects)
                ) / (len(last_word.rects) + len(w.rects))

                last_word.bounds = ocr.bounds_union(last_word.bounds, w.bounds)

                for r in reversed(w.rects):
                    last_word.rects.insert(0, r)

            else:

                last_word = w
                words2.insert(0, last_word)
        else:
            last_word = w
            words2.insert(0, last_word)

    return words2


def is_word_in_phrase(phrase, word):

    max_word_gap = min(word.guessed_char_width(), phrase.guessed_char_width()) * 2

    if not is_y_overlapped(phrase, word):
        return False

    if (word.left() - phrase.right()) <= max_word_gap:
        return True
    else:
        return False


def group_phrases(image, words: List[Word]) -> List[Phrase]:
    phrases = []
    cur_phrase = None
    for w in words:
        if cur_phrase is not None:
            if is_word_in_phrase(cur_phrase, w):
                phrase_rect_count = cur_phrase.rect_count()

                cur_phrase.avg_rect_width = (
                    cur_phrase.avg_rect_width * phrase_rect_count
                    + w.avg_rect_width * len(w.rects)
                ) / (phrase_rect_count + len(w.rects))
                cur_phrase.avg_rect_height = (
                    cur_phrase.avg_rect_height * phrase_rect_count
                    + w.avg_rect_height * len(w.rects)
                ) / (phrase_rect_count + len(w.rects))

                cur_phrase.bounds = ocr.bounds_union(cur_phrase.bounds, w.bounds)

                cur_phrase.words.append(w)
            else:
                cur_phrase = None

        if cur_phrase is None:

            cur_phrase = Phrase()

            cur_phrase.bounds = w.bounds

            cur_phrase.words.append(w)

            cur_phrase.avg_rect_width = w.avg_rect_width

            cur_phrase.avg_rect_height = w.avg_rect_height

            phrases.append(cur_phrase)

    return phrases


def calc_avg_phrase_char_height(phrase):
    count = 0
    total = 0
    for word in phrase.words:
        for rect in word.rects:
            count = count + 1
            total = total + rect.bounds[3]
    return total / count


def count_phrase_chars(phrase, avg_char_height):
    count = 0
    last_rect = None
    for word in phrase.words:
        word_chars = 0
        for rect in word.rects:
            if rect == last_rect:
                continue

            if rect.height() <= avg_char_height / 2:
                continue

            if last_rect is not None and rect.inside(last_rect):
                last_rect = rect
                continue

            chars_in_rect = rect.guess_chars(word.avg_rect_height)
            count = count + chars_in_rect
            word_chars = word_chars + chars_in_rect

            last_rect = rect
        word.char_count = word_chars
    return count



EASY_DETECT_OPS = [
    "!",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "(",
    ")",
    "+",
    "=",
    "{",
    "}",
    "|",
    "[",
    "]",
    "\\",
    ":",
    '"',
    "'",
    "<",
    ">",
    "?",
    "/",
]


def count_label_chars(label: str):
    count = 0
    for ch in label:
        if not ch.isspace() and (
            ch.isdigit() or ch.isalpha() or not EASY_DETECT_OPS.__contains__(ch)
        ):
            count = count + 1
    return count


def find_label_words(label: str):
    count = 0
    words = []
    word = ""
    for ch in label:
        if ch.isspace():
            if len(word) > 0:
                count = count + 1
                words.append(word)
            word = ""
        else:
            word = word + ch
    if len(word) > 0:
        count = count + 1
        words.append(word)
    return words


CHAR_TOP_WAVE = {
    3: 'bdfhijklABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!$%(){}|[]?\\/"^',
    2: "t@#&",
    1: "acegmnopqrsuvwxyz+=:<>*",
    0: " ,'~–-_;,.\t",
}
CHAR_BOTTOM_WAVE = {
    2: "fabcdehiklmnorstuvwxz+=:<>ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!$%(){}|[]?\\/@#&*,\"'~^–-_;,",
    1: "gjpqy",
    0: " .",
}

WAVE_TYPE_UP = "^"
WAVE_TYPE_DOWN = "v"
WAVE_TYPE_STABLE = "-"


def get_char_level(WAVE, ch):
    for k, v in WAVE.items():
        if ch in v:
            return k
    return 0


def get_label_wave(WAVE: dict, label: str):
    last = None
    wave = ""
    for ch in label:
        if last is not None:
            last_level = get_char_level(WAVE, last)
            level = get_char_level(WAVE, ch)
            if ch == " ":
                w = " "
            elif level > last_level:
                w = WAVE_TYPE_UP
            elif level == last_level:
                w = WAVE_TYPE_STABLE
            else:
                w = WAVE_TYPE_DOWN
            wave = wave + w
        last = ch
    return wave


def get_phrase_wave(phrase: Phrase):
    last_word = None
    top_wave = ""
    bottom_wave = ""

    for word in phrase.words:

        if last_word is not None:

            w = " "
            top_wave = top_wave + w
            bottom_wave = bottom_wave + w

        last_rect = None
        for rect in word.rects:

            if last_word is None and last_rect is None:
                pass

            elif last_rect is None:
                w = WAVE_TYPE_UP
                top_wave = top_wave + w
                bottom_wave = bottom_wave + w

            else:
                last_top = last_rect.bounds[1]
                last_bottom = last_top + last_rect.bounds[3]
                cur_top = rect.bounds[1]
                cur_bottom = cur_top + rect.bounds[3]

                top_diff = cur_top - last_top
                bottom_diff = cur_bottom - last_bottom

                ref_height = min(last_bottom - last_top, cur_bottom - cur_top)
                relative_threshold = (1 / 5) * ref_height
                absolute_threshold = 5
                if (
                    abs(top_diff) >= relative_threshold
                    or abs(top_diff) >= absolute_threshold
                ):
                    if top_diff > 0:
                        w = WAVE_TYPE_DOWN
                    else:
                        w = WAVE_TYPE_UP
                else:
                    w = WAVE_TYPE_STABLE
                top_wave = top_wave + w

                if (
                    abs(bottom_diff) >= relative_threshold
                    or abs(bottom_diff) >= absolute_threshold
                ):
                    if bottom_diff > 0:
                        w = WAVE_TYPE_DOWN
                    else:
                        w = WAVE_TYPE_UP
                else:
                    w = WAVE_TYPE_STABLE
                bottom_wave = bottom_wave + w

            last_rect = rect

        last_word = word
    return top_wave, bottom_wave


def find_closest_value_index(data: list, val):
    min_diff = sys.maxsize
    index = -1
    for i in range(len(data)):
        diff = abs(data[i] - val)
        if diff < min_diff:
            min_diff = diff
            index = i

        if data[i] > val:
            break
    return index

def calc_words_score(label_words: list, phrase):
    label_words_count = len(label_words)
    phrase_words_count = len(phrase.words)

    if phrase_words_count < label_words_count:

        word_score = 1 - ((1 + 0.1) ** abs(phrase_words_count - label_words_count) - 1)
        common_len = min(phrase_words_count, label_words_count)
        for i in range(common_len):
            w = label_words[i]
            p = phrase.words[i]
            word_size_diff = min(abs(len(w) - p.char_count), len(w))
            word_score -= (1 / label_words_count) * (word_size_diff / len(w))
        return word_score
    else:
        len_diff = phrase_words_count - label_words_count
        max_word_score = 0
        for k in range(0, len_diff + 1):
            sum = 0
            for i in range(label_words_count):
                w = label_words[i]
                p = phrase.words[i + k]
                word_size_diff = min(abs(len(w) - p.char_count), len(w))
                sum += (1 / label_words_count) * (word_size_diff / len(w))
            word_score = 1 - sum
            if word_score > max_word_score:
                max_word_score = word_score

        if len_diff <= 1 and phrase.words[0].char_count <= 1.2:

            pass
        else:
            max_word_score -= (1 + 0.1) ** abs(
                phrase_words_count - label_words_count
            ) - 1
        return max_word_score

    return word_score


def calc_distibution_similarity(label_words: list, phrase, shift):
    points = []
    LABEL = 1
    PHRASE = 0
    pos = 0
    for w in label_words:
        p = pos + len(w)
        points.append((p, LABEL, len(w)))
        pos = p
    pos = shift
    for w in phrase.words:
        p = pos + w.char_count
        points.append((p, PHRASE, 0))
        pos = p

    points.sort(key=lambda x: x[0])

    sum = 0
    last_phrase_point = 0
    for i in range(len(points)):
        pos, source, word_len = points[i]
        if source == LABEL:
            left_gap = pos - last_phrase_point
            next_phrase_point = -1
            for k in range(i + 1, len(points)):
                if points[k][1] == PHRASE:
                    next_phrase_point = points[k][0]
                    break
            if next_phrase_point >= 0:
                gap = min(left_gap, next_phrase_point - pos)
            else:
                gap = left_gap
            score = min(gap, word_len) / word_len
            sum += score
        else:
            last_phrase_point = pos

    result = 1 - sum / len(label_words)
    return result


def get_phrase_width_range(label_len, phrase_height):
    min_width = label_len * phrase_height / 3

    max_width = label_len * phrase_height + phrase_height * 3
    return min_width, max_width


def merge_character_fragments(phrase):
    for word in phrase.words:
        last_rect = None
        new_rects = []
        for rect in word.rects:
            keep_rect = True
            if last_rect is not None:

                intersection = min(last_rect.right(), rect.right()) - max(
                    last_rect.left(), rect.left()
                )
                if intersection / min(last_rect.width(), rect.width()) > 0.65:
                    last_rect.bounds = ocr.bounds_union(last_rect.bounds, rect.bounds)
                    keep_rect = False

                else:
                    if len(new_rects) > 1:
                        last_last_rect = new_rects[len(new_rects) - 2]
                        intersect_left = min(
                            last_last_rect.right(), last_rect.right()
                        ) - max(last_last_rect.left(), last_rect.left())
                        if intersect_left / last_rect.width() > 0.25:
                            intersect_right = min(
                                rect.right(), last_rect.right()
                            ) - max(rect.left(), last_rect.left())

                            if (
                                intersect_left + intersect_right
                            ) / last_rect.width() > 0.8:
                                keep_rect = False
                                bounds = ocr.bounds_union(
                                    last_last_rect.bounds, last_rect.bounds
                                )
                                last_last_rect.bounds = ocr.bounds_union(
                                    bounds, rect.bounds
                                )
                                last_rect = last_last_rect
                                new_rects.pop()

            if keep_rect:
                new_rects.append(rect)
                last_rect = rect

        word.rects = new_rects

        total_height = 0
        for r in new_rects:
            total_height += r.height()
        word.avg_rect_height = total_height / len(new_rects)



def get_smallsize_shape_score(
    image, phrase: Phrase, label_top_wave, label_bottom_wave, avg_char_height
):
    img_h, img_w = image.shape[:2]
    top_wave = ""
    bottom_wave = ""
    phrase_char_count = 0

    first_word = True
    for w in phrase.words:
        x, y, width, height = w.bounds
        if x > 1:
            x -= 1
        if y > 1:
            y -= 1
        if x + width < img_w:
            width += 2
        if y + height < img_h:
            height += 2
        word_image = image[y : y + height, x : x + width]

        binary = shape_wave.get_label_binary(word_image)

        binary = binary[1 : height - 1, 0 : width - 1]

        w_top_wave, w_bottom_wave, w_char_distribution = shape_wave.get_shape_wave(
            binary, debug=False
        )
        if not first_word:
            top_wave += " "
            bottom_wave += " "

        top_wave += w_top_wave
        bottom_wave += w_bottom_wave

        last_index = 1
        char_count = 0
        for i in w_char_distribution:
            char_width = i - last_index
            last_index = i
            if char_width < w.avg_rect_height:

                char_count += 1
            else:

                char_count += char_width / (w.avg_rect_height * 0.9)

        w.char_count = char_count
        phrase_char_count += char_count
        first_word = False

    shape_score = shape_wave.calc_shape_wave_similarity(
        top_wave, bottom_wave, label_top_wave, label_bottom_wave
    )
    return shape_score, phrase_char_count


def get_small_size_label_wave(label_words):
    label_top_wave = ""
    label_bottom_wave = ""

    first_word = True
    for w in label_words:
        w_top_wave, w_bottom_wave = shape_wave.abstract_label_wave(w)
        if not first_word:
            label_top_wave += " "
            label_bottom_wave += " "

        label_top_wave += w_top_wave
        label_bottom_wave += w_bottom_wave
        first_word = False

    return label_top_wave, label_bottom_wave


def guess_matched_phrases(image, label: str, rows_of_phrases: list) -> list:
    label_len = len(label)
    label_chars = count_label_chars(label)
    label_words = find_label_words(label)

    label_top_wave = get_label_wave(CHAR_TOP_WAVE, label)
    label_bottom_wave = get_label_wave(CHAR_BOTTOM_WAVE, label)

    label_smallsize_top_wave, label_smallsize_bottom_wave = get_small_size_label_wave(
        label_words
    )

    candidate_phrases = []
    discard_phrases = []

    for row in rows_of_phrases:

        for phrase in row:

            phrase_width = phrase.bounds[2]
            phrase_height = phrase.bounds[3]

            min_width, max_width = get_phrase_width_range(label_len, phrase_height)

            if min_width < phrase_width < max_width:

                avg_char_height = calc_avg_phrase_char_height(phrase)
                merge_character_fragments(phrase)

                ups = 0
                downs = 0

                is_easy_connect_font = avg_char_height < CHAR_EASY_CONNECT_FONT_HEIGHT

                if is_easy_connect_font:
                    wave_score, phrase_chars = get_smallsize_shape_score(
                        image,
                        phrase,
                        label_smallsize_top_wave,
                        label_smallsize_bottom_wave,
                        avg_char_height,
                    )
                else:
                    phrase_chars = count_phrase_chars(phrase, avg_char_height)

                    phrase_top_wave, phrase_bottom_wave = get_phrase_wave(phrase)
                    phrase.top_wave = phrase_top_wave
                    phrase.bottom_wave = phrase_bottom_wave

                    r1 = Levenshtein.ratio(label_top_wave, phrase_top_wave)
                    r2 = Levenshtein.ratio(label_bottom_wave, phrase_bottom_wave)
                    wave_score = (r1 + r2) / 2

                char_score = 1 - (abs(phrase_chars - label_chars) / label_chars)
                if len(label_words) == 1:
                    score = max(0, 0.6 * char_score + 0.4 * wave_score)

                else:
                    word_score = calc_words_score(label_words, phrase)

                    score = max(
                        0, 0.5 * char_score + 0.3 * wave_score + 0.2 * word_score
                    )

                candidate_phrases.append((phrase, score))
            else:
                discard_phrases.append(phrase.bounds)

    candidate_phrases.sort(key=lambda i: i[1], reverse=True)
    return candidate_phrases, discard_phrases

def ocr_run(image_file, cv_image, debug=False, timeout=20, resize_rate=None):
    results = ocr_tesseract.ocr_run(image_file, cv_image, debug, timeout, resize_rate)
    return results


def ocr_recognize(image_file, cv_image=None, debug=False, timeout=20, resize_rate=None):
    results = ocr_tesseract.ocr_recognize(image_file, cv_image, debug)
    return results


