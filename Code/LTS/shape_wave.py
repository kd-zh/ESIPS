import Levenshtein
import cv2
import numpy as np
import utils
from utils import *


CHAR_TOP_EDGE_WAVE = {
    "A": "353",
    "B": "554",
    "C": "454",
    "D": "554",
    "E": "555",
    "F": "555",
    "G": "454",
    "H": "535",
    "I": "050",
    "J": "255",
    "K": "535",
    "L": "511",
    "M": "535",
    "N": "535",
    "O": "454",
    "P": "554",
    "Q": "454",
    "R": "554",
    "S": "353",
    "T": "555",
    "U": "515",
    "V": "515",
    "W": "53535",
    "X": "535",
    "Y": "535",
    "Z": "555",
    "a": "343",
    "b": "533",
    "c": "343",
    "d": "335",
    "e": "343",
    "f": "355",
    "g": "345",
    "h": "543",
    "i": "454",
    "j": "454",
    "k": "524",
    "l": "511",
    "m": "434",
    "n": "433",
    "o": "343",
    "p": "433",
    "q": "334",
    "r": "434",
    "s": "343",
    "t": "454",
    "u": "414",
    "v": "424",
    "w": "42424",
    "x": "424",
    "y": "424",
    "z": "444",
    "0": "454",
    "1": "151",
    "2": "454",
    "3": "555",
    "4": "345",
    "5": "555",
    "6": "343",
    "7": "555",
    "8": "454",
    "9": "454",
    "&": "343",
    "_@#$(){}|[]": "",
    "+–-*/%=~<>": "",
    "!?,.:;": "",
    "\\\"'^": "",
    " ": "000",
    "\t": "000",
}


CHAR_BOTTOM_EDGE_WAVE = {
    "A": "434",
    "B": "443",
    "C": "343",
    "D": "443",
    "E": "444",
    "F": "433",
    "G": "343",
    "H": "424",
    "I": "242",
    "J": "341",
    "K": "424",
    "L": "444",
    "M": "424",
    "N": "424",
    "O": "343",
    "P": "433",
    "Q": "445",
    "R": "434",
    "S": "343",
    "T": "141",
    "U": "343",
    "V": "242",
    "W": "434",
    "X": "434",
    "Y": "141",
    "Z": "444",
    "a": "344",
    "b": "443",
    "c": "434",
    "d": "344",
    "e": "343",
    "f": "353",
    "g": "354",
    "h": "414",
    "i": "242",
    "j": "251",
    "k": "424",
    "l": "043",
    "m": "41414",
    "n": "414",
    "o": "343",
    "p": "543",
    "q": "345",
    "r": "411",
    "s": "343",
    "t": "141",
    "u": "343",
    "v": "242",
    "w": "434",
    "x": "424",
    "y": "353",
    "z": "444",
    "0": "343",
    "1": "242",
    "2": "444",
    "3": "343",
    "4": "334",
    "5": "343",
    "6": "343",
    "7": "432",
    "8": "343",
    "9": "343",
    "&": "343",
    "_@#$(){}|[]": "",
    "+–-*/%=~<>": "",
    "!?,.:;": "",
    "\\\"'^": "",
    " ": "000",
    "\t": "000",
}

DEFAULT_WAV = "343"


def is_char_white_color(binary_img):
    (h, w) = binary_img.shape
    top_row = binary_img[0:1, :]
    bottom_row = binary_img[h - 1 : h, :]
    left_column = binary_img[:, 0:1]
    right_column = binary_img[:, w - 1 : w]

    mean1 = np.mean(top_row)
    mean2 = np.mean(bottom_row)
    mean3 = np.mean(left_column)
    mean4 = np.mean(right_column)
    mean = (mean1 + mean2 + mean3 + mean4) / 4
    if mean > 255 / 2:

        return False
    else:

        return True


def get_label_binary(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )

    if not is_char_white_color(binary):
        binary = 255 - binary

    return binary


def get_char_wav(ch, char_waves):
    if ch in char_waves.keys():
        w = char_waves[ch]
    else:
        w = DEFAULT_WAV
    return w


def get_label_shape_wave(label, char_waves):
    wave = []
    for c in label:
        w = get_char_wav(c, char_waves)
        for h in w:
            size = int(h)
            wave.append(size)

        wave.append(0)
    return wave


def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def abstract_wave(binary, arr, from_image, count_char):

    max_a = max(arr)
    if from_image:

        tall = max_a - 1
    else:
        tall = max_a

    left = 0
    last_trends = None
    char_distribution = []
    trends = []
    trends_index = []
    last_sep = 0
    for k in range(0, len(arr)):
        elmt = arr[k]

        if elmt > left:
            v = 1
        elif elmt < left:
            v = -1
        else:
            v = 0
        if count_char and left > 0:
            if elmt == 0:
                last_sep = k
                char_distribution.append(k)

            elif k - last_sep > 0 and abs(elmt - left) > 2 and binary is not None:

                left = binary[:, k - 1]
                right = binary[:, k]

                sum = left + right + 1
                if np.max(sum) < 255:

                    left_shift_down = shift(left, 1, 0)
                    sum = left_shift_down + right + 1
                    if np.max(sum) < 255:
                        left_shift_up = shift(left, -1, 0)
                        sum = left_shift_up + right + 1
                        if np.max(sum) < 255:
                            last_sep = k
                            char_distribution.append(k)

        left = elmt

        if v == 0:
            continue
        else:
            if last_trends == v:

                trends_index[len(trends_index) - 1] = k
            else:
                trends.append(v)
                trends_index.append(k)

        last_trends = v

    if count_char and arr[k] > 0:
        char_distribution.append(k + 1)

    abstract = ""
    for k in range(0, len(trends)):

        t_index = trends_index[k]
        if trends[k] == 1:
            if arr[t_index] >= tall:
                abstract += "|"
            else:
                abstract += "^"
        elif trends[k] == -1:

            abstract += ""

    return abstract, char_distribution


def get_shape_wave(binary, debug=False):

    (h, w) = binary.shape

    line = np.ones(w) * 255
    c = np.insert(binary, h, values=line, axis=0)
    top = np.argmax(c, axis=0)
    top = h - top

    c = np.flipud(binary)
    c = np.insert(c, h, values=line, axis=0)
    bottom = np.argmax(c, axis=0)
    bottom = h - bottom

    top_wave, char_distribution_top = abstract_wave(binary, top, True, True)
    bottom_wave, _ = abstract_wave(binary, bottom, True, False)
    return top_wave, bottom_wave, char_distribution_top


def abstract_label_wave(label):
    label_tops = get_label_shape_wave(label, CHAR_TOP_EDGE_WAVE)
    label_top_wave, _ = abstract_wave(None, label_tops, False, False)

    label_bottoms = get_label_shape_wave(label, CHAR_BOTTOM_EDGE_WAVE)
    label_bottom_wave, _ = abstract_wave(None, label_bottoms, False, False)
    return label_top_wave, label_bottom_wave


def calc_shape_wave_similarity(
    top_wave, bottom_wave, label_top_wave, label_bottom_wave
):
    r1 = Levenshtein.ratio(top_wave, label_top_wave)
    r2 = Levenshtein.ratio(bottom_wave, label_bottom_wave)
    sim = (r1 + r2) / 2
    return sim


def get_shape_wave_similarity(binary, label, debug=False):
    top_wave, bottom_wave, char_distribution = get_shape_wave(binary, debug)
    label_top_wave, label_bottom_wave = abstract_label_wave(label)

    r1 = Levenshtein.ratio(top_wave, label_top_wave)
    r2 = Levenshtein.ratio(bottom_wave, label_bottom_wave)
    sim = (r1 + r2) / 2

    if debug:
        print(top_wave)
        print(label_top_wave)
        print(r1)

        print(bottom_wave)
        print(label_bottom_wave)
        print(r2)
        print(sim)
    return sim, char_distribution
