import json

import numpy
# import yaml
import cv2
import math
import os
import numpy as np

def get_dbg_output_folder(image_file):
    filename_and_ext = os.path.basename(image_file)
    filename = os.path.splitext(filename_and_ext)[0]

    foldername = os.path.basename(os.path.dirname(image_file))

    project_dir = os.path.dirname(os.path.dirname(__file__))
    result = os.path.join(project_dir, "output", foldername, filename)

    if not os.path.exists(result):
        os.makedirs(result)
    return result


def get_dbg_output_file(image_file, dbg_file):
    folder = get_dbg_output_folder(image_file)
    result = os.path.join(folder, dbg_file)
    
    return result


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]  

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_show_imagefile(name, img_file):
    img = cv2.imread(img_file, 1)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut_img(img, x, y, offset):
    cutimg = img[y:y + offset, x:x + offset]
    return cutimg


def get_keys(d, value):
    return [k for k, v in d.items() if v == value]

def list_to_str(a_list):
    s = ''
    for item in a_list:
        if len(s) > 0:
            s = s + ','
        s = s + str(item)
    return s


def show_contours(title, img, contours, color=(0, 255, 0)):
    copy = img.copy()
    cv2.drawContours(copy, contours, -1, color, 3)
    cv2.imshow(title, copy)
    cv2.waitKey(0)


def save_contours(file, img, contours, color=(0, 255, 0)):
    copy = img.copy()
    cv2.drawContours(copy, contours, -1, color, 1)
    cv2.imwrite(file, copy)


def show_rectangles(title, img, rectangles):
    copy = img.copy()
    for x, y, w, h in rectangles:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imshow(title, copy)
    cv2.waitKey(0)

def save_rectangles(img, rectangles, result_file):
    copy = img.copy()
    for x, y, w, h in rectangles:
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imwrite(result_file, copy)

def get_center_point(img, x, y, z):
    center_x = x + z / 2
    center_y = y + z / 2
    print(center_x, center_y)
    return center_x, center_y


def draw_circle(img, positions):
    for p in positions:
        cv2.circle(img, p, 1, (0, 255, 0), 10)
    img = resize(img, width=800)
    cv2.imshow('img', img)
    cv2.waitKey(0)



def dict_to_list(rico_bounds_dict: dict) -> list:
    rico_bounds = list()
    for k, v in rico_bounds_dict.items():
        if v is True:
            rico_bounds.append(k)
    return rico_bounds


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_json(path, json_data):
    output = json.dumps(json_data, cls=JsonEncoder, indent=4)
    jsonfile = open(path, 'w')
    jsonfile.write(output)
    jsonfile.close()


def get_file_id(path):
    file_name = os.path.basename(path)
    name_no_ext = os.path.splitext(file_name)[0]
    id = int(name_no_ext)
    return  id


def collect_files(dir, exts=['.jpg', '.png'], max_num=-1):
    paths = []
    count = 0

    for root, dirs, files in os.walk(dir):
        if max_num>0 and count>max_num:
            break

        for file in files:
            if max_num > 0 and count > max_num:
                break

            p = file.rfind('.')
            ext = file[p:]
            if ext not in exts:
                continue

            path = os.path.join(root, file)
            paths.append(path)
            count += 1

    return paths