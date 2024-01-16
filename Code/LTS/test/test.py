
import sys
sys.path.append('../')
import time
import ocr
        
def test_match_label():
    algo = ocr.OCR_ALGO.CANNY_TESSERACT
    ocr.load_ocr_module(algo)
    img_dir = "1.PNG"
    text = "properties & saved"
    expected_bounds = [0, 1100, 0, 2000]
    record = {'image': img_dir, 'text': text, 'bounds': expected_bounds}
    try:
        ocr.match_label(algo, img_dir, text, None, debug=False, stat=record, check_head_strategy="all",
                    revert=False, rank_detected_boxes=True)
    except Exception as e:
        print(e)
    print(record)
    return record

def test_match_text(bounds):
    algo = ocr.OCR_ALGO.CANNY_TESSERACT
    ocr.load_ocr_module(algo)
    img_dir = "1.PNG"
    text = "properties & saved"
    expected_bounds = bounds
    record = {'image': img_dir, 'text': text, 'bounds': expected_bounds}
    try:
        result = ocr.match_text(algo, img_dir, text, expected_bounds, None, debug=False, stat=record)
        print("record: ", record)
        print([str(item) for item in result])
    except Exception as e:
        print(e)
        
t1 = time.time()
record = test_match_label()
match_bounds = record['match_bounds']


test_match_text(match_bounds)

match_label_time = time.time() - t1
print("all: ", match_label_time)