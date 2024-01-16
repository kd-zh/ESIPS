"""
ele:min-grad: gradient threshold to produce binary map         
ele:ffl-block: fill-flood threshold
ele:min-ele-area: minimum area for selected elements 
ele:merge-contained-ele: if True, merge elements contained in others
text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

Tips:
1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
"""

import os
from os.path import join as pjoin
import cv2
import detect_compo.ip_region_proposal as ip

# Do not run if this is being imported as module!
if __name__ == '__main__':
    key_params = {'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 500, 'merge-contained-ele': False,
                  'max-word-inline-gap': 6, 'max-line-gap': 1}

    # Set input image path
    input_path_img = 'data/input/domain1.PNG'
    output_root = 'data/output'

    # To do - don't want to resize height
    org = cv2.imread(input_path_img)
    height, width = org.shape[:2]
    
    os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
    ip.compo_detection(input_path_img, output_root, key_params,
                        classifier=None,  show=True)
    print("run_single.py done!")