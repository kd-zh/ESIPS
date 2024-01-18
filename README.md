# Resolution-Independent Widget Detection on Mobile Screens

Code from my [ESIPS](https://www.sydney.edu.au/engineering/study/scholarships/engineering-sydney-industry-placement-scholarship.html) thesis. 

**Aim:** Correctly identify widgets commonly present during onboarding processes on a mobile application

**Motivating problem:** Simulate different users undergoing onboarding across multiple mobile devices

## Acknowledgements 

**Changes/adaptations were made to the following code and datasets:**

UIED taken from:
* [https://github.com/MulongXie/UIED/releases/tag/v2.3](https://github.com/MulongXie/UIED/releases/tag/v2.3) (Apache License 2.0)
* Any changes made to UIED were to get it working, and the code was otherwise left unchanged.

LTS taken from:
* [https://doi.org/10.6084/m9.figshare.19722013.v1](https://doi.org/10.6084/m9.figshare.19722013.v1) (CC BY 4.0)
* Any changes made to LTS were to get it working, and the code was otherwise left unchanged. Dowl and Rico were removed.
  
Example dataset taken from DatAndroid:

* [https://zenodo.org/records/4605429](https://zenodo.org/records/4605429) (CC BY 4.0)
* Addition of widget list added

Thank you!

## Set Up

The Python version used is 3.11.7.

The required modules to be installed are:
* opencv-python (4.8.1.78)
* pandas (2.1.3)
* Levenshtein (0.23.0)
* Pillow (10.1.0)
* pytesseract (0.3.10)
* tesserocr (2.6.2)
* easyocr (1.7.1)

Tesserocr may give an error if not installed with ``pip install --no-binary :all: tesserocr``.

A lower version of the above packages may also be fine.

## Run Example

1. Navigate to the code folder

``cd code``

2. Run runExample.py

``python3 runExample.py``
