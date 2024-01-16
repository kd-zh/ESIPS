import os
import sys
import cv2
import time
import json
import ast
from PIL import Image, ImageDraw, ImageFont
from os.path import join as pjoin
import pandas as pd
sys.path.append('UIED-2.3')
import detect_compo.ip_region_proposal as ip

sys.path.append('LTS/code')
import ocr

def bounding_box_area(bounding_box):
    """Calculate the area of a bounding box."""
    return bounding_box[2] * bounding_box[3]

def contains_box(box1, box2):
    """Check if box 1 contains box 2."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2))

def lts_find_text(img_input_path, expected_bounds, text):
    """
    Locate and identify specified text within given bounds in an image.

    Parameters:
    img_input_path  (string):   Path to the image
    expected_bounds (number[]): [x1, y1, x2, y2]
    text            (string):   Text being matched on, or leave empty to simply perform OCR within those spaces 

    Returns:
    record (dict): 
        image (string): As above
        text  (string): As above
        bounds (number[]): [x1, y1, width, height]
        detect_time:
        detect_num:
        rank_time:
        match_rank:
        match_bounds:
        recognize_timex:
        label_detected (bool): True is the text was detected
        total_time:
    'detect_time': 0.020453214645385742, 
    'detect_num': 2, 
    'rank_time': 5.793571472167969e-05, 
    'match_rank': 0, 
    'match_bounds': [400, 1263, 160, 47], 
    'recognize_time': 0.052001953125, 
    'label_detected': True, 
    'total_time': 0.07251310348510742}
    """
    algo = ocr.OCR_ALGO.CANNY_TESSERACT
    ocr.load_ocr_module(algo)
    img_dir = input_path_img
    
    record = {'image': img_dir, 'text': text, 'bounds': expected_bounds}
    try:
        ocr.match_label(algo, img_dir, expected_bounds, text, None, debug=False, stat=record, check_head_strategy="all",
                    revert=False, rank_detected_boxes=True)
    except Exception as e:
        print(e)
    return record

# Keep track of the total time


# Make a directory if it doesn't exist
os.makedirs('ip', exist_ok=True)

# List all sizes being tested
# screen_sizes = [ "iPadPro.webp", 
#                 "iPhone13Mini.PNG", 
#                 "iPhone13Pro.webp", 
#                 "iPhoneSE.PNG"]

# Loop through folder names
for i in range(1, 72):
    if i > 30:
        screen_sizes = ["iPhone13Mini.PNG",  
                "iPhoneSE.PNG"]
    else:
        screen_sizes = [ "iPadPro.webp", 
                "iPhone13Mini.PNG", 
                "iPhone13Pro.webp", 
                "iPhoneSE.PNG"]
    for screen in screen_sizes:
        t1 = time.time()
        folder = str(i)
        folder_path = os.path.join('Dataset', folder)
        input_path_img = os.path.join(folder_path, str(screen))
        input_path_text = os.path.join(folder_path, 'text.txt')
        print(input_path_img,input_path_text)
        # consider keeping track of time taken
        # consider also the # of items that need to be found
        file_name = str(i) + '_' + str(screen.split('.')[0])
        
        try:
            with open(input_path_text, 'r') as file:
                find_widget_text = file.read()
                find_widget_text = ast.literal_eval(find_widget_text)
        except FileNotFoundError:
            print(f"File not found: {find_widget_text}")
        except Exception as e:
            print(f"An error occurred: {e}")
            

        # Read in the given image
        org = cv2.imread(input_path_img)
        
        height, width = org.shape[:2]

        # 1) Detect widgets with UIED 2.3
        key_params = {'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 500, 'merge-contained-ele': False,
                        'max-word-inline-gap': 6, 'max-line-gap': 1}
        uicompos = ip.compo_detection(input_path_img, '', key_params, file_name,
                            classifier=None, show=True)
        print("UIED 2.3 is finished!")
            
        # Open the JSON file produced by UIED and extract the components
        with open('compo.json', 'r') as file:
            dict = json.load(file)
        components = dict["compos"]

        # The first item is always the size of the screenshot (TODO - create a check for this?)
        # popped = components.pop(0) 

        # 2) Detect text within those regions with LTS
        list_widgets_dicts=[]

        LTS_full = []
        # TODO - swap this around hahaha
        # Loop through all expected widgets and the components found, trying to find a match
        for text in find_widget_text:
            for item in components:
                item['bounding_box'] = [item["column_min"], item["row_min"], item["width"],item["height"]]
                record = lts_find_text(input_path_img, item['bounding_box'], text)
                    
                if record["label_detected"]:
                    if item["width"] == width and item["height"] == height:
                        LTS_full.append({"bounding_box": record["match_bounds"], "text": text})
                    # The minimum x and y must be added due to the cropping
                    record["match_bounds"][0] += item["column_min"]
                    record["match_bounds"][1] += item["row_min"]
                    widget_dict = {
                        'bounding_box': item['bounding_box'],
                        'bounding_box_text': record['match_bounds'],
                        'text': record['text']
                    }
                    # If we haven't already put the entry in
                    if not list_widgets_dicts:
                        
                        list_widgets_dicts.append(widget_dict)
                    if widget_dict not in list_widgets_dicts:
                        # If we have discovered the same text, make sure we only keep the one with the smaller bounding box
                        for index, widget in enumerate(list_widgets_dicts):
                            if contains_box(widget['bounding_box_text'], widget_dict['bounding_box_text']):
                                list_widgets_dicts[index] = widget_dict
                            else:
                                list_widgets_dicts.append(widget_dict)

        # Group widgets by text
        from collections import defaultdict
        grouped_widgets = defaultdict(list)
        for widget in list_widgets_dicts:
            grouped_widgets[widget['text']].append(widget)
            

        # # Identify problems with text
        # # Load the image
        # image1 = Image.open(input_path_img)

        # # Create a drawing object
        # draw1 = ImageDraw.Draw(image1)

        # for item in list_widgets_dicts:
        #     # Define the coordinates of the box (x1, y1, x2, y2)
        #     box = (item["bounding_box_text"][0], item["bounding_box_text"][1], item["bounding_box_text"][0]+item["bounding_box_text"][2], item["bounding_box_text"][1]+item["bounding_box_text"][3])
        #     # Draw the box
        #     draw1.rectangle(box, outline="blue", width=10)
            
        #     box1 = (item["bounding_box"][0], item["bounding_box"][1], item["bounding_box"][0]+item["bounding_box"][2], item["bounding_box"][1]+item["bounding_box"][3])
        #     # Draw the box
        #     draw1.rectangle(box1, outline="red", width=10)
            
        #     # Define the text and its position to fit near the top-left
        #     text = item["text"]
        #     text_x = item["bounding_box_text"][0] + 15
        #     text_y = item["bounding_box_text"][1] + 15 

        #     # Draw the text
        #     draw1.text((text_x, text_y), text, fill="blue")

        # # Save the image
        # image1.save('text/' + input_path_img) 

        # print(list_widgets_dicts)
        # raise ValueError()
        
        def find_closest_y1(bounding_boxes, target_y1):
            """
            Finds the bounding box with the y1 value closest to the given target y1 value.

            Parameters:
            bounding_boxes (list): A list of dictionaries, each containing a 'bounding_box' key with [x1, y1, x2, y2] values.
            target_y1 (int): The target y1 value to which the closest y1 is found.

            Returns:
            closest_box (dict): The bounding box whose y1 value is closest to the target y1.
            """

            # Initialize the closest y1 difference and the corresponding bounding box
            closest_diff = float('inf')
            closest_box = None

            # Iterate through each bounding box to find the y1 closest to target_y1
            for box in bounding_boxes:
                # Extract the y1 value
                y1 = box['bounding_box'][1]

                # Calculate the difference from the target y1
                diff = abs(y1 - target_y1)

                # Update closest_diff and closest_box if the current y1 is closer
                if diff < closest_diff:
                    closest_diff = diff
                    closest_box = box

            return closest_box

        # Find the largest non-containing bounding box for each text group
        list_widgets_result = []
        for text, group in grouped_widgets.items():
            # Sort the group by descending area of bounding boxes
            group.sort(key=lambda widget: bounding_box_area(widget['bounding_box']), reverse=True)

            for widget in group:
                list_contained_widgets = []
                # Ensure the box is the smallest non-containing box in the group of text
                if not any(contains_box(widget['bounding_box'], other_widget['bounding_box']) 
                    # Do not compare a bounding box with itself
                    for other_widget in group if other_widget != widget):
                        # Add a check to see if a smaller widget exists below the given text, within the given bounds
                        # Loop through all the boxes found
                        for item in components:
                            # Is the new widget contained within the current widget?
                            # widget is the text, item is the component
                            if contains_box(widget['bounding_box'], item['bounding_box']):
                                # print(widget['bounding_box'], item['bounding_box'])
                                # print("x1", widget['bounding_box_text'][0], item['bounding_box'][0], widget['bounding_box_text'][0] >= item['bounding_box'][0])
                                # print("x2", widget['bounding_box_text'][2]+widget['bounding_box'][0], item['bounding_box'][2]+item['bounding_box'][0], widget['bounding_box_text'][2]+widget['bounding_box'][0]  <= item['bounding_box'][2]+item['bounding_box'][0])
                                    
                                # If so, does the widget text sit within the x coordinates of the item, and above the y coordinates of the item/widget?
                                if widget['bounding_box_text'][0] >= item['bounding_box'][0] and widget['bounding_box_text'][2]+widget['bounding_box'][0]  <= item['bounding_box'][2]+item['bounding_box'][0] and widget['bounding_box'][3]+widget['bounding_box_text'][1] >= item['bounding_box'][3]+item['bounding_box'][1]: 
                                    
                                    list_contained_widgets.append(item)
                        if list_contained_widgets:
                            actual_widget = find_closest_y1(list_contained_widgets, widget['bounding_box_text'][1])
                            list_widgets_result.append({'bounding_box': actual_widget['bounding_box'], 'text': text})
                            break
                                # if yes, create a new widget since the text doesn't sit within the box
                        else:  
                            # Otherwise simply append
                            list_widgets_result.append(widget)
                            break  # Break after adding the first valid widget           


        # Loop to replace entire screen captures with only the text found
        for item in list_widgets_result:
            for lts_item in LTS_full:
                # Check if the text matches and bounding box of list_widgets_result item is [0, 0, width, height]
                if item['text'] == lts_item['text'] and item['bounding_box'][0] == 0 and item['bounding_box'][1] == 0 and item['bounding_box'][2] == width and item['bounding_box'][3] == height:
                    item['bounding_box'] = lts_item['bounding_box']
                        
        # 3) Display the results in a visually friendly way
        # Load the image
        image = Image.open(input_path_img)
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)

        for item in list_widgets_result:
            # Define the coordinates of the box (x1, y1, x2, y2)
            box = (item["bounding_box"][0], item["bounding_box"][1], item["bounding_box"][0]+item["bounding_box"][2], item["bounding_box"][1]+item["bounding_box"][3])
            # Draw the box
            draw.rectangle(box, outline="red", width=10)
            
            # Define the text and its position to fit near the top-left
            text = item["text"]
            text_x = item["bounding_box"][0] + 15
            text_y = item["bounding_box"][1] + 15 

            # Draw the text
            draw.text((text_x, text_y), text, fill="red")

        # Save the image
        image.save('ip/' + file_name + '.png') 

        # Print out the total time taken
        total_time = time.time() - t1
        print("Total time taken: ", total_time)
        
        list_widgets_result.append({"total_time": total_time})

        with open('ip/' + file_name + ".txt", "w") as file:
            file.write(str(list_widgets_result))
        
        data = [screen, folder, total_time]
        # Path to the Excel file
        excel_filename = "resultsAppleThesis.xlsx"
        excel_path = os.path.join("..", excel_filename)
        # Check if the file exists
        if os.path.exists(excel_path):
            # Read the existing Excel file
            dfExisting = pd.read_excel(excel_path)
        else:
            # Create a new DataFrame if the file does not exist
            dfExisting = pd.DataFrame(columns=['Screen', 'Folder', 'Total Time'])
        # Append the new data
        dfExisting.loc[len(dfExisting)] = data
        # Write the DataFrame back to Excel
        dfExisting.to_excel(excel_path, index=False)
        print(f"Excel file updated at {excel_path}")

        # def test_match_text(bounds):
        #     algo = ocr.OCR_ALGO.CANNY_TESSERACT
        #     ocr.load_ocr_module(algo)
        #     img_dir = "1.PNG"
        #     text = "properties & saved"
        #     expected_bounds = bounds
        #     record = {'image': img_dir, 'text': text, 'bounds': expected_bounds}
        #     try:
        #         result = ocr.match_text(algo, img_dir, text, expected_bounds, None, debug=False, stat=record)
        #         print("record: ", record)
        #         print([str(item) for item in result])
        #     except Exception as e:
        #         print(e)
        # match_bounds = record['match_bounds']
        # test_match_text(match_bounds)