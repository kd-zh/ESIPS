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
total_start_time = time.time()

ANDROID_DEVICES = [
    "Google Nexus 5",
    "Google_Nexus_4",
    "Google_Nexus_5X",
    "Nexus_6_API_25",
    "Nexus_6P_API_25",
    "Nexus_S_API",
    "Pixel_2_API_25",
    "Pixel_2_XL_API_25",
    "Pixel_3_API_25",
    # "Pixel_3_XL_API_25",
    "Pixel_3a_API_25",
    "Pixel_3a_XL_API_25",
    "Pixel_API_25",
    "Pixel_XL_API_25"
]

ANDROID_APPS = [
    "AcsNote_8.03",
    "appunti_scritto_0.0.16",
    "ASCII_Text_Symbols_1.0",
    "Bangla_Note_2.3",
    "Belt.io_1.0.1",
    "Catch_Notes_5.2.11",
    "Cek_Pulsa_Kuota_Semua_Operator_Seluler_1.2",
    "Character_Story_Planner_2_1.91",
    # "Clipboard_Contents_6.3.1.10",
    # "Clipboard_Manager_4",
    "ColorNote_Notepad_4.1.4",
    "Cute_Text_Photo_Maker_and_Editor_1.0",
    "Death_Note_3.1.5",
    "Diario_Segreto_2.8",
    # "Do_Note_2.1",
    # "Document_Manager_1.9",
    "Fancy_Text_Free_3.5",
    "Fast_Notepad_1.4.4",
    # "Floating_Stickies_2.1",
    "GenialWriting_1.38.0811",
    "GirlsDiary_1.9.5",
    "GNotes_1.8.3.8",
    "GO_Note_Widget_2.33",
    # "Gratitude_journal_-_Private_Diary_3.0.7",
    "Handrite_2.16",
    "Handy_Journal_3.1.3",
    "Hashnote_1.5.1",
    # "iPhone_Notifications_Lite_6.4",
    "JotterPad_12.10.3-pi",
    # "Just_Note_1.1.1",
    # "Khmer_Notes_1.0.3",
    "Kingsoft_Clip_1.0.1_1.5.1",
    "Mindjet_Maps_4.1",
    "Mio_Diario_2.0.19",
    "My_Diary_1.3.5",
    # "My_Diary_6.1",
    "MyBitCast_1.0.17.7621",
    "Native_Clipboard_4.7.2",
    # "NOTA_2.13",
    # "NoteLedge_Lite_1.4.1",
    "Notepad_(Blocco_note)_2.3.8",
    # "Notepad_Plus_-_To-Do_&_Diary_1.2.0",
    "notePad+_3.2.10",
    # "Notes_(Beta)_1.07",
    # "Notes_Plus_1.0.0",
    "Notes_with_Password_0.1",
    "Notif_0.7.1",
    "NTW_Lite!_1.89b",
    "OCR_Scanner_-_Text_to_Speech,_Voice_to_Text_1.4",
    "Office_365_Admin_3.5.0.0",
    "Office_Documents_Viewer_(Free)_1.26.19",
    "OI_Notepad_1.3",
    "Open_Note_1.2.2016",
    "Open365_2.0.4",
    # "Paper_Formats_v7",
    # "PostStatut_5",
    "Quick_Notes_1.0.2",
    "Quick_Notes_1.3.0",
    "Quickoffice_-_Google_Apps_6.5.1.12",
    # "QuickThoughts_2.17.6",
    "Quotes_Kingdom_1.0",
    # "QwickNote_1.0",
    # "Read_Out_1.0.0",
    # "Safe_Notes_1.7",
    "SC_2.0",
    "Schedule_Deluxe_3.7.2",
    # "Secret_Notes2_1.2",
    "Secure_Notes_1.4.4",
    # "Simple_Notepad_1.8.8",
    "SimpleNotes_3.6.1",
    "Speechnotes_1.69",
    "Status_New_3.9",
    "Sync_Notes_-_Notepad_5.0",
    "Tamil_Keyboard_1.6.2",
    # "Tasks_&_Notes_11.7.11",
    # "Texpand_1.8.7",
    "Text_Scanner_(Scan_Computer)_-_Voice_Read_Pro",
    # "Text_Viewer_0.1.11",
    # "TextWarrior_0.93",
    # "To_Do_Reminder_2.68.50",
    # "To-Do_List_Widget_2.0",
    "Todoist_15.0.3",
    "Turbo_Editor_2.4",
    # "Txt_Reader_e_Writer_1.9",
    "Unicode_-_Bijoy_Converter_2.0.0",
    "Voice_to_Text_1.0.0",
    "WeNote_1.89",
    "WhatTheFont_1.1.1",
    "Write_Urdu_On_Photo_1.0.5",
    "Writeaday_2.7.0",
    "Writeometer_1.9.1",
    # "Writer_1.1",
    # "Writer_Plus_1.46"
]

# Make a directory if it doesn't exist
os.makedirs('ip', exist_ok=True)


# Loop through all
for application in ANDROID_APPS:
    for device in ANDROID_DEVICES:
        
        folder = str(application)
        input_path_img = f"../datAndroidDataset/{device}/{application}/Screenshot_0/Screenshot_0.png"
        input_path_text = f"../datAndroidDataset/Google Nexus 5/{application}/Screenshot_0/text.txt"

        # consider keeping track of time taken
        # consider also the # of items that need to be found
        file_name = str(application) + '_' + str(device)
        
        # Read in the given image
        org = cv2.imread(input_path_img)
        height, width = org.shape[:2]

        t1 = time.time()
        try:
            with open(input_path_text, 'r') as file:
                find_widget_text = file.read()
                find_widget_text = ast.literal_eval(find_widget_text)
        except FileNotFoundError:
            print(f"File not found: {find_widget_text}")
        except Exception as e:
            print(f"An error occurred: {e}")
            

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

        # 2) Detect text within those regions with LTS
        list_widgets_dicts=[]

        LTS_full = []
        # TODO - swap this around hahaha
        # Loop through all expected widgets and the components found, trying to find a match
        for text in find_widget_text:
            for item in components:
                item['bounding_box'] = [item["column_min"], item["row_min"], item["width"],item["height"]]
                record = lts_find_text(input_path_img, item['bounding_box'], text)
                # print(record)
                if record.get("label_detected") and record["label_detected"]:
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
            
        data = [device, application, total_time]
        # Path to the Excel file
        excel_filename = "resultsAndroidThesis.xlsx"
        excel_path = os.path.join("..", excel_filename)
        # Check if the file exists
        if os.path.exists(excel_path):
            # Read the existing Excel file
            dfExisting = pd.read_excel(excel_path)
        else:
            # Create a new DataFrame if the file does not exist
            dfExisting = pd.DataFrame(columns=['Device', 'Application', 'Total Time'])
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
        
total_end_time = time.time()
print("Total time: " + str(total_end_time-total_start_time))
