# Imports
import os
import glob
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import cv2
from tqdm import tqdm

# Setting up Directory Paths:
PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR= os.path.join(PROJECT_ROOT, 'data', 'raw', 'train_data', 'labels')
SHOWCASE_DIR= os.path.join(PROJECT_ROOT, 'data', 'showcase')
MODEL_JSON_PATH= os.path.join(PROJECT_ROOT, 'models', 'xgboost_classifier.json')
OUTPUT_CSV_PATH= os.path.join(PROJECT_ROOT, 'data', 'processed', 'showcase.csv')

def load_pipelines():
    """
    Initializes and loads the multimodal machine learning pipeline into memory.

    This function instantiates the frozen ResNet50 deep learning model for 
    feature extraction and deserializes the pre-trained XGBoost model 
    for triage classification.

    Returns:
        tuple: A tuple containing:
            - resnet (tf.keras.Model): The loaded ResNet50 feature extractor.
            - xgb_model (xgboost.XGBClassifier): The loaded decision tree classifier.
    """

    # Loading ResNet50:
    print('Loading Frozen ResNet50 Feature Extractor...')
    resnet= tf.keras.applications.ResNet50(
        include_top= False,
        weights= 'imagenet',
        pooling= 'avg'
    )

    # Loading XGBoost Classifier:
    xgb_model= xgb.XGBClassifier()
    xgb_model.load_model(fname= MODEL_JSON_PATH)

    return resnet, xgb_model

def extract_real_buidings(image_name):
    """
    Parses the xBD JSON label file to extract true building bounding boxes.

    This function locates the corresponding JSON label file for a given image,
    extracts the WKT (Well-Known Text) polygon coordinates for every structure,
    and calculates a Cartesian bounding box (ymin, xmin, ymax, xmax).

    Args:
        image_name (str): The filename of the image (e.g., 'hurricane-michael_001.png').

    Returns:
        list of dicts: A list containing building metadata and coordinates. 
                       Format: [{'uid': str, 'box': (ymin, xmin, ymax, xmax)}, ...]
    """

    # Image Path fromImage Name:
    json_name= image_name.replace('.png', '.json').replace('.jpg', '.json')
    json_path= os.path.join(LABEL_DIR, json_name)

    if not os.path.exists(json_path):
        return []
    
    buildings= []
    with open(json_path, 'r') as f:
        data= json.load(f)

    # Getting Polygon Features from JSON File:
    features= data.get('features', {}).get('xy', [])

    # Getting Bounding Box Coordinates from features:
    for feature in features:
        uid= feature.get('properties', {}).get('uid', 'unknown')
        wkt_string= feature.get('wkt', '')

        # Getting POLYGON numbers from wkt. Format: POLYGON ((x1 y1, x2 y2, x3 y3...)):
        match= re.search(r'\(\((.*?)\)\)', wkt_string)
        if not match:
            continue

        coords_string= match.group(1)
        points= coords_string.split(',')

        x_coords= []
        y_coords= []

        for point in points:
            coords= point.strip().split(' ')
            if len(coords) == 2:
                x_coords.append(float(coords[0]))
                y_coords.append(float(coords[1]))

        # Calculating Bounding Box from the Coordinates:
        xmin, xmax=  int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))

        # ensuring Coordinates are Mathematically Valid (Preventing 0-pixel Crops):
        if xmax > xmin and ymax > ymin:
            buildings.append({
                'uid': uid,
                'box': (ymin, xmin, ymax, xmax)
            })
        
    return buildings

def showcase_csv_prep():
    """
    Orchestrates the end-to-end data pipeline to generate the UI deployment payload.

    Iterates through all Golden Sample images in the showcase directory, extracts
    ground-truth building crops, processes them through the ResNet50/XGBoost 
    classification pipeline, and aggregates the analytical results into a single,
    lightweight CSV file for the Streamlit front-end.
    """

    # Loading ResNet50 and XGBoost Classifier: 
    resnet, xgb_model= load_pipelines()

    # Loading Showcase Image Paths:
    image_paths= glob.glob(os.path.join(SHOWCASE_DIR, '*.png')) + \
                 glob.glob(os.path.join(SHOWCASE_DIR, '*.jpg'))

    if not image_paths:
        print(f'Error: No images found in {SHOWCASE_DIR}. Run 03_showcase_curation.py first.')

    print(f'Found {len(image_paths)} Golden Samples. Starting CSV generation...')
    csv_data= []

    # Reading Images, Cropping Buildings, Generating Features from Buidlings and Passing Features through Classifier for Generating Prediction, appending in CSV:
    for img_path in tqdm(image_paths):
        image_name= os.path.basename(img_path)
        img= cv2.imread(img_path)

        if img is None:
            continue

        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pulling Building Coordinates in the Image:
        buildings= extract_real_buidings(image_name= image_name)

        # Processing all found buidlings:
        for  bldg in buildings:
            uid= bldg['uid']
            ymin, xmin, ymax, xmax= bldg['box']

            # Cropping Building from Image:
            crop= img[ymin:ymax, xmin:xmax]

            # Pre-Processing Cropped Building for ResNet50:
            crop_resized= cv2.resize(crop, (224,224))
            crop_expanded= np.expand_dims(crop_resized, axis= 0)
            crop_preprocessed= tf.keras.applications.resnet50.preprocess_input(crop_expanded.astype(np.float32))

            # Extracting Feature Vector from Building using Frozen ResNet50:
            feature_vector= resnet.predict(crop_preprocessed, verbose= 0)

            # Predicting Building's Damage Level using XGBoost Classifier:
            probabilities= xgb_model.predict_proba(feature_vector)[0]

            # Calculating Building's Footprint Area:
            footprint_area= (ymax- ymin) * (xmax - xmin)

            # Appending Buidling Stats and Predicted Damage Level to CSV:
            csv_data.append({
                'image_name': image_name,
                'building_uid': uid,
                'ymin': ymin,
                'xmin': xmin,
                'ymax': ymax,
                'xmax': xmax,
                'footprint_sq_px': footprint_area,
                'prob_no_damage': round(float(probabilities[0]), 4),
                'prob_minor': round(float(probabilities[1]), 4),
                'prob_major': round(float(probabilities[2]), 4),
                'prob_destroyed': round(float(probabilities[3]), 4)
            })

    
    # Final Dataframe to save to CSV file:
    df= pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV_PATH, index= False)

    print("\n" + "="*50)
    print(f"SUCCESS: Final CSV Saved with {len(df)} total structures.")
    print(f"File Saved at: {OUTPUT_CSV_PATH}")
    print("="*50 + "\n")

    
if __name__ == '__main__':
    # temp= extract_real_buidings('palu-tsunami_00000064_post_disaster.png')
    # print(temp)
    # print(len(temp))
    showcase_csv_prep()