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

# Setting up Directory Paths:
PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR= os.path.join(PROJECT_ROOT, 'data', 'raw', 'train_data', 'labels')
SHOWCASE_DIR= os.path.join(PROJECT_ROOT, 'data', 'showcase')
MODEL_JSON_PATH= os.path.join(PROJECT_ROOT, 'models', 'xgboost_classifier.json')
OUTPUT_CSV_PATH= os.path.join(PROJECT_ROOT, 'data', 'processed', 'showcase.csv')

def load_pipeline():
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

if __name__ == '__main__':
    temp= extract_real_buidings('palu-tsunami_00000064_post_disaster.png')
    print(temp)
    print(len(temp))