"""
01_batch_extractor.py
Disaster Triage Pipeline: Post-Disaster Feature Extraction

This script runs a two-phase extraction process for the xBD challenge dataset:
Phase 1: Parses post-disaster JSON labels to extract bounding boxes and damage tags, 
         saving a master metadata.csv.
Phase 2: Streams the metadata into a tf.data pipeline, crops the buildings, 
         extracts 2048-d feature vectors using a frozen ResNet50, and serializes 
         the output to a TFRecord file.
"""

# Imports:
import os
import glob
import json
import csv
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Configuration and Paths:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw", "train_data")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")


IMAGES_DIR= os.path.join(RAW_DATA_DIR, 'images')
LABELS_DIR= os.path.join(RAW_DATA_DIR, 'labels')

METADATA_CSV= os.path.join(PROCESSED_DIR, 'metadata.csv')
TFRECORD_OUT= os.path.join(PROCESSED_DIR, 'features.tfrecord')

# Ensuring Processed Directory Exists:
os.makedirs(PROCESSED_DIR, exist_ok= True)

# ---------------------------------------------------------------------
# PHASE 1: METADATA PARSING & CSV GENERATION
# ---------------------------------------------------------------------


def extract_bbox_from_wkt(wkt_polygon: str) -> tuple[int, int, int, int]:
    """
    Converts an xBD Well-Known Text (WKT) polygon string into a standard bounding box.

    Args:
        wkt_polygon (str): The polygon string from the xBD JSON (e.g., 'POLYGON ((x1 y1, ...))').

    Returns:
        tuple[int, int, int, int]: A tuple containing the bounding box coordinates 
        in the format (ymin, xmin, ymax, xmax).
    """

    # Cleaning the wkt_polygon string and Splitting into Coordinates:
    wkt_clean= wkt_polygon.replace('POLYGON ((', '').replace('))', '')
    points= wkt_clean.split(', ')
    x_coords= []
    y_coords= []

    for point in points:
        x, y= point.split(' ')
        x_coords.append(float(x))
        y_coords.append(float(y))

    # Bounding Box:
    xmin, xmax= int(min(x_coords)), int(max(x_coords))
    ymin, ymax= int(min(y_coords)), int(max(y_coords))

    return ymin, xmin, ymax, xmax

def build_metadata_csv() -> None:
    """
    Scans the predefined labels directory for post-disaster JSON files, extracts 
    building coordinates and damage labels, and generates a unified CSV.

    This function reads from `LABELS_DIR` and writes to `METADATA_CSV`. It strictly 
    filters for files ending in '_post_disaster.json' and skips records where the 
    corresponding image file does not exist.

    Returns:
        None
    """

    print('Scanning for Post Disaster JSON Files...')
    json_files= glob.glob(os.path.join(LABELS_DIR, '*_post_disaster.json'))

    records= []

    for file_path in json_files:
        with open(file= file_path, mode= 'r') as f:
            data= json.load(f)
        
        image_name= data['metadata']['img_name']
        image_path= os.path.join(IMAGES_DIR, image_name)

        # Skipping if image does not exist:
        if not os.path.exists(image_path):
            continue
        
        # Parsing Every Building Polygons from the image:
        for feature in data['features']['xy']:
            properties= feature['properties']

            # Extracting Damage lebel from sub-type:
            damage_lebel= properties.get('subtype', 'un-classified')
            uid= properties.get('uid', 'unknown')

            building_id= f"{image_name.replace('.png','')}_{uid}"
            wkt_polygon= feature['wkt']

            ymin, xmin, ymax, xmax= extract_bbox_from_wkt(wkt_polygon= wkt_polygon)

            records.append({
                'building_id': building_id,
                'image_path': image_path,
                'ymin': ymin,
                'xmin': xmin,
                'ymax': ymax,
                'xmax': xmax,
                'label': damage_lebel
            })

    print(f"[Phase 1] found {len(records)} Post-Disaster Buildings.")

    # Writing to CSV:
    with open(METADATA_CSV, 'w', newline= '') as f:
        writer= csv.DictWriter(f= f,
                               fieldnames= ['building_id', 'image_path', 'ymin', 'xmin', 'ymax', 'xmax', 'label'])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"[Phase 1] Metadata Saved to {METADATA_CSV}")

if __name__ == '__main__':
    #a,b,c,d= extract_bbox_from_wkt('POLYGON ((-90.81544679490855 14.39086318334812, -90.81537467350067 14.39060467857134, -90.81584174451893 14.39043032647906, -90.81586635209965 14.39049581582557, -90.81593344431286 14.39048145754227, -90.81595559689623 14.39057367091926, -90.81587964155047 14.39059650626524, -90.81590706308843 14.39071123556855, -90.81544679490855 14.39086318334812))')
    #print(a, b, c, d)

    build_metadata_csv()