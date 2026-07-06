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

# ---------------------------------------------------------------------
# PHASE 2: tf.data Pipeline & Feature Extraction
# ---------------------------------------------------------------------

def load_and_crop_image(image_path: tf.Tensor, ymin: tf.Tensor, xmin: tf.Tensor, 
                        ymax: tf.Tensor, xmax: tf.Tensor) -> tf.Tensor:
    """
    Loads an image from disk, crops it using bounding box coordinates, and applies 
    ResNet50 preprocessing.

    Designed to be mapped over a tf.data.Dataset. It decodes a PNG image, applies 
    `tf.image.crop_to_bounding_box`, resizes the crop to (224, 224), and zero-centers 
    the color channels using `preprocess_input`.

    Args:
        image_path (tf.Tensor): A string tensor containing the filepath to the image.
        ymin (tf.Tensor): An integer tensor representing the minimum Y coordinate.
        xmin (tf.Tensor): An integer tensor representing the minimum X coordinate.
        ymax (tf.Tensor): An integer tensor representing the maximum Y coordinate.
        xmax (tf.Tensor): An integer tensor representing the maximum X coordinate.
    """

    # Reading and Decoding the Image:
    image_raw= tf.io.read_file(image_path)
    image= tf.image.decode_png(image_raw, channels= 3)

    # Calculating Dimensions for the Crop:
    height= ymax - ymin
    width= xmax - xmin

    # Cropping a building out of the larger image:
    cropped_image= tf.image.crop_to_bounding_box(image, ymin, xmin, height, width)

    # Resizing The Cropped Image to ResNet50's Expected Input Size:
    resized_image= tf.image.resize(cropped_image, [224, 224])

    # Applying ResNet50's Pre-Processing:
    preprocessed_image= preprocess_input(resized_image)

    return preprocessed_image

def serialize_example(building_id: tf.Tensor, feature_vector: tf.Tensor, label: tf.Tensor) -> bytes:
    """
    Serializes a single building's extracted data into a tf.train.Example message.

    Args:
        building_id (tf.Tensor): A string tensor containing the unique building identifier.
        feature_vector (tf.Tensor): A float32 tensor of shape (2048,) representing the 
            extracted ResNet50 features.
        label (tf.Tensor): A string tensor containing the xBD damage classification label.

    Returns:
        bytes: A serialized tf.train.Example proto string ready to be written to a TFRecord.
    """

    # Helper Functions to Format data for TFRedords:
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value= value.numpy()
        return tf.train.Feature(bytes_list= tf.train.BytesList(value= [value]))

    def _float_feature(value):
        return tf.train.Feature(float_list= tf.train.FloatList(value= value))
    
    feature= {
        'building_id': _bytes_feature(building_id.numpy()),
        'label': _bytes_feature(label.numpy()),
        'feature_vector': _float_feature(feature_vector.numpy().flatten().tolist())
    }

    examplt_proto= tf.train.Example(features= tf.train.Features(feature= feature))
    return examplt_proto.SerializeToString()

if __name__ == '__main__':
    #a,b,c,d= extract_bbox_from_wkt('POLYGON ((-90.81544679490855 14.39086318334812, -90.81537467350067 14.39060467857134, -90.81584174451893 14.39043032647906, -90.81586635209965 14.39049581582557, -90.81593344431286 14.39048145754227, -90.81595559689623 14.39057367091926, -90.81587964155047 14.39059650626524, -90.81590706308843 14.39071123556855, -90.81544679490855 14.39086318334812))')
    #print(a, b, c, d)

    #build_metadata_csv()
    temp_str= serialize_example(building_id= tf.constant('buidling_101'),
                                feature_vector= tf.constant([0.23, 0.81, 0.17], dtype= tf.float32),
                                label= tf.constant('destroyed'))
    print(temp_str)