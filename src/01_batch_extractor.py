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
from tqdm import tqdm

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

def run_extraction_pipeline() -> None:
    """
    Initializes the ResNet50 model, configures the tf.data stream, and executes 
    the feature extraction, writing the results to a TFRecord file.

    This function loads a frozen ResNet50 model (without the top classification layer), 
    reads the pre-generated `METADATA_CSV` to construct a high-performance `tf.data.Dataset`, 
    maps the cropping function across batches, and serializes the resulting feature 
    vectors to `TFRECORD_OUT`.

    Returns:
        None
    """

    print('[Phase 2] Initializing ResNet50 (frozen) on GPU...')

    # Loading ResNet50 without classification head:
    extractor_model= ResNet50(weights= 'imagenet',
                              include_top= False,
                              pooling= 'avg')
    
    # Loading Metadat:
    print('[Phase 2] Loading Metadata Stream...')
    building_ids, image_paths, ymins, xmins, ymaxs, xmaxs, labels,= [], [], [], [], [], [], []

    with open(METADATA_CSV, 'r') as f:
        reader= csv.DictReader(f)
        for row in reader:
            building_ids.append(row['building_id'].encode('utf-8'))
            image_paths.append(row['image_path'])
            ymins.append(int(row['ymin']))
            xmins.append(int(row['xmin']))
            ymaxs.append(int(row['ymax']))
            xmaxs.append(int(row['xmax']))
            labels.append(row['label'].encode('utf-8'))

    # Creating tf.data.Dataset:
    dataset= tf.data.Dataset.from_tensor_slices((
        image_paths, ymins, xmins, ymaxs, xmaxs, building_ids, labels
    ))

    # Applying Crop Fucntion to crop images:
    def map_fn(img_path, ymin, xmin, ymax, xmax, b_id, lbl):
        processed_img= load_and_crop_image(image_path= img_path,
                                           ymin= ymin,
                                           xmin= xmin,
                                           ymax= ymax,
                                           xmax= xmax)
        
        return processed_img, b_id, lbl
    
    batch_size= 32
    dataset= dataset.map(map_func= map_fn, num_parallel_calls= tf.data.AUTOTUNE)
    dataset= dataset.batch(batch_size= batch_size).prefetch(buffer_size= tf.data.AUTOTUNE)

    print(f'[Phase 2] Beginning feature extraction to {TFRECORD_OUT}...')
    with tf.io.TFRecordWriter(path= TFRECORD_OUT) as writer:
        for batch_images, batch_ids, batch_labels in tqdm(dataset, desc="Extracting Features"):
            
            # Forward Pass through ResNet50
            features= extractor_model(batch_images, training= False)

            # Serializing and Writing Each Image in the Batch:
            for i in range(len(features)):
                tf_example= serialize_example(building_id= batch_ids[i],
                                              feature_vector= features[i],
                                              label= batch_labels[i])
                writer.write(tf_example)
    
    print('[Phase 2] Extraction Complete.')


if __name__ == '__main__':
    if not os.path.exists(METADATA_CSV):
        build_metadata_csv()
    else:
        print('[Phase 1] metadata.csv already exists. Skipping rebuild.')
    
    run_extraction_pipeline()