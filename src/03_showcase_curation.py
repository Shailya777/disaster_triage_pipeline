# Imports
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Suppressing Tensorflow Warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Defining Paths:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFRECORD_PATH= os.path.join(PROJECT_DIR, 'data', 'processed', 'features.tfrecord')
MODEL_JSON_PATH= os.path.join(PROJECT_DIR, 'models', 'xgboost_classifier.json')
SOURCE_IMG_DIR= os.path.join(PROJECT_DIR, 'data', 'raw', 'train_data', 'images')
SHOWCASE_DIR= os.path.join(PROJECT_DIR, 'data', 'showcase')

# making sure output directory exists:
os.makedirs(SHOWCASE_DIR, exist_ok= True)

# ---------------------------------------------------------------------
# Data Loading Helper Functions
# ---------------------------------------------------------------------

# Fuction to Parse a Single tf.train.Example element:
def parse_tfr_element(element):
    """
    Parses a single tf.train.Example protocol buffer.
    """
    data= {
        'building_id': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'feature_vector': tf.io.FixedLenFeature([2048], tf.float32)
    }

    content= tf.io.parse_single_example(element, data)
    return content['feature_vector'], content['label'], content['building_id']

# Function to Load TFRecords Dataset and Convert it into Numpy Arrays:
def load_dataset():
    """
    Loads the TFRecord dataset and converts it to numpy arrays.
    """
    print('Unpacking TFRecords Dataset...')
    dataset= tf.data.TFRecordDataset(TFRECORD_PATH)
    dataset= dataset.map(parse_tfr_element)

    X_list, y_list, id_list= [], [], []

    for features, label, b_id in dataset:
        X_list.append(features.numpy())
        y_list.append(label.numpy().decode('utf-8'))
        id_list.append(b_id.numpy().decode('utf-8'))
    
    X= np.array(X_list)
    y= np.array(y_list)
    ids= np.array(id_list)

    print(f'Successfully Loaded {X.shape[0]} Records.')
    print(f'Feature Matrix Shape: {X.shape}')

    return X, y, ids