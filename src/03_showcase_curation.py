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
