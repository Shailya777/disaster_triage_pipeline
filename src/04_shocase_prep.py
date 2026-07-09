# Imports
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import cv2

# Setting up Directory Paths:
PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE_DIR= os.path.join(PROJECT_ROOT, 'data', 'showcase')
MODEL_JSON_PATH= os.path.join(PROJECT_ROOT, 'models', 'xgboost_classifier.json')
OUTPUT_CSV_PATH= os.path.join(PROJECT_ROOT, 'data', 'processed', 'showcase.csv')

def load_pipeline():
    """
    Initializes both the deep learning extractor and the XGBoost classifier.
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