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