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