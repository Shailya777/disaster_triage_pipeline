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

# To Show all columns without truncation
pd.set_option('display.max_columns', None)

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

# ---------------------------------------------------------------------
# Curating Golden Dataset from Training Images
# ---------------------------------------------------------------------

def curate_golden_samples():
    """
    Algorithmic curation pipeline to generate a 50-image 'Golden Sample' showcase.
    
    This function bypasses manual image selection by mathematically evaluating the 
    model's performance across the dataset. It loads the frozen ResNet50 features 
    and the serialized XGBoost champion model, generates in-memory predictions using 
    custom disaster triage thresholds, and aggregates performance metrics at the 
    image level using Pandas.

    Images are selected based on three strict operational criteria:
    1. High Density & High Confidence: Images with 50+ structures, >85% accuracy, 
       and >3 destroyed targets.
    2. The 'Needle in a Haystack': Images with 20+ intact structures but only 1-2 
       destroyed targets, flagged with >90% accuracy.
    3. Edge Cases & Terrain Diversity: Top-performing images (>80% accuracy) 
       sampled evenly across different disaster categories (e.g., hurricanes, fires).

    The final set of up to 50 unique, deduplicated images is automatically copied 
    from the raw source directory to the designated showcase directory for 
    downstream processing by the web application.

    Dependencies:
        - Requires `features.tfrecord` and `xgboost_weighted_champion.json`.
        - Relies on global environment variables: TFRECORD_PATH, MODEL_JSON_PATH, 
          SOURCE_IMG_DIR, and SHOWCASE_DIR.

    Returns:
        None (Executes file I/O copying operations directly).
    """

    # -------------------------------------------------------
    # Phase 1: Genrating Predictions using XGBoost Classifier for entire Dataset
    # -------------------------------------------------------

    # Loading Dataset:
    X_raw, y_raw, building_ids= load_dataset()

    class_mapping= {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 2,
        'destroyed': 3,
        'un-classified': 0
        }
    
    # Generating Encoded Lables:
    y_true= np.array([class_mapping.get(label, 0) for label in y_raw])

    # Loading XGBoost Classifier:
    xgb_model= xgb.XGBClassifier()
    xgb_model.load_model(fname= MODEL_JSON_PATH)

    # Generating Predictions across Entire Dataset using XGBoost Classifier:
    # Using Same Custom Threshold Logic from Experiment 5 in 02_xgboost_experiments.ipynb:
    raw_probs= xgb_model.predict_proba(X_raw)
    y_pred= []

    for p in raw_probs:
        if p[3] > 0.30: y_pred.append(3)
        elif p[2] > 0.30: y_pred.append(2)
        else: y_pred.append(np.argmax(p[:2]))
    
    y_pred= np.array(y_pred)

    #print(building_ids[:5])
    
    # -------------------------------------------------------
    # Phase 2: Data Curation
    # -------------------------------------------------------

    print('Building Dataframe for Data Curation...')
    df= pd.DataFrame({
        'building_id': building_ids,
        'true_label': y_true,
        'pred_label': y_pred
    })

    # Extracting Image name from Building ID: (Buidling ID Format: imagename_uid)
    df['image_name']= df['building_id'].apply(lambda x: '_'.join(x.split('_')[:4]) + '.png')
    df['disaster_type']= df['building_id'].apply(lambda x: x.split('_')[0])
    df['is_correct']= (df['true_label'] == df['pred_label']).astype(int)
    df['is_destroyed']= (df['true_label'] == 3).astype(int)
    df['is_nodamage']= (df['true_label'] == 0).astype(int)

    # Generating Image Stats:
    image_stats= df.groupby('image_name').agg(
        disaster_type= ('disaster_type', 'first'),
        total_buildings= ('building_id', 'count'),
        accuracy= ('is_correct', 'mean'),
        total_destroyed= ('is_destroyed', 'sum'),
        total_nodamage= ('is_nodamage', 'sum')
    ).reset_index()

if __name__ == '__main__':
    curate_golden_samples()