# Imports:
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# 1. Page Configuration:
st.set_page_config(
    page_title= 'Disaster Triage AI',
    page_icon= '🛰️',
    layout= 'wide',
    initial_sidebar_state= 'expanded'   
)

## Dark Mode Aesthetic
st.markdown("""
            <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .css-1d391kg { background-color: #262730; }
    </style>
            """,
            unsafe_allow_html= True)

# 2. Defining Directory Paths:
SHOWCASE_DIR= 'data/showcase'
SHOWCASE_PATH= 'data/processed/showcase.csv'

# 3. Loading Showcase CSV:
@st.cache_data
def load_data():
    return pd.read_csv(SHOWCASE_PATH)

try:
    df= load_data()
except FileNotFoundError:
    st.error(f'Critical Error: Could not find Showcase file at {SHOWCASE_PATH}.')
    st.stop()

# 4. Sidebar Panel Configuration:
st.sidebar.title('Triage Command')
st.sidebar.markdown('---')

## Extracting Unique Disaster Names and Images for The Dropdown
df['disaster_name']= df['image_name'].apply(lambda x: x.split('_')[0].replace('-', ' ').title())
disaster_list= df['disaster_name'].unique()
selected_disaster= st.sidebar.selectbox('Select Event:', disaster_list)

## Filtering Images based on Selected Disaster
filtered_images= df[df['disaster_name'] == selected_disaster]['image_name'].unique()
selected_image= st.sidebar.selectbox('Intercept Satellite Feed:', filtered_images)

## Slider for Selecting Threshold
st.sidebar.markdown('---')
st.sidebar.subheader('Triage Thresholds')

destroyed_thresh= st.sidebar.slider('Destroyed Confidence Threshold', min_value= 0.30, max_value= 0.99, value= 0.50, step= 0.05)
st.sidebar.caption('Highlight structures with a high probability of total collapse.')


# 5. Data Filtering:

## Getting Data only for the Selected Image
image_df= df[df['image_name'] == selected_image].copy()

# 6. The Main Visualizer View:
st.title(f'Sector Analysis: {selected_disaster}')
st.markdown('Automated Structural Damage Assessment via ResNet50 & XGBoost')

## Loading The Raw Image
img_path= os.path.join(SHOWCASE_DIR, selected_image)
if not os.path.exists(img_path):
    st.error(f'Image missing from Showcase: {selected_image}')
else:
    ### Reading Image and Converting to RGB:
    img= cv2.imread(img_path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ### Tracking Image Metrics:
    total_structures= len(image_df)
    critical_targets= 0

    #### Drawing Bounding Boxes:
    for index, row in image_df.iterrows():
        ymin, xmin, ymax, xmax= int(row['ymin']), int(row['xmin']), int(row['ymax']), int(row['xmax'])

        # Checking if it meets Destroyed Threshold:
        if row['prob_destroyed'] >= destroyed_thresh:
            critical_targets += 1

            # Drawing Red Box:
            cv2.rectangle(img= img, 
                          pt1= (xmin, ymin), 
                          pt2= (xmax, ymax), 
                          color= (255, 0, 0),
                          thickness= 2)
            
            # Adding a Label:
            cv2.putText(img= img,
                        text= f"{row['prob_destroyed']:.2f}",
                        org= (xmin, ymin- 5),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale= 0.4,
                        color= (255, 0, 0),
                        thickness= 1)
        
        else:
            # Drawing Green Box:
            cv2.rectangle(img= img,
                          pt1= (xmin, ymin),
                          pt2= (xmax, ymax),
                          color= (0, 255, 0),
                          thickness= 1)
    
    ### Top Level Metrics:
    col1, col2, col3= st.columns(3)
    col1.metric('Total Structures Detected', total_structures)
    col2.metric('Critical Targets (Destroyed)', critical_targets)
    col3.metric('Deployement Readiness', 'ACTIVE', delta= 'Optimized', delta_color= 'normal')

    
            