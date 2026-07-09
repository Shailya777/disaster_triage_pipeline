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

## Dark Mode Aesthetic:
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

# Extracting Unique Disaster Names and Images for The Dropdown:
df['disaster_name']= df['image_name'].apply(lambda x: x.split('_')[0].replace('-', ' ').title())
disaster_list= df['disaster_name'].unique()
selected_disaster= st.sidebar.selectbox('Select Event:', disaster_list)

## Filtering Images based on Selected Disaster:
filtered_images= df[df['disaster_name'] == selected_disaster]['image_name'].unique()
selected_image= st.sidebar.selectbox('Intercept Satellite Feed:', filtered_images)

## Slider for Selecting Threshold:
st.sidebar.markdown('---')
st.sidebar.subheader('Triage Thresholds')

destroyed_thresh= st.sidebar.slider('Destroyed Confidence Threshold', min_value= 0.30, max_value= 0.99, value= 0.50, step= 0.05)
st.sidebar.caption('Highlight structures with a high probability of total collapse.')