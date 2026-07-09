# Imports:
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Page Configuration:
st.set_page_config(
    page_title= 'Disaster Triage AI',
    page_icon= '🛰️',
    layout= 'wide',
    initial_sidebar_state= 'expanded'   
)

# Dark Mode Aesthetic:
st.markdown("""
            <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .css-1d391kg { background-color: #262730; }
    </style>
            """,
            unsafe_allow_html= True)

# Defining Directory Paths:
SHOWCASE_DIR= 'data/showcase'
SHOWCASE_PATH= 'data/processed/showcase.csv'

# Loading Showcase CSV:
@st.cache_data
def load_data():
    return pd.read_csv(SHOWCASE_PATH)

try:
    df= load_data()
except FileNotFoundError:
    st.error(f'Critical Error: Could not find Showcase file at {SHOWCASE_PATH}.')
    st.stop()

# Sidebar Panel Configuration:
st.sidebar.title('Triage Command')
st.sidebar.markdown('---')

# Extracting Unique Disaster Names and Images for The Dropdown:
df['disaster_name']= df['image_name'].apply(lambda x: x.split('_')[0].replace('-', ' ').title())
disaster_list= df['disaster_name'].unique()
selected_disaster= st.sidebar.selectbox('Select Event:', disaster_list)

# Filtering Images based on Selected Disaster:
filtered_images= df[df['disaster_name'] == selected_disaster]['image_name'].unique()
selected_image= st.sidebar.selectbox('Intercept Satellite Feed:', filtered_images)