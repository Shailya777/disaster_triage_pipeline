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
