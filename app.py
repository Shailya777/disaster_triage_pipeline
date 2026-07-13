# Imports:
import os
import cv2
import pandas as pd
import numpy as np
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

# 4. Initializing Session State to Track "Assess Damage" Button and Image Changes:
if 'assessed' not in st.session_state:
    st.session_state.assessed= False
if 'current_image' not in st.session_state:
    st.session_state.current_image= None

# 5. Sidebar Panel Configuration:
st.sidebar.title('Triage Command')
st.sidebar.markdown('---')

## Extracting Unique Disaster Names and Images for The Dropdown
df['disaster_name']= df['image_name'].apply(lambda x: x.split('_')[0].replace('-', ' ').title())
disaster_list= df['disaster_name'].unique()
selected_disaster= st.sidebar.selectbox('Select Event:', disaster_list)

## Filtering Images based on Selected Disaster
filtered_images= df[df['disaster_name'] == selected_disaster]['image_name'].unique()
selected_image= st.sidebar.selectbox('Intercept Satellite Feed:', filtered_images)

## Resetting The Assessment State if User Selects a New Image:
if st.session_state.current_image != selected_image:
    st.session_state.assessed= False
    st.session_state.current_image= selected_image

## Slider for Selecting Threshold
st.sidebar.markdown('---')
st.sidebar.subheader('Triage Thresholds')

destroyed_thresh= st.sidebar.slider('Destroyed Confidence Threshold', min_value= 0.30, max_value= 0.99, value= 0.50, step= 0.05)
st.sidebar.caption('Highlight structures with a high probability of total collapse.')


# 6. Data Filtering:

## Getting Data only for the Selected Image
image_df= df[df['image_name'] == selected_image].copy()

# 7. The Main Visualizer View:
st.title(f'Sector Analysis: {selected_disaster}')

## Pre / Post Disaster Toggle Button:
view_mode= st.radio(
    label= 'Satellite Feed Timeline',
    options= ('Pre-Disaster (Archive)', 'Post-Disaster (Current)'),
    horizontal= True
)

#st.markdown('Automated Structural Damage Assessment via ResNet50 & XGBoost')

## Determining File Paths:
post_img_path= os.path.join(SHOWCASE_DIR, selected_image)
pre_img_path= os.path.join(SHOWCASE_DIR, selected_image.replace('post_disaster', 'pre_disaster'))

# Selecting Image based on View Mode Radio Button:
if view_mode == 'Pre-Disaster (Archive)':
    if not os.path.exists(pre_img_path):
        st.warning('Archiev Image not Found!')
    else:
        img_pre= cv2.cvtColor(cv2.imread(pre_img_path), cv2.COLOR_BGR2RGB)
        st.image(img_pre, use_column_width= True, caption= f'Archive Feed: {os.path.basename(pre_img_path)}')
        st.info("Switch to 'Post-Disaster' to run Damage Assessement.")

else:
    if not os.path.exists(post_img_path):
        st.error(f'Image Missing: {selected_image}')
    else:
        # Layout for Assess Damage Button
        col_btn, col_empty= st.columns([1, 4])
        with col_btn:
            if st.button('Assess Damage', type= 'primary', use_container_width= True):
                st.session_state.assessed= True
        
        img_post= cv2.cvtColor(cv2.imread(post_img_path), cv2.COLOR_BGR2RGB)

        ## Drawing Boxes when Assess Damage Button is Clicked:
        if st.session_state.assessed:

            ### Tracking Image Metrics:
            total_structures= len(image_df)
            critical_targets= 0

            ### Drawing Bounding Boxes:
            for index, row in image_df.iterrows():
                ymin, xmin, ymax, xmax= int(row['ymin']), int(row['xmin']), int(row['ymax']), int(row['xmax'])

                # Checking if it meets Destroyed Threshold:
                if row['prob_destroyed'] >= destroyed_thresh:
                    critical_targets += 1

                    # Drawing Red Box:
                    cv2.rectangle(img= img_post, 
                          pt1= (xmin, ymin), 
                          pt2= (xmax, ymax), 
                          color= (255, 0, 0),
                          thickness= 2)
            
                    # Adding a Label:
                    cv2.putText(img= img_post,
                        text= f"{row['prob_destroyed']:.2f}",
                        org= (xmin, ymin- 5),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale= 0.4,
                        color= (255, 0, 0),
                        thickness= 1)
        
                else:
                    # Drawing Green Box
                    cv2.rectangle(img= img_post,
                          pt1= (xmin, ymin),
                          pt2= (xmax, ymax),
                          color= (0, 255, 0),
                          thickness= 1)
                    
            ### Top Level Metrics
            col1, col2, col3= st.columns(3)
            col1.metric('Total Structures Detected', total_structures)
            col2.metric('Critical Targets (Destroyed)', critical_targets)
            col3.metric('Deployement Readiness', 'ACTIVE', delta= 'Optimized', delta_color= 'normal')

        ### Rendering The Image
        st.image(image= img_post,
             width= 'stretch',
             caption= f'Current Feed: {selected_image}')

        # 7. Dataframe with Information:
        st.subheader('Target Ledger')

    ## Formatiing Dataframe:
    display_df= image_df[['building_uid', 'prob_destroyed', 'prob_major', 'prob_minor', 'prob_no_damage', 'footprint_sq_px']].copy()
    display_df= display_df.sort_values(by= ['prob_destroyed', 'footprint_sq_px'], ascending= [False, False])
    display_df= display_df.reset_index(drop= True)

    ## Renaming Columns:
    display_df.columns= ['Building UID', 'P(Destroyed)', 'P(Major)', 'P(Minor)', 'P(Intact)', 'Footprint (px)']

    ## Rendering Dataframe:
    st.dataframe(data= display_df,
                 use_container_width= True)