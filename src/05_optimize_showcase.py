# Imports:
import os
import glob
import pandas as pd
from PIL import Image

# Setting Up Directory Paths:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE_DIR= os.path.join(PROJECT_DIR, 'data', 'showcase')
CSV_PATH= os.path.join(PROJECT_DIR, 'data', 'processed', 'showcase.csv')

