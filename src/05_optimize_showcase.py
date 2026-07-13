# Imports:
import os
import glob
import pandas as pd
from PIL import Image

# Setting Up Directory Paths:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE_DIR= os.path.join(PROJECT_DIR, 'data', 'showcase')
CSV_PATH= os.path.join(PROJECT_DIR, 'data', 'processed', 'showcase.csv')

def optimize_images():
    """
    Takes all Images from Showcase Directory and converts from .png to .jpeg.
    Deletes original .png files after conversion.
    """

    # Grabbing all .png files from Showcase Dir:
    png_files= glob.glob(os.path.join(SHOWCASE_DIR, '*.png'))

    if not png_files:
        print('No PNG Files Found.')
        return

    print(f'Found {len(png_files)} PNG Files. Startign Conversion...')

    success_count= 0
    for png_path in png_files:

        # Defining new JPEG Path:
        jpg_path= png_path.replace('.png', '.jpg')

        try:
            img= Image.open(png_path)
            rgb_img= img.convert(mode= 'RGB')
            rgb_img.save(fp= jpg_path, format= 'JPEG', quaulity= 85)

            # Deleting the Original .png file:
            os.remove(png_path)
            success_count += 1
        
        except Exception as e:
            print(f'Error Processing {png_path}: {e}')

    print(f'Successfully Converted {success_count} Images.')
