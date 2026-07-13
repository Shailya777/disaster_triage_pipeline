# Imports:
import os
import glob
import pandas as pd
from PIL import Image

# Setting Up Directory Paths:
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE_DIR= os.path.join(PROJECT_DIR, 'data', 'showcase')
CSV_PATH= os.path.join(PROJECT_DIR, 'data', 'processed', 'showcase.csv')

def convert_images():
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
            rgb_img.save(fp= jpg_path, format= 'JPEG', quaulity= 90)

            # Deleting the Original .png file:
            os.remove(png_path)
            success_count += 1
        
        except Exception as e:
            print(f'Error Processing {png_path}: {e}')

    print(f'Successfully Converted {success_count} Images.')

def update_showcase_csv():
    """
    Updates Showcase CSV file to have image_name column .jpg instead of .png.
    """
    print('Updating Showcase CSV...')
    if not os.path.exists(CSV_PATH):
        print(f'Error: Could Not Find {CSV_PATH}')
        return
    
    df= pd.read_csv(CSV_PATH)

    # Replacing .png with .jpg in image_name column:
    df['image_name']= df['image_name'].str.replace('.png', '.jpg')
    df.to_csv(path_or_buf= CSV_PATH, index= False)
    print('Showcase CSV Successfully Updated.')

if __name__ == '__main__':
    convert_images() # Converts All Showcase Images form .png to .jpg, deleting original .png files after successful conversion.
    update_showcase_csv() # Updates Showcase CSV file to have .jpg instead of .png in image_name column.