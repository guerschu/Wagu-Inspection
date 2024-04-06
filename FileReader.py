import pandas as pd
from PIL import Image
import os

def load_images_and_annotations(images_dir, data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Could not find the CSV file at the path: {data_path}")
   
    annotations = pd.read_csv(data_path)
    images = {}
    rows_to_drop = []  # Store the indices of rows to drop
    for index, row in annotations.iterrows():
        filename = row['Filename']
        file_path = os.path.join(images_dir, filename)
        print(f"Attempting to load image from path: {file_path}")
        if not os.path.isfile(file_path):
            print(f"File does not exist at the path: {file_path}. Dropping row {index}.")
            rows_to_drop.append(index)  # Add index to list of rows to drop
            continue
        try:
            with Image.open(file_path) as img:
                images[filename] = img.copy()
                print(f"Loaded image: {filename}")
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}. Dropping row {index}.")
            rows_to_drop.append(index)  # Add index to list of rows to drop
    
    # Drop rows with missing or corrupted images
    annotations_cleaned = annotations.drop(index=rows_to_drop).reset_index(drop=True)
    print(f"Number of rows dropped: {len(rows_to_drop)}")
    
    # Get file names from file paths in images directory
    image_filenames = [os.path.basename(image_path) for image_path in os.listdir(images_dir)]
    
    # Replace "Filename" column in annotations DataFrame with file names from images directory
    annotations_cleaned['Filename'] = image_filenames
    
    return images, annotations_cleaned

images_directory_path = "TrainFiles"
data_csv_path = "annotations.csv"

print(f"Current working directory: {os.getcwd()}")

try:
    loaded_images, annotations_df = load_images_and_annotations(images_directory_path, data_csv_path)
except FileNotFoundError as e:
    print(e)
