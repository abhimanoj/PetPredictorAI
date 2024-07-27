# Import necessary libraries
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Seed setting function for reproducibility
from helpers.helper_functions import seed_everything
seed_everything()

# Constants
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
DATASET_PATH = "datasets/animals10/raw-img"

def check_and_display_directory_contents(directory):
    """ Check if the directory exists and print its contents. """
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {directory}")
        return False
    print(f"Directory found: {directory}\nListing contents:")
    for item in os.listdir(directory):
        print(item)
    return True

def load_and_prepare_data(dataset_path):
    """ Load image file paths and their labels into a DataFrame. """
    image_dir = Path(dataset_path)
    file_patterns = ['**/*.JPG', '**/*.jpg', '**/*.jpeg', '**/*.PNG', '**/*.png']
    filepaths = [fp for pattern in file_patterns for fp in image_dir.glob(pattern)]
    labels = [os.path.split(os.path.split(fp)[0])[1] for fp in filepaths]
    return pd.DataFrame({'Filepath': filepaths, 'Label': labels})

def setup_data_augmentation():
    """ Setup and return a data augmentation pipeline. """
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

if check_and_display_directory_contents(DATASET_PATH):
    image_df = load_and_prepare_data(DATASET_PATH)
    print(image_df.head())
    print(f"Total images loaded: {len(image_df)}")

    # Initialize data augmentation pipeline
    augment = setup_data_augmentation()
else:
    print("Failed to find or access dataset directory. Please check the path and permissions.")
