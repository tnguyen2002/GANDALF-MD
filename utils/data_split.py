import pandas as pd
import os
import shutil

# Read the CSV file
df = pd.read_csv('/home/anhn/236_data/ham/HAM/HAM10000_metadata_old_train.csv')

# Split the dataframe
val_df = df.sample(n=2003, random_state=42)  # Randomly select 2003 rows for validation
train_df = df.drop(val_df.index)  # The rest will be used for training

# Create directories for the training and validation images
train_dir = '/home/anhn/236_data/ham/HAM/train_images_128'
val_dir = '/home/anhn/236_data/ham/HAM/val_images_128'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to copy images to new directories
def copy_images(df, source_dir, target_dir):
    for image_id in df['image_id']:
        src_path = os.path.join(source_dir, image_id + '.jpg')  # Adjust the extension if necessary
        dst_path = os.path.join(target_dir, image_id + '.jpg')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Image not found: {src_path}")

# Assuming your images are stored in a directory
image_directory = '/home/anhn/236_data/ham/HAM/old_train_images_128'

# Copy images
copy_images(train_df, image_directory, train_dir)
copy_images(val_df, image_directory, val_dir)

# Save the new CSV files
train_df.to_csv('/home/anhn/236_data/ham/HAM/HAM10000_metadata_train.csv', index=False)
val_df.to_csv('/home/anhn/236_data/ham/HAM/HAM10000_metadata_val.csv', index=False)