import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import shutil

# Configuration
SOURCE_DIR = Path("Dataset")
TARGET_DIR = Path("Dataset_grayscale_eq")
IMG_SIZE = (256, 256)  # Match the validation script

def preprocess_image(img_path, target_path):
    """
    Load image, convert to grayscale, apply histogram equalization, and save.
    """
    try:
        # Load image
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)

        # Convert to grayscale
        gray_img = ImageOps.grayscale(img)

        # Convert to numpy array for OpenCV
        gray_array = np.array(gray_img)

        # Apply histogram equalization
        eq_array = cv2.equalizeHist(gray_array)

        # Convert back to PIL Image
        eq_img = Image.fromarray(eq_array)

        # Save to target path
        os.makedirs(target_path.parent, exist_ok=True)
        eq_img.save(target_path)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def preprocess_dataset(source_dir, target_dir):
    """
    Process all images in the dataset.
    """
    if target_dir.exists():
        print(f"Target directory {target_dir} already exists. Removing it.")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        print(f"Processing class: {class_name}")

        for img_path in class_dir.glob("*.jpg"):
            target_path = target_dir / class_name / img_path.name
            preprocess_image(img_path, target_path)
            total_processed += 1

    print(f"Preprocessing complete. Processed {total_processed} images.")
    print(f"Modified dataset saved to: {target_dir}")

if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"Source directory {SOURCE_DIR} not found. Please ensure Dataset exists.")
        exit(1)

    preprocess_dataset(SOURCE_DIR, TARGET_DIR)