import os
from PIL import Image

dataset_path = os.getcwd() + '/Dataset_filtered/'

TARGET_SIZE = (256, 256)

def resize_images(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)

                with Image.open(file_path) as img:
                        img = img.convert("RGB")

                        img_resized = img.resize(TARGET_SIZE)

                        img_resized.save(file_path)



resize_images(dataset_path)