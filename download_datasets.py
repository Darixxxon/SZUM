import kagglehub
import shutil
import os

# Define target directories
project_root = os.getcwd()
mango_target = os.path.join(project_root, "MangoLeaf")
plantvillage_target = os.path.join(project_root, "PlantVillage")

# Ensure target directories exist or create them
os.makedirs(mango_target, exist_ok=True)
os.makedirs(plantvillage_target, exist_ok=True)

# Download MangoLeaf dataset
print("Downloading MangoLeaf dataset...")
mango_path = kagglehub.dataset_download("aryashah2k/mango-leaf-disease-dataset")
print(f"MangoLeaf downloaded to: {mango_path}")

# Move MangoLeaf dataset to correct path
if os.path.exists(mango_target) and os.listdir(mango_target):
    print(f"Warning: {mango_target} is not empty. Removing existing contents.")
    shutil.rmtree(mango_target)
    os.makedirs(mango_target, exist_ok=True)

for item in os.listdir(mango_path):
    src = os.path.join(mango_path, item)
    dst = os.path.join(mango_target, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print(f"MangoLeaf dataset moved to: {mango_target}")

# Download PlantVillage dataset
print("Downloading PlantVillage dataset...")
plantvillage_path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
print(f"PlantVillage downloaded to: {plantvillage_path}")
plantvillage_path = plantvillage_path + "/PlantVillage"

# Move PlantVillage dataset to correct path
if os.path.exists(plantvillage_target) and os.listdir(plantvillage_target):
    print(f"Warning: {plantvillage_target} is not empty. Removing existing contents.")
    shutil.rmtree(plantvillage_target)
    os.makedirs(plantvillage_target, exist_ok=True)

for item in os.listdir(plantvillage_path):
    src = os.path.join(plantvillage_path, item)
    dst = os.path.join(plantvillage_target, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print(f"PlantVillage dataset moved to: {plantvillage_target}")
print("All datasets downloaded and placed in correct directories.")