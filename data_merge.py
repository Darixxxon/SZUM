import os
import shutil


path = os.getcwd()
plant_village_train_path = path + '/PlantVillage/train/'
plant_village_val_path = path + '/PlantVillage/val/'
mango_leaf_path = path + '/MangoLeaf' # ADJUST TO YOUR OWN DIRECTORY NAME

# Extraction of Plant Village dataset - ignoring directory structure for learning purposes
def extract_dirs_pv(base_path):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            tmp_folder = 'Dataset/' + folder.lower()
            target_path = os.path.join(path, tmp_folder)

            os.makedirs(target_path, exist_ok=True)

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(target_path, filename)

                shutil.copy(src, dst)

# Extraction of Mango Leaf Disease Dataset
def extract_dirs_ml(base_path):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            # Convert "Disease Name" → "mango___disease_name"
            disease_name = folder.lower().replace(" ", "_")
            tmp_folder = f'Dataset/mango___{disease_name}'
            target_path = os.path.join(path, tmp_folder)

            os.makedirs(target_path, exist_ok=True)

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(target_path, filename)

                shutil.copy(src, dst)

# Call extraction methods
extract_dirs_pv(plant_village_train_path)
extract_dirs_pv(plant_village_val_path)
extract_dirs_ml(mango_leaf_path)
