import os
import shutil
import imagehash
from collections import defaultdict
from PIL import Image

path = os.getcwd()
plant_village_train_path = path + '/PlantVillage/train/'
plant_village_val_path = path + '/PlantVillage/val/'
mango_leaf_path = path + '/MangoLeaf'  # ADJUST TO YOUR OWN DIRECTORY NAME

# Classes to skip
SKIP_CLASSES = {
    'mango___cutting_weevil',
    'corn_(maize)___healthy',
    'corn_(maize)___common_rust_',
    'tomato___late_blight'
}

class DuplicateChecker:
    def __init__(self, bucket_prefix=6, threshold=5):
        self.bucket_prefix = bucket_prefix
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.seen_exact = set()
        self.seen_buckets = defaultdict(list)

    def fingerprint_image(self, img_path):
        img = Image.open(img_path)
        img = img.convert('L').resize((256, 256), Image.LANCZOS)
        return {
            'ahash': imagehash.average_hash(img),
            'phash': imagehash.phash(img),
            'dhash': imagehash.dhash(img)
        }

    def bucket_keys(self, fingerprint):
        return {
            str(fingerprint['ahash'])[:self.bucket_prefix],
            str(fingerprint['phash'])[:self.bucket_prefix],
            str(fingerprint['dhash'])[:self.bucket_prefix]
        }

    def is_duplicate(self, img_path):
        """Check whether the image is a duplicate using optimized bucketed matching."""
        try:
            fingerprint = self.fingerprint_image(img_path)
            exact_key = str(fingerprint['ahash'])

            if exact_key in self.seen_exact:
                return True

            candidates = []
            for key in self.bucket_keys(fingerprint):
                candidates.extend(self.seen_buckets[key])

            for old_fp in candidates:
                ah_dist = fingerprint['ahash'] - old_fp['ahash']
                ph_dist = fingerprint['phash'] - old_fp['phash']
                dh_dist = fingerprint['dhash'] - old_fp['dhash']
                if ah_dist <= self.threshold and ph_dist <= self.threshold and dh_dist <= self.threshold:
                    return True

            self.seen_exact.add(exact_key)
            for key in self.bucket_keys(fingerprint):
                self.seen_buckets[key].append(fingerprint)

            return False
        except Exception as e:
            print(f"Error hashing {img_path}: {e}")
            return False  # Skip on error


duplicate_checker = DuplicateChecker()

# Extraction of Plant Village dataset - ignoring directory structure for learning purposes
def extract_dirs_pv(base_path, target_base='Dataset', skip_classes=False, dedup=False):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            class_name = folder.lower()
            tmp_folder = f'{target_base}/' + class_name
            if skip_classes and class_name in SKIP_CLASSES:
                print(f"Skipping class: {class_name}")
                continue

            target_path = os.path.join(path, tmp_folder)
            os.makedirs(target_path, exist_ok=True)

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(target_path, filename)

                if dedup and duplicate_checker.is_duplicate(src):
                    print(f"Skipping duplicate: {src}")
                    continue

                shutil.copy(src, dst)

# Extraction of Mango Leaf Disease Dataset
def extract_dirs_ml(base_path, target_base='Dataset', skip_classes=False, dedup=False):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            # Convert "Disease Name" → "mango___disease_name"
            disease_name = folder.lower().replace(" ", "_")
            class_name = f'mango___{disease_name}'
            tmp_folder = f'{target_base}/{class_name}'
            if skip_classes and class_name in SKIP_CLASSES:
                print(f"Skipping class: {class_name}")
                continue

            target_path = os.path.join(path, tmp_folder)
            os.makedirs(target_path, exist_ok=True)

            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(target_path, filename)

                if dedup and duplicate_checker.is_duplicate(src):
                    print(f"Skipping duplicate: {src}")
                    continue

                shutil.copy(src, dst)

# Call extraction methods for full dataset
print("Creating full Dataset...")
extract_dirs_pv(plant_village_train_path, target_base='Dataset')
extract_dirs_pv(plant_village_val_path, target_base='Dataset')
extract_dirs_ml(mango_leaf_path, target_base='Dataset')

# Reset duplicate detection state for filtered dataset
duplicate_checker.reset()

# Call extraction methods for filtered dataset
print("Creating filtered Dataset_filtered...")
extract_dirs_pv(plant_village_train_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
extract_dirs_pv(plant_village_val_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
extract_dirs_ml(mango_leaf_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
