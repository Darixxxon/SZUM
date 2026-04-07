import os
import shutil
import imagehash
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

path = os.getcwd()
plant_village_train_path = path + '/PlantVillage/train/'
plant_village_val_path = path + '/PlantVillage/val/'
mango_leaf_path = path + '/MangoLeaf'  # ADJUST TO YOUR OWN DIRECTORY NAME

# Toggle duplicate display
SHOW_DUPLICATES = False  # Set to True to display duplicate images side by side

# Classes to skip
SKIP_CLASSES = {
    'mango___cutting_weevil',
    'corn_(maize)___healthy',
    'corn_(maize)___common_rust_',
    'tomato___late_blight'
}

class DuplicateChecker:
    def __init__(self, bucket_prefix=6, threshold=1, show_duplicates=False):
        self.bucket_prefix = bucket_prefix
        self.threshold = threshold
        self.show_duplicates = show_duplicates
        self.reset()

    def reset(self):
        self.seen_exact = {}
        self.seen_buckets = defaultdict(list)

    def fingerprint_image(self, img_path):
        img = Image.open(img_path)
        img = img.convert('L').resize((256, 256), Image.LANCZOS)
        return imagehash.phash(img)

    def bucket_keys(self, fingerprint):
        return {str(fingerprint)[:self.bucket_prefix]}

    def display_duplicate(self, img_path1, img_path2):
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img1)
        ax1.set_title(f'Current: {os.path.basename(img_path1)}')
        ax1.axis('off')
        ax2.imshow(img2)
        ax2.set_title(f'Duplicate: {os.path.basename(img_path2)}')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def is_duplicate(self, img_path):
        """Check whether the image is a duplicate using optimized bucketed matching. Returns the path of the duplicate if found, else None."""
        try:
            fingerprint = self.fingerprint_image(img_path)
            exact_key = str(fingerprint)

            if exact_key in self.seen_exact:
                if self.show_duplicates:
                    self.display_duplicate(img_path, self.seen_exact[exact_key])
                return self.seen_exact[exact_key]

            candidates = []
            for key in self.bucket_keys(fingerprint):
                candidates.extend(self.seen_buckets[key])

            for old_fp, old_path in candidates:
                dist = fingerprint - old_fp
                if dist <= self.threshold:
                    if self.show_duplicates:
                        self.display_duplicate(img_path, old_path)
                    return old_path

            self.seen_exact[exact_key] = img_path
            for key in self.bucket_keys(fingerprint):
                self.seen_buckets[key].append((fingerprint, img_path))

            return None
        except Exception as e:
            print(f"Error hashing {img_path}: {e}")
            return None  # Skip on error


duplicate_checker = DuplicateChecker(show_duplicates=SHOW_DUPLICATES)

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

                if dedup:
                    dup_path = duplicate_checker.is_duplicate(src)
                    if dup_path:
                        if not duplicate_checker.show_duplicates:
                            print(f"Duplicate detected: {src} is duplicate of {dup_path}")
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

                if dedup:
                    dup_path = duplicate_checker.is_duplicate(src)
                    if dup_path:
                        if not duplicate_checker.show_duplicates:
                            print(f"Duplicate detected: {src} is duplicate of {dup_path}")
                        continue

                shutil.copy(src, dst)


print("Creating full Dataset...")
extract_dirs_pv(plant_village_train_path, target_base='Dataset')
extract_dirs_pv(plant_village_val_path, target_base='Dataset')
extract_dirs_ml(mango_leaf_path, target_base='Dataset')

# Reset duplicate detection state for filtered dataset
duplicate_checker.reset()

print("Creating filtered Dataset_filtered...")
extract_dirs_ml(mango_leaf_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
extract_dirs_pv(plant_village_train_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
extract_dirs_pv(plant_village_val_path, target_base='Dataset_filtered', skip_classes=True, dedup=True)
