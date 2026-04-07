import os
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

path = os.getcwd() + '/Dataset_filtered'

count = {}
count_fruit = {}
count_resolution = {}
healthy_count = {}

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        fruit = folder.split('_')[0]
        num_files = len(os.listdir(folder_path))

        if fruit not in count_fruit:
            count_fruit[fruit] = 0
        count_fruit[fruit] += num_files

        count[folder] = num_files

        if 'healthy' in folder.lower():
            if fruit not in healthy_count:
                healthy_count[fruit] = 0
            healthy_count[fruit] += num_files
        else:
            key = f"{fruit}_not_healthy"
            if key not in healthy_count:
                healthy_count[key] = 0
            healthy_count[key] += num_files

        # for filename in os.listdir(folder_path):
        #     im = cv2.imread(os.path.join(folder_path, filename))
        #     resolution = f"{im.shape[1]}x{im.shape[0]}"
        #     if resolution not in count_resolution:
        #         count_resolution[resolution] = 0
        #     count_resolution[resolution] += 1


sorted_count = dict(sorted(count.items(), key=lambda item: (item[0].split('_')[0], item[1]), reverse=True))
sorted_fruit = dict(sorted(count_fruit.items(), key=lambda item: item[1], reverse=True))
sorted_resolution = dict(sorted(count_resolution.items(), key=lambda item: item[1], reverse=True))
sorted_healthy = dict(sorted(healthy_count.items(), key=lambda item: item[1], reverse=True))

print(sorted_count)
print(sorted_fruit)
print(sorted_resolution)



plt.figure(figsize=(10,5))
plt.bar(range(len(sorted_fruit)), list(sorted_fruit.values()), align='center')
plt.xticks(range(len(sorted_fruit)), list(sorted_fruit.keys()), rotation='vertical')

for i, v in enumerate(sorted_fruit.values()):
    plt.text(i, v + max(sorted_fruit.values()) * 0.01, str(v),
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14,6))
plt.bar(range(len(sorted_count)), list(sorted_count.values()), align='center')
plt.xticks(range(len(sorted_count)), list(sorted_count.keys()), rotation='vertical')

for i, v in enumerate(sorted_count.values()):
    plt.text(i, v + max(sorted_count.values()) * 0.01, str(v),
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()


grouped = defaultdict(dict)

for folder, cnt in count.items():
    parts = folder.split('_')
    fruit = parts[0]
    disease = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'

    grouped[fruit][disease] = cnt


fruits = list(grouped.keys())

all_diseases = sorted({d for diseases in grouped.values() for d in diseases})

data = {disease: [] for disease in all_diseases}

for fruit in fruits:
    for disease in all_diseases:
        data[disease].append(grouped[fruit].get(disease, 0))


plt.figure(figsize=(12,6))

bottom = [0] * len(fruits)

for disease in all_diseases:
    values = data[disease]
    plt.bar(fruits, values, bottom=bottom, label=disease)

    bottom = [b + v for b, v in zip(bottom, values)]

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()