import os
import matplotlib.pyplot as plt
import cv2

path = os.getcwd() + '/Dataset_filtered'

count = {}
count_fruit = {}
count_resolution = {}
healthy_count = {}

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        fruit = folder.split('_')[0]
        if fruit not in count_fruit:
            count_fruit[fruit] = 0
        count_fruit[fruit] += len(os.listdir(folder_path))
        count[folder] = len(os.listdir(folder_path))
        if 'healthy' in folder_path:
            if fruit not in healthy_count:
                healthy_count[fruit] = 0
            healthy_count[fruit] += len(os.listdir(folder_path))
        else:
            if f"{fruit}_not_healthy" not in healthy_count:
                healthy_count[f"{fruit}_not_healthy"] = 0
            healthy_count[f"{fruit}_not_healthy"] += len(os.listdir(folder_path))
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


# plt.pie(list(sorted_fruit.values()), labels=list(sorted_fruit.keys()), autopct='%1.1f%%')
# plt.axis('equal')
# plt.show()

# plt.pie(list(sorted_healthy.values()), labels=list(sorted_healthy.keys()), autopct='%1.1f%%')
# plt.axis('equal')
# plt.show()

plt.bar(range(len(sorted_fruit)), list(sorted_fruit.values()), align='center')
plt.xticks(range(len(sorted_fruit)), list(sorted_fruit.keys()), rotation='vertical')
for i, v in enumerate(sorted_fruit.values()):
    plt.text(i, v + max(sorted_fruit.values()) * 0.01, str(v),
             ha='center', va='bottom', fontsize=8)
plt.show()

plt.bar(range(len(sorted_count)), list(sorted_count.values()), align='center')
plt.xticks(range(len(sorted_count)), list(sorted_count.keys()), rotation='vertical')
for i, v in enumerate(sorted_count.values()):
    plt.text(i, v + max(sorted_count.values()) * 0.01, str(v),
             ha='center', va='bottom', fontsize=8)
plt.show()

# plt.bar(range(len(sorted_resolution)), list(sorted_resolution.values()), align='center')
# plt.xticks(range(len(sorted_resolution)), list(sorted_resolution.keys()), rotation='vertical')
# for i, v in enumerate(sorted_resolution.values()):
#     plt.text(i, v + max(sorted_resolution.values()) * 0.01, str(v),
#              ha='center', va='bottom')
# plt.show()