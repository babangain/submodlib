from sklearn.datasets import make_blobs
import numpy as np

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist

def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:1000]
    return selected_lines

# Root directory where the datasets are located
root_dir = '../datasets'  # Change this to your actual datasets folder path

examples_list = []

# Recursively iterate through all files in the datasets folder
for subdir, dirs, files in sorted(os.walk(root_dir)):
    for file in files:
        if file == 'train.jsonl':
            file_path = os.path.join(subdir, file)
            random_lines = read_lines(file_path)
            examples_list.extend(random_lines)

print(len(examples_list))

data = np.load('../continual/cluster/embeddings_large_test.npy')

from submodlib import FacilityLocationFunction
objFL = FacilityLocationFunction(n=6000, data=data, mode="dense", metric="euclidean")
greedyList = objFL.maximize(budget=10,optimizer='NaiveGreedy',show_progress=True)
print(greedyList)
for item in greedyList:
    print(examples_list[item[0]])