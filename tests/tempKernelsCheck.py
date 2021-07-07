from sklearn.datasets import make_blobs
import random
import numpy as np
#import submodlib_cpp as subcp
import submodlib.helper as helper
import time
import csv

method_dictionary = {"np_numba": "create_kernel_dense_np_numba",
                     "np": "create_kernel_dense_np",
                     "fastdist": "create_kernel_dense_fastdist",
                     #"scipy_numba": "create_kernel_dense_scipy_numba",
                     "scipy": "create_kernel_dense_scipy",
                     #"sklearn_numba": "create_kernel_dense_sklearn_numba",
                     "sklearn": "create_kernel_dense_sklearn",
                     #"current_numba": "create_kernel_numba",
                     "current": "create_kernel"
}

cluster_std_dev = 2
num_executions = 1
num_places = 6
num_features = 1024

param = (3, 1)

num_samples = param[0]
num_clusters = param[1]

points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
data = list(map(tuple, points))
dataArray = np.array(data)

for method in method_dictionary:
    print("\n***Kernel from ", method)
    func = getattr(helper, method_dictionary[method])
    kernel = func(dataArray, "euclidean")
    print(kernel)