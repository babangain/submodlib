import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from sklearn.cluster import Birch #https://scikit-learn.org/stable/modules/clustering.html#birch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
#from fastdist import fastdist
from numba import jit, config
import pickle
import time
import os
import sys
import faiss
import re
from tqdm.auto import tqdm
#from tqdm import tqdm
#from tqdm import trange

#config.THREADING_LAYER = 'default'

#TODO: https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
#Represent a dense kernel with upper triangular entries only to save memory
def cos_sim_square(A):
    # base similarity matrix (all dot products)
    similarity = np.dot(A, A.T)

    # squared magnitude of vectors
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

def cos_sim_rectangle(A, B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def euc_dis(A, B):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)
    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)

@jit(nopython=True, parallel=True)
def euc_dis_numba(A, B):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)
    D_squared = np.where(D_squared < 0.0, 0, D_squared)
    #zero_mask = np.less(D_squared, 0.0)
    #D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)

@jit(nopython=True, parallel=True)
def cos_sim_square_numba(A):
    # base similarity matrix (all dot products)
    similarity = np.dot(A, A.T)

    # squared magnitude of vectors
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

@jit(nopython=True, parallel=True)
def cos_sim_rectangle_numba(A, B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def create_sparse_kernel(X, metric, num_neigh, n_jobs=1, method="sklearn"):
    if num_neigh>np.shape(X)[0]:
        raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    dense = None
    if method == "sklearn":
        dense = create_kernel_dense_sklearn(X, metric)
    elif method == "np_numba":
        dense = create_kernel_dense_np_numba(X, metric)
    else:
        raise Exception("For creating sparse kernel, only 'sklearn' and 'np_numba' methods are supported")
    #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
    dense_ = None
    if num_neigh==-1:
        num_neigh=np.shape(X)[0] #default is total no of datapoints
    nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
    _, ind = nbrs.kneighbors(X)
    ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
    row, col = zip(*ind_l)   #unzipping
    mat = np.zeros(np.shape(dense))
    mat[row, col]=1
    dense_ = dense*mat #Only retain similarity of nearest neighbours
    sparse_csr = sparse.csr_matrix(dense_)
    return sparse_csr

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_numba(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None):
#     #print(type(X_rep))
#     if type(X_rep)!=type(None) and mode=="sparse":
#         raise Exception("ERROR: sparse mode not allowed when using rectangular kernel")

#     if type(X_rep)!=type(None) and num_neigh!=-1:
#         raise Exception("ERROR: num_neigh can't be specified when using rectangular kernel")

#     if num_neigh==-1 and type(X_rep)==type(None):
#         num_neigh=np.shape(X)[0] #default is total no of datapoints

#     if num_neigh>np.shape(X)[0]:
#         raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    
#     if mode in ['dense', 'sparse']:
#         dense=None
#         D=None
#         if metric=="euclidean":
#             if type(X_rep)==type(None):
#                 D = euclidean_distances(X)
#             else:
#                 D = euclidean_distances(X_rep, X)
#             gamma = 1/np.shape(X)[1]
#             dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#         else:
#             if metric=="cosine":
#                 if type(X_rep)==type(None):
#                     dense = cosine_similarity(X)
#                 else:
#                     dense = cosine_similarity(X_rep, X)
#             else:
#                 raise Exception("ERROR: unsupported metric")
        
#         #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
#         dense_ = None
#         if type(X_rep)==type(None):
#             nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
#             _, ind = nbrs.kneighbors(X)
#             ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
#             row, col = zip(*ind_l)
#             mat = np.zeros(np.shape(dense))
#             mat[row, col]=1
#             dense_ = dense*mat #Only retain similarity of nearest neighbours
#         else:
#             dense_ = dense

#         if mode=='dense': 
#             if num_neigh!=-1:       
#                 return num_neigh, dense_
#             else:
#                 return dense_ #num_neigh is not returned because its not a sensible value in case of rectangular kernel
#         else:
#             sparse_csr = sparse.csr_matrix(dense_)
#             return num_neigh, sparse_csr
      
#     else:
#         raise Exception("ERROR: unsupported mode")

#@jit(nopython=True, parallel=True)
def create_kernel_dense_other(X, metric, X_rep=None):
    D = None
    if metric == 'euclidean':
        D = pairwise_distances(X, Y=X_rep, metric='euclidean', squared=True)
        D = np.subtract(D.max(), D, out=D)
    elif metric == 'cosine':
        D = pairwise_distances(X, Y=X_rep, metric="cosine")
        D = np.subtract(1, D, out=D)
        D = np.square(D, out=D)
        D = np.subtract(1, D, out=D)
        D = np.subtract(1, D, out=D)
    else:
        raise Exception("Unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(D.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(D.shape == (X.shape[0], X.shape[0]))
    return D


def create_kernel_dense_rowwise(X, metric, X_rep=None):
    if type(X_rep) != type(None):
        num_rows = X_rep.shape[0]
    else:
        num_rows = X.shape[0]
    tempFile = 'kernel'+str(time.time())
    with open(tempFile, 'ab') as f:
        if metric == "cosine":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    similarity = cosine_similarity(i.reshape(1, -1), X).flatten()
                    pickle.dump(similarity, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    similarity = cosine_similarity(i.reshape(1, -1), X).flatten()
                    pickle.dump(similarity, f)
        elif metric == "euclidean":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    distance = euclidean_distances(i.reshape(1, -1), X).flatten()
                    pickle.dump(distance, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    distance = euclidean_distances(i.reshape(1, -1), X).flatten()
                    pickle.dump(distance, f)
        elif metric == "dot":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    similarity = np.dot(i.reshape(1, -1), X.T).flatten()
                    pickle.dump(similarity, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    similarity = np.dot(i.reshape(1, -1), X.T).flatten()
                    pickle.dump(similarity, f)
        else:
            raise Exception("Unsupported metric for this method of kernel creation")
    with open(tempFile, 'rb') as f:
        D = []
        #for i in trange(num_rows):
        for i in range(num_rows):
            D.append(pickle.load(f))
        D = np.array(D)
        if metric == "cosine" or metric == "dot":
            dense = D
        elif metric == "euclidean":
            gamma = 1/np.shape(X)[1]
            dense = np.exp(-D * gamma)
    os.remove(tempFile)
    assert(dense.shape == (num_rows, X.shape[0]))
    return dense

def create_kernel_dense_sklearn(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = euclidean_distances(X)
        else:
            D = euclidean_distances(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            dense = cosine_similarity(X)
        else:
            dense = cosine_similarity(X_rep, X)
    elif metric == "dot":
        if type(X_rep)==type(None):
            dense = np.matmul(X, X.T)
        else:
            dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_dense_sklearn_numba(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = euclidean_distances(X)
#         else:
#             D = euclidean_distances(X_rep, X)
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     else:
#         if metric=="cosine":
#             if type(X_rep)==type(None):
#                 dense = cosine_similarity(X)
#             else:
#                 dense = cosine_similarity(X_rep, X)
#         else:
#             raise Exception("ERROR: unsupported metric")
#     return dense

def create_kernel_dense_scipy(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = distance_matrix(X, X)
        else:
            D = distance_matrix(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            D = pdist(X, metric="cosine")
            dense = squareform(D, checks=False)
            dense = 1-dense
        else:
            dense = cdist(X_rep, X, metric="cosine")
            dense = 1-dense
    else:
        raise Exception("ERROR: unsupported metric")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_dense_scipy_numba(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = distance_matrix(X, X)
#         else:
#             D = distance_matrix(X_rep, X)
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     else:
#         if metric=="cosine":
#             if type(X_rep)==type(None):
#                 D = pdist(X, metric="cosine")
#                 dense = squareform(D, checks=False)
#                 dense = 1-dense
#             else:
#                 dense = cdist(X_rep, X, metric="cosine")
#                 dense = 1-dense
#         else:
#             raise Exception("ERROR: unsupported metric")
#     return dense

# def create_kernel_dense_fastdist(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = fastdist.matrix_pairwise_distance(X, fastdist.euclidean, "euclidean", return_matrix=True)
#         else:
#             D = fastdist.matrix_to_matrix_distance(X_rep, X, fastdist.euclidean, "euclidean")
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     elif metric=="cosine":
#         if type(X_rep)==type(None):
#             D = fastdist.matrix_pairwise_distance(X, fastdist.cosine, "cosine", return_matrix=True)
#         else:
#             D = fastdist.matrix_to_matrix_distance(X_rep, X, fastdist.cosine, "cosine")
#         #dense = 1-D
#         dense = D
#     else:
#         raise Exception("ERROR: unsupported metric")
#     if type(X_rep) != type(None):
#         assert(dense.shape == (X_rep.shape[0], X.shape[0]))
#     else:
#         assert(dense.shape == (X.shape[0], X.shape[0]))
#     return dense

def create_kernel_dense_np(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = euc_dis(X, X)
        else:
            D = euc_dis(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            dense = cos_sim_square(X)
        else:
            dense = cos_sim_rectangle(X_rep, X)
    elif metric=="dot":
        if type(X_rep)==type(None):
            dense = np.matmul(X, X.T)
        else:
            dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

@jit(nopython=True, parallel=True)
def create_kernel_dense_np_numba(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        D = euc_dis_numba(X, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        dense = cos_sim_square_numba(X)
    # elif metric=="dot":
    #     dense = np.matmul(X, X.T)
    else:
        raise Exception("ERROR: unsupported metric")
    assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

@jit(nopython=True, cache=True, parallel=True)
def create_kernel_dense_np_numba_rectangular(X, metric, X_rep):
    dense=None
    D=None
    if metric=="euclidean":
        D = euc_dis_numba(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        dense = cos_sim_rectangle_numba(X_rep, X)
    elif metric=="dot":
        dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

def create_cluster_kernels(X, metric, cluster_lab=None, num_cluster=None, onlyClusters=False): #Here cluster_lab is a list which specifies custom cluster mapping of a datapoint to a cluster
    
    lab=[]
    if cluster_lab==None:
        obj = Birch(n_clusters=num_cluster) #https://scikit-learn.org/stable/modules/clustering.html#birch
        obj = obj.fit(X)
        lab = obj.predict(X).tolist()
        if num_cluster==None:
            num_cluster=len(obj.subcluster_labels_)
    else:
        if num_cluster==None:
            raise Exception("ERROR: num_cluster needs to be specified if cluster_lab is provided")
        lab=cluster_lab
    
    #print("Custer labels: ", lab)

    l_cluster= [set() for _ in range(num_cluster)]
    l_ind = [0]*np.shape(X)[0]
    l_count = [0]*num_cluster
    
    for i, el in enumerate(lab):#For any cluster ID (el), smallest datapoint (i) is filled first
                                #Therefore, the set l_cluster will always be sorted
        #print(f"{i} is in cluster {el}")
        l_cluster[el].add(i)
        l_ind[i]=l_count[el]
        l_count[el]=l_count[el]+1

    #print("l_cluster inside helper: ", l_cluster)

    if onlyClusters:
        return l_cluster, None, None
        
    l_kernel =[]
    for el in l_cluster: 
        k = len(el)
        l_kernel.append(np.zeros((k,k))) #putting placeholder matricies of suitable size
    
    M=None
    if metric=="euclidean":
        D = euclidean_distances(X)
        gamma = 1/np.shape(X)[1]
        M = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        M = cosine_similarity(X)
    else:
        raise Exception("ERROR: unsupported metric")
    
    #Create kernel for each cluster using the bigger kernel
    for ind, val in np.ndenumerate(M): 
        if lab[ind[0]]==lab[ind[1]]:#if a pair of datapoints is in same cluster then update the kernel corrosponding to that cluster 
            c_ID = lab[ind[0]]
            i=l_ind[ind[0]]
            j=l_ind[ind[1]]
            l_kernel[c_ID][i,j]=val
            
    return l_cluster, l_kernel, l_ind

def create_kernel(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None, method="sklearn"):
    if type(X_rep) != type(None):
        assert(X_rep.shape[1] == X.shape[1])
    if mode == "dense":
        dense = None
        if method == "np_numba" and type(X_rep) != type(None):
            dense = create_kernel_dense_np_numba_rectangular(X, metric, X_rep)
        else:
            dense = globals()['create_kernel_dense_'+method](X, metric, X_rep)
        return dense
    elif mode == "sparse":
        if type(X_rep) != type(None):
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, n_jobs, method)
    else:
        raise Exception("ERROR: unsupported mode")

def create_sparse_kernel_faiss_innerproduct(
        X, index_key, logger, ngpu=-1, 
        tempmem=-1, altadd=False, 
        use_float16=True, use_precomputed_tables=True, 
        replicas=1, max_add=-1, add_batch_size=32768, 
        query_batch_size=16384, nprobe=128, nnn=10
    ):
    
    if nnn>np.shape(X)[0]:
        raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    if nnn==-1:
        nnn=np.shape(X)[0] #default is total no of datapoints
    # Parse index_key
    # The index_key is a valid factory key that would work, but we decompose the training to do it faster
    pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                    '(IVF[0-9]+),' +
                    '(PQ[0-9]+|Flat)')
    matchobject=pat.match(index_key)
    assert matchobject, "could not parse "+index_key
    mog=matchobject.groups()
    preproc_str=mog[0]
    ivf_str=mog[2]
    pqflat_str=mog[3]
    ncent=int(ivf_str[3:])

    class IdentPreproc:
        """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""
        def __init__(self, d):
            self.d_in=self.d_out=d
        
        def apply_py(self, x):
            return x

    # Wake up GPUs
    logger.info(f"preparing resources for {ngpu} GPUs")
    gpu_resources=[]
    for i in range(ngpu):
        res=faiss.StandardGpuResources()
        if tempmem>=0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    def make_vres_vdev(i0=0, i1=-1):
        " return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        return vres, vdev

    # get preprocessor
    if preproc_str:
        logger.info(f"train preproc {preproc_str}")
        d = X.shape[1]
        t0 = time.time()
        if preproc_str.startswith('OPQ'):
            fi = preproc_str[3:-1].split('_')
            m = int(fi[0])
            dout = int(fi[1]) if len(fi) == 2 else d
            preproc = faiss.OPQMatrix(d, m, dout)
        elif preproc_str.startswith('PCAR'):
            dout = int(preproc_str[4:-1])
            preproc = faiss.PCAMatrix(d, dout, 0, True)
        else:
            assert False
        preproc.train(np.ascontiguousarray(X.astype("float32")))
        logger.info("preproc train done in %.3f s" % (time.time() - t0))
    else:
        d=X.shape[1]
        preproc=IdentPreproc(d)

    # coarse_quantizer=prepare_coarse_quantizer(preproc)
    nt=max(1000000, 256*ncent)
    logger.info("train coarse quantizer...")
    t0=time.time()
    # centroids=train_coarse_quantizer(X[:nt], ncent, preproc)
    d=preproc.d_out
    clus=faiss.Clustering(d, ncent)
    clus.spherical=True
    clus.verbose=True
    clus.max_points_per_centroid=10000000
    logger.info(f"apply preproc on shape  {X[:nt].shape}, k=, {ncent}")
    t0=time.time()
    x=preproc.apply_py(np.ascontiguousarray(X[:nt]))
    logger.info("preproc %.3f s output shape %s"%(time.time()-t0, x.shape))
    vres, vdev = make_vres_vdev()
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlatIP(d))
    clus.train(x, index)
    centroids=faiss.vector_float_to_array(clus.centroids)
    del x
    centroids=centroids.reshape(ncent, d)
    logger.info("Coarse train time: %.3f s"%(time.time()-t0))
    coarse_quantizer=faiss.IndexFlatIP(preproc.d_out)
    coarse_quantizer.add(centroids)
    d=preproc.d_out
    if pqflat_str=="Flat":
        logger.info("making an IVFFlat Index")
        indexall=faiss.IndexIVFFlat(coarse_quantizer, d, ncent, faiss.METRIC_INNER_PRODUCT)
    else:
        m=int(pqflat_str[2:])
        assert m<56 or use_float16, f"PQ{m} will work only with -float16"
        logger.info(f"making an IVFPQ index, m={m}")
        indexall=faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)
    coarse_quantizer.this.disown()
    indexall.own_fields=True
    # finish the training on CPU
    t0=time.time()
    logger.info("Training vector codes")
    indexall.train(preproc.apply_py(np.ascontiguousarray(X.astype("float32"))))
    logger.info("done %.3f s"%(time.time()-t0))
    # Prepare the index
    if not altadd:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = use_float16
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = use_precomputed_tables
        co.indicesOptions = faiss.INDICES_CPU
        co.verbose = True
        co.reserveVecs = max_add if max_add > 0 else X.shape[0]
        co.shard = True
        assert co.shard_type in (0, 1, 2)
        vres, vdev = make_vres_vdev()
        gpu_index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall, co)
        logger.info("add...")
        t0 = time.time()
        nb=X.shape[0]
        block_ranges=[(i0, min(nb, i0+add_batch_size)) for i0 in range(0, nb, add_batch_size)]
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            gpu_index.add_with_ids(xs, np.arange(i0, i1))
            if max_add>0 and gpu_index.ntotal>max_add:
                logger.info("Flush indexes to CPU")
                for i in range(ngpu):
                    index_src_gpu=faiss.downcast_index(gpu_index.at(i))
                    index_src=faiss.index_gpu_to_cpu(index_src_gpu)
                    logger.info(f"index {i} size {index_src.ntotal}")
                    index_src.copy_subset_to(indexall, 0, 0, nb)
                    index_src_gpu.reset()
                    index_src_gpu.reserveMemory(max_add)
                gpu_index.syncWithSubIndexes()
            logger.info('\r%d/%d (%.3f s)  ' % (
                i0, nb, time.time() - t0))
            sys.stdout.flush()
        logger.info("Add time: %.3f s"%(time.time()-t0))
        # logger.info("Aggregate indexes to CPU")
        # t0=time.time()
        # if hasattr(gpu_index, "at"):
        #     # it is a sharded index
        #     for i in range(ngpu):
        #         index_src=faiss.index_gpu_to_cpu(gpu_index.at(i))
        #         logger.info(f"index {i} size {index_src.ntotal}")
        #         index_src.copy_subset_to(indexall, 0, 0, nb)
        # else:
        #     # simple index
        #     index_src=faiss.index_gpu_to_cpu(gpu_index)
        #     index_src.copy_subset_to(indexall, 0, 0, nb)
        # logger.info("done in %.3f s"%(time.time()-t0))
        if max_add>0:
            # it does not contain all the vectors
            gpu_index=None
    else:
        # set up a 3-stage pipeline that does:
        # - stage 1: load + preproc
        # - stage 2: assign on GPU
        # - stage 3: add to index
        vres, vdev = make_vres_vdev()
        coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall.quantizer)
        nb=X.shape[0]
        block_ranges=[(i0, min(nb, i0+add_batch_size)) for i0 in range(0, nb, add_batch_size)]
        logger.info("add...")
        t0 = time.time()
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            _, assign=coarse_quantizer_gpu.search(xs, 1)
            if indexall.__class__==faiss.IndexIVFPQ:
                indexall.add_core_o(i1-i0, faiss.swig_ptr(xs), None, None, faiss.swig_ptr(assign))
            elif indexall.__class__==faiss.IndexIVFFlat:
                indexall.add_core(i1-i0, faiss.swig_ptr(xs), None, faiss.swig_ptr(assign))
            else:
                assert False
            logger.info('\r%d/%d (%.3f s)  ' % (
                i0, nb, time.time() - t0))
            sys.stdout.flush()
        logger.info("Add time: %.3f s"%(time.time()-t0))
        gpu_index=None

    
    co=faiss.GpuMultipleClonerOptions()
    co.useFloat16=use_float16
    co.useFloat16CoarseQuantizer=False
    co.usePrecomputed=use_precomputed_tables
    co.indicesOptions=0
    co.verbose=True
    co.shard=True # The replicas will be made "manually"
    t0=time.time()
    logger.info(f"CPU index contains {indexall.ntotal} vectors, move to GPU")
    if replicas==1:
        if not gpu_index:
            logger.info("copying loaded index to GPUs")
            vres, vdev = make_vres_vdev()
            index = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
        else:
            index = gpu_index
    else:
        del gpu_index # We override the GPU index
        logger.info(f"Copy CPU index to {replicas} sharded GPU indexes")
        index=faiss.IndexReplicas()
        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)
            logger.info(f"dispatch to GPUs {gpu0}:{gpu1}")
            index1 = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
            index1.this.disown()
            index.addIndex(index1)
            index.own_fields = True
    del indexall
    logger.info("move to GPU done in %.3f s"%(time.time()-t0))
    
    ps=faiss.GpuParameterSpace()
    ps.initialize(index)
    logger.info("search...")
    nq=X.shape[0]
    ps.set_index_parameter(index, 'nprobe', nprobe)
    t0=time.time()
    if query_batch_size==0:
        D, I=index.search(preproc.apply_py(np.ascontiguousarray(X.astype("float32"))), nnn)
    else:
        I=np.empty((nq, nnn), dtype="int32")
        D=np.empty((nq, nnn), dtype="float32")
        block_ranges=[(i0, min(nb, i0+query_batch_size)) for i0 in range(0, nb, query_batch_size)]
        pbar=tqdm(range(len(block_ranges)))
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            Di, Ii=index.search(xs, nnn)
            I[i0:i1]=Ii
            D[i0:i1]=Di
            pbar.update(1)
    logger.info("search completed in %.3f s"%(time.time()-t0))
    data=np.reshape(D, (-1,))
    row_ind=np.repeat(np.arange(nb), nnn)
    col_ind=np.reshape(I, (-1,))
    sparse_csr = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nb,nb))
    return sparse_csr