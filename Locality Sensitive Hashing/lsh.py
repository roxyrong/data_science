import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from copy import copy
import matplotlib.pyplot as plt


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def train_lsh(data, num_vector=16, seed=None):
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = np.random.rand(num_vector, dim)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    bin_index_bits = (data.dot(random_vectors) >= 0)
    bin_indices = bin_index_bits.dot(powers_of_two)
    table = {}
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            table[bin_index] = []
        else:
            table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model


corpus = load_sparse_csr('Locality Sensitive Hashing/people_wiki_tf_idf.npz')
model = train_lsh(corpus, 16, 143)
table = model['table']