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
    bin_index_bits = np.array(data.dot(random_vectors) >= 0, dtype='int')
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


def norm(x):
    sum_sq = x.dot(x.T)
    n = np.sqrt(sum_sq)
    return n


def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy / (norm(x) * norm(y))
    return 1 - dist[0, 0]


# cosine similarity example

wiki = pd.read_csv('Locality Sensitive Hashing/people_wiki.csv')

obama = wiki[wiki['name'] == 'Barack Obama']
biden = wiki[wiki['name'] == 'Joe Biden']
obama_tf_idf = corpus[obama.index[0], :]
biden_tf_idf = corpus[biden.index[0], :]
print(cosine_distance(obama_tf_idf, biden_tf_idf))  # 0.703

bits_agree = np.array(model['bin_index_bits'][obama.index] == model['bin_index_bits'][biden.index], dtype=int).sum() # 14

obama_sim_index = model['table'][model['bin_indices'][obama.index[0]]]
docs = wiki[wiki.index.isin(obama_sim_index)].reset_index()
docs['sim'] = docs['index'].map(lambda x: cosine_distance(obama_tf_idf, corpus[x, :]) if x != obama.index[0] else 1)


def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    from itertools import combinations
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    candidate_set = copy(initial_candidates)
    for different_bits in combinations(range(num_vector), search_radius):
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = not query_bin_bits[i]
        nearby_bin = alternate_bits.dot(powers_of_two)
        if nearby_bin in table:
            candidate_set |= set(table[nearby_bin])
    return candidate_set


obama_bin_index = model['bin_index_bits'][obama.index[0]]
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1)


def query(vec, model, k, max_search_radius):
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()

    candidate_set = set()
    for search_radius in np.arange(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
        print(len(candidate_set))

    nearest_neighbors = pd.DataFrame({'id': list(candidate_set)})
    candidates = data[np.array(list(candidate_set)), :]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()

    return nearest_neighbors.sort_values('distance', ascending=False).iloc[:k], len(candidate_set)


vec = corpus[obama.index[0], :]
result, num_candidates_considered = query(corpus[35817,:], model, k=10, max_search_radius=3)
kNN = pd.merge(result, wiki[['id', 'name']], how='inner', on='id')