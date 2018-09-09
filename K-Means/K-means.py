import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import sys
import os
import json


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


wiki = pd.read_csv('K-Means/people_wiki.csv')

tf_idf = load_sparse_csr('K-Means/people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)

with open('K-Means/people_wiki_map_index_to_word.json', 'r') as read_file:
    map_index_to_word = json.load(read_file)


def get_initial_centroids(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = data.shape[0]

    rand_indices = np.random.randint(0, n, k)
    centroids = data[rand_indices, :].toarray()
    return centroids


def assign_clusters(data, centroid):
    distances_from_centroids = pairwise_distances(data, centroid, metric='euclidean')
    cluster_assignment = np.array([list(row).index(min(row)) for row in distances_from_centroids])
    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in np.arange(k):
        member_data_points = data[cluster_assignment == i]
        centroid = np.mean(member_data_points, axis=0)
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in np.arange(k):
        member_data_points = data[cluster_assignment == i, :]

        if member_data_points.shape[0] > 0:
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)
    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    """
    This function runs k-means on given data and initial set of centroids.


    :param data: observations
    :param k: number of clusters
    :param initial_centroids: k centroids to start with
    :param maxiter: maximum number of iterations to run.
    :param record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations
                             if None, do not store the history.
    :param verbose: if True, print how many data points changed their cluster labels in each iteration
    :return:
    """

    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in np.arange(maxiter):
        if verbose:
            print(itr)

        cluster_assignment = assign_clusters(data, centroids)
        centroids = revise_centroids(data, k, cluster_assignment)

        if prev_cluster_assignment is not None and \
                (prev_cluster_assignment == cluster_assignment).all():
            break

        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))

        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7, 4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=heterogeneity, verbose=True)
plot_heterogeneity(heterogeneity, k)