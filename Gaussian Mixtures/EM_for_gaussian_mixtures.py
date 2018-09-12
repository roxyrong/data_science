import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy

from PIL import Image
from io import BytesIO
import glob


def pull_image_data():
    image_list = []
    for filename in glob.glob('Gaussian Mixtures/images/*/*.jpg'):
        im = np.array(Image.open(filename).getdata())
        rgb = [np.mean(im[:, 0] / 256.0), np.mean(im[:, 1] / 256.0), np.mean(im[:, 2] / 256.0)]
        image_list.append(rgb)
    return image_list


def log_sum_exp(Z):
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
    num_clusters = len(means)
    num_dim = len(data[0])

    ll = 0
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            Z[k] += np.log(weights[k])
            Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
        ll += log_sum_exp(Z)
    return ll


def compute_responsibilities(data, weights, means, covariances):
    num_data = len(data)
    num_clusters = len(means)
    resp = np.zeros((num_data, num_clusters))

    for i in range(num_data):
        for k in range(num_clusters):
            resp[i, k] = weights[k] * multivariate_normal.pdf(data[i], means[k], covariances[k])

    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums
    return resp


def compute_soft_counts(resp):
    counts = np.sum(resp, axis=0)
    return counts


def compute_weights(counts):
    num_clusters = len(counts)
    weights = [0.] * num_clusters
    for k in range(num_clusters):
        weights[k] = counts[k] / counts.sum()
    return weights


def compute_means(data, resp, counts):
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters

    for k in range(num_clusters):
        weighted_sum = 0
        for i in range(num_data):
            weighted_sum += resp[i, k] * data[i]
        means[k] = weighted_sum / counts[k]
    return means


def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim, num_dim))] * num_clusters

    for k in range(num_clusters):
        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            weighted_sum += resp[i, k] * np.outer(data[i] - means[k], data[i] - means[k])
        covariances[k] = weighted_sum / counts[k]

    return covariances


def em(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)

    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]

    for it in range(maxiter):
        if it % 5 == 0:
            print("Iteration %s" % it)

        # E-step: compute responsibilities
        resp = compute_responsibilities(data, weights, means, covariances)
        # M-step: compute the total responsibility assigned to each cluster
        counts = compute_soft_counts(resp)
        # Update the weight for cluster
        weights = compute_weights(counts)
        # Update means for cluster
        means = compute_means(data, resp, counts)
        # Update covariances for cluster
        covariances = compute_covariances(data, resp, counts, means)
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest

    if it % 5 != 0:
        print("Iteration %s" % it)
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}
    return out

