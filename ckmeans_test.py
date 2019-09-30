from collections import defaultdict
from functools import partial

import numpy as np
from numpy import linalg as LA
from scipy.fftpack import fft, ifft
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


def ckmeans(data, n_clusters, max_iterations, seed=0, rtol=1e-5):
    # rtol for centers convergence, can be set as 1e-4
    #total_iteration = 0
    label_convergence = []
    labels = np.empty(data.shape[0])
    centers_shift = []

    centroids = init_centroids(data, n_clusters)
    #while total_iterations < max_iterations:
    for total_iterations in tqdm(range(max_iterations)):
        previous_centroids = centroids.copy()
        previous_labels = labels.copy()

        #labels, centroids = perform_iteration(data, centroids)
        labels, centroids = perform_iteration_multi(data, centroids)

        label_convergence.append(sum(np.array(previous_labels) != np.array(labels)))

        #if np.allclose(previous_centroids, centroids, rtol):
        #    print('centroids the same')
        #    break
        c_shift = cal_centers_shift(previous_centroids, centroids)
        centers_shift.append(c_shift)
        if c_shift <= rtol:
            print('centers close')
            break

        total_iterations += 1

    return labels, label_convergence, centroids, centers_shift

def ckmeans_predict(data, centroids):
    # cluster test data using sample centroids
    func = partial(assign_to_cluster, centroids=centroids)
    results = Pool().map(func, data)

    labels = np.empty(data.shape[0])
    for point_idx, (cluster_idx, point_shift) in enumerate(results):
        labels[point_idx] = cluster_idx

    return labels


def compute_distance(point, centroid):
    angbin = 8
    radbin = 3
    r_m = np.empty((radbin, angbin))
    point_m = point.reshape(radbin, angbin)
    center_m = centroid.reshape(radbin, angbin)
    for i in range(radbin):
        p = point_m[i, :]
        c = center_m[i, :]
        r = ifft(fft(c) * fft(p).conj()).real
        r_m[i, :] = r
    
    r_norm = LA.norm(r_m, axis=0)
    rotation_index = np.argmax(r_norm)
    r_max = np.mean(r_m[:, rotation_index])
    #r_max = np.max(r_m[:, rotation_index])
    
    '''
    r = ifft(fft(centroid) * fft(point).conj()).real
    r_max = max(r)
    rotation_index = np.argmax(r)
    '''
    distance = -2 * r_max
    distance += np.dot(point, point)
    distance += np.dot(centroid, centroid)

    return distance, rotation_index


def assign_to_cluster(point, centroids):
    angbin = 8
    radbin = 3
    distances, rotation_indices = zip(
        *[compute_distance(point, centroid) for centroid in centroids]
    )
    closest_cluster = np.argmin(distances)
    rotation_index = rotation_indices[closest_cluster]
    '''
    point_shift = np.roll(point, rotation_index)
    point_shift = point_shift.flatten()
    '''
    point_m = point.reshape(radbin, angbin)
    point_shift = np.zeros((radbin,angbin))
    # I tried just rotate the point together too, but the clustering results the same
    # I feel rotating the point by each radius makes more sense 
    for i in range(radbin):
        point_shift[i, :] = np.roll(point_m[i, :], rotation_index)
    point_shift = point_shift.flatten()

    return closest_cluster, point_shift


def init_centroids(data, n_clusters, seed=0):
    k = n_clusters
    n_samples = data.shape[0]
    random_state = np.random.RandomState(seed)
    seeds = random_state.permutation(n_samples)[:k]
    centers = data[seeds]

    return np.array(centers)

def init_centroids_plus(data, n_clusters, seed=0):
    # selects initial cluster centers in a smart way to speed up converge
    # see "k-means++: the advantages of careful seeding"

    n_samples, n_features = data.shape
    centers = np.empty((n_clusters, n_features), dtype=data.dtype)
    n_local_trials = 2 + int(np.log(n_clusters))

    # initial first center randomly
    random_state = np.random.RandomState(seed)
    seeds = random_state.randint(n_samples)
    centers[0] = data[seeds]
    
    closest_dist = []
    for i in range(n_samples):
        dist, temp = compute_distance(data[i], centers[0])
        closest_dist.append(dist)
    closest_dist = np.array(closest_dist)
    current_pot = closest_dist.sum()

    # pick the remaining n-1 points
    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_dist), rand_vals)
        #np.clip(candidate_ids, None, closest_dist.size - 1, out=candidate_ids)
        np.clip(candidate_ids, None, n_samples - 1, out=candidate_ids)

        dist_to_candidate = np.empty((len(candidate_ids), n_samples))
        for i in range(len(candidate_ids)):
            ids = candidate_ids[i]
            for j in range(n_samples):
                dist, temp = compute_distance(data[j], data[ids])
                dist_to_candidate[i][j] = dist
        
        np.minimum(closest_dist, dist_to_candidate, out=dist_to_candidate)
        candidate_pot = dist_to_candidate.sum(axis=1)

        best_candidate = np.argmin(candidate_pot)
        current_pot = candidate_pot[best_candidate]
        closest_dist = dist_to_candidate[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = data[best_candidate]

    return centers



def perform_iteration(data, centroids):
    labels = np.empty(data.shape[0])
    cluster_points = defaultdict(list)
    # First assign each point to the nearest cluster
    for point_idx, point in enumerate(data):
        cluster_idx, point_shift = assign_to_cluster(point, centroids)
        cluster_points[cluster_idx].append(point_shift)
        labels[point_idx] = cluster_idx

    # Now update the centroids based on the new cluster assignments
    for cluster_idx, shifted_points in cluster_points.items():
        centroids[cluster_idx] = np.mean(shifted_points, axis=0)

    return labels, centroids


def perform_iteration_multi(data, centroids):
    # First assign each point to the nearest cluster mapping points on pool of processes
    func = partial(assign_to_cluster, centroids=centroids)
    results = Pool().map(func, data)

    labels = np.empty(data.shape[0])
    cluster_points = defaultdict(list)
    for point_idx, (cluster_idx, point_shift) in enumerate(results):
        cluster_points[cluster_idx].append(point_shift)
        labels[point_idx] = cluster_idx

    # Now update the centroids based on the new cluster assignments
    for cluster_idx, shifted_points in cluster_points.items():
        centroids[cluster_idx] = np.mean(shifted_points, axis=0)

    return labels, centroids

def cal_centers_shift(previous_centers, centers):
    shift = np.array(previous_centers) - np.array(centers)
    shift_total = np.ravel(shift, order='K')
    
    return np.dot(shift_total, shift_total)


def evaluate_ckmeans(data, label, center, n_clusters):
    # evaluate results using distance
    # elbow point

    D = np.zeros(n_clusters)
    for i in range(len(label)):
        k = label[i]
        c = center[int(k)]
        d, r = compute_distance(data[i], c)
        D[int(k)] += d

    return sum(D)

