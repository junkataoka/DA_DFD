"""Kernel K-means"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

import os
import time
import torch
import gc


def kernel_k_means_wrapper(target_features, target_targets, 
                   pseudo_labels, epoch, args, best_prec):

    # define kernel k-means clustering
    kkm = KernelKMeans(n_clusters=args.num_classes, max_iter=10, 
                       random_state=0, kernel="rbf", 
                       gamma=None, verbose=1)
    kkm.fit(np.array(target_features.cpu()), 
            initial_label=np.array(pseudo_labels.cpu()), 
            true_label=np.array(target_targets.cpu()), args=args, epoch=epoch)

    idx_sim = torch.from_numpy(kkm.labels_).cuda()
    c_tar = torch.cuda.FloatTensor(args.num_classes, target_features.size(1)).fill_(0)
    count = torch.cuda.FloatTensor(args.num_classes, 1).fill_(0)
    for i in range(target_targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
    c_tar /= (count + 1e-6)

    prec1 = kkm.prec1_
    is_best = prec1 > best_prec
    if is_best:
        best_prec = prec1

    gc.collect()
    torch.cuda.empty_cache()

    # return cluster center and best precision and prediction from k-means
    return c_tar, idx_sim, best_prec

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=10, tol=1e-4, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None, initial_label=None, true_label=None, args=None, epoch=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight is not None else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = initial_label if initial_label is not None else rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        end = time.time()
        it = 0
        while it < self.max_iter:

            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the accuracy of clustering
            if true_label is not None:
                prec1 = 100 * ((self.labels_ == true_label).sum()) / n_samples
                self.prec1_ = prec1
                cluster_time = time.time() - end
                end = time.time()
                print('Epoch %d - Kernel K-means clustering %d: Clustering time %.3f, Prec@1 %.3f' % (epoch, it, cluster_time, prec1))
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('\nEpoch %d - Kernel K-means clustering %d: Clustering time %.3f, Prec@1 %.3f' % (epoch, it, cluster_time, prec1))
                log.close()

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)

            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                #self.labels_ = labels_old
                break
            else:
                it += 1

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                #raise ValueError("Empty cluster found, try smaller n_cluster.")
                continue

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False) #why False???
        return dist.argmin(axis=1)
'''
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=1000, centers=5, random_state=0)

    kkm = KernelKMeans(n_clusters=5, max_iter=100, random_state=0, verbose=1)
    #ipdb.set_trace()
    print(kkm.fit_predict(X)[10:20])
    print(kkm.predict(X[10:20]))
'''


