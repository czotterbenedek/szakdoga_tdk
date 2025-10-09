import pandas as pd
import numpy as np



class OnlineKMeans:
    def __init__(
        self,
        n_clusters=4,
        max_clusters=20,
        metric="euclidean",   # "euclidean" or "cosine"
        new_cluster_threshold=None, # if None, no dynamic creation
        merge_threshold=None,  # if not None, merge centroids closer than this
        decay=1.0,             # multiply counts by this factor periodically (<=1.0)
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.metric = metric
        self.new_cluster_threshold = new_cluster_threshold
        self.merge_threshold = merge_threshold
        self.decay = decay
        self.rng = np.random.RandomState(random_state)

        self.centroids = None     # (k, d)
        self.counts = None        # (k,)
        self.sums = None          # (k, d) running sums (optional convenience)
        self.vars = None          # per-cluster variance estimate (scalar)
        self.total_seen = 0

    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _pairwise_dist(self, X, C):
        # returns (n_points, n_centroids) distances
        if self.metric == "euclidean":
            # squared distance then sqrt
            d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
            return np.sqrt(d2 + 1e-12)
        else:  # cosine distance = 1 - dot(normalized)
            Xn = self._normalize(X)
            Cn = self._normalize(C)
            return 1.0 - (Xn @ Cn.T)

    def _kmeans_pp_init(self, X, k):
        # returns k centers chosen from X (indices) by kmeans++
        n = X.shape[0]
        centers = []
        first = self.rng.randint(0, n)
        centers.append(first)
        d2 = np.full(n, np.inf)
        for _ in range(1, k):
            cur = X[centers[-1 :], :]
            dist = np.sum((X - cur) ** 2, axis=1)
            d2 = np.minimum(d2, dist)
            probs = d2 / (d2.sum() + 1e-12)
            next_idx = self.rng.choice(n, p=probs)
            centers.append(int(next_idx))
        return np.array(centers, dtype=int)

    def initialize_centroids(self, X_init):
        # initialize centroids using kmeans++ on X_init (or random fallback)
        k = self.n_clusters
        if X_init.shape[0] < k:
            # fallback: random pick with replacement
            idx = self.rng.choice(len(X_init), k, replace=True)
            centers = X_init[idx]
        else:
            idx = self._kmeans_pp_init(X_init, k)
            centers = X_init[idx]
        if self.metric == "cosine":
            centers = self._normalize(centers)
        self.centroids = centers.copy()
        self.counts = np.zeros(len(self.centroids), dtype=float)
        self.sums = np.zeros_like(self.centroids)
        self.vars = np.zeros(len(self.centroids), dtype=float)
        self.total_seen = 0

    def partial_fit(self, X_batch):
        """
        Vectorized minibatch update:
         - assign points to nearest centroid
         - for each cluster, compute batch sum and batch count
         - do exact incremental mean update using counts
        Also can create new clusters for far-away points (optional).
        """
        X = np.asarray(X_batch, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.metric == "cosine":
            X = self._normalize(X)

        if self.centroids is None:
            # init from first batch
            self.initialize_centroids(X)
            # if we used kmeans++ the centroids are populated but counts=0

        # compute pairwise distances
        D = self._pairwise_dist(X, self.centroids)
        min_dist = D.min(axis=1)
        labels = D.argmin(axis=1)

        # dynamic creation of new clusters for far-away points
        if (self.new_cluster_threshold is not None) and (len(self.centroids) < self.max_clusters):
            far_idx = np.where(min_dist > self.new_cluster_threshold)[0]
            # create up to capacity new clusters from these points
            to_create = min(len(far_idx), self.max_clusters - len(self.centroids))
            for j in range(to_create):
                i = far_idx[j]
                new_center = X[i].copy()
                if self.metric == "cosine":
                    new_center = self._normalize(new_center.reshape(1, -1))[0]
                # append new centroid
                self.centroids = np.vstack([self.centroids, new_center])
                self.counts = np.concatenate([self.counts, np.array([0.0])])
                self.sums = np.vstack([self.sums, np.zeros_like(new_center)])
                self.vars = np.concatenate([self.vars, np.array([0.0])])
                # reassign label of this point to the new cluster
                labels[i] = len(self.centroids) - 1

            # recompute distances/labels if centers changed (safe)
            D = self._pairwise_dist(X, self.centroids)
            min_dist = D.min(axis=1)
            labels = D.argmin(axis=1)

        # vectorized update: for each cluster compute batch_sum and m
        k_now = len(self.centroids)
        for k in range(k_now):
            mask = labels == k
            m = mask.sum()
            if m == 0:
                continue
            batch_points = X[mask]
            batch_sum = batch_points.sum(axis=0)
            batch_mean = batch_sum / m

            # incremental mean update using counts
            n_old = self.counts[k]
            if n_old == 0:
                # pure initialization for this cluster
                self.centroids[k] = batch_mean
                self.counts[k] = m
                self.sums[k] = batch_sum
                # compute variance (scalar average squared distance)
                diffs = batch_points - batch_mean
                self.vars[k] = np.mean(np.sum(diffs ** 2, axis=1))
            else:
                new_count = n_old + m
                # update centroid to combined mean:
                new_centroid = (n_old * self.centroids[k] + batch_sum) / new_count
                # update variance with combined formula:
                # var_new = (n_old*(var_old + (mu_old - mu_new)^2) + m*(var_batch + (mu_batch-mu_new)^2)) / new_count
                var_old = self.vars[k]
                mu_old = self.centroids[k]
                mu_batch = batch_mean
                # batch variance
                diffs = batch_points - mu_batch
                var_batch = np.mean(np.sum(diffs ** 2, axis=1))
                delta_old = mu_old - new_centroid
                delta_batch = mu_batch - new_centroid
                var_new = (n_old * (var_old + np.sum(delta_old ** 2)) + m * (var_batch + np.sum(delta_batch ** 2))) / new_count

                # commit updates
                self.centroids[k] = new_centroid
                self.counts[k] = new_count
                self.sums[k] += batch_sum
                self.vars[k] = var_new

        self.total_seen += len(X)

        # renormalize centroids for cosine
        if self.metric == "cosine":
            self.centroids = self._normalize(self.centroids)

        # optional: merge very close clusters
        if (self.merge_threshold is not None) and (len(self.centroids) > 1):
            self._merge_close_clusters()

    def _merge_close_clusters(self):
        # merges centroids that are closer than merge_threshold
        C = self.centroids
        k = len(C)
        D = self._pairwise_dist(C, C)  # symmetric
        np.fill_diagonal(D, np.inf)
        merge_pairs = np.argwhere(D < self.merge_threshold)
        if len(merge_pairs) == 0:
            return
        to_remove = set()
        for i, j in merge_pairs:
            if i in to_remove or j in to_remove:
                continue
            # merge j into i (weighted)
            n_i = self.counts[i]
            n_j = self.counts[j]
            total = n_i + n_j if (n_i + n_j) > 0 else 1.0
            new_centroid = (n_i * self.centroids[i] + n_j * self.centroids[j]) / total
            self.centroids[i] = new_centroid
            # update counts, sums, variance
            self.counts[i] = total
            self.sums[i] = self.sums[i] + self.sums[j]
            self.vars[i] = (self.vars[i] + self.vars[j]) / 2.0
            to_remove.add(j)
        if to_remove:
            keep = [idx for idx in range(len(self.centroids)) if idx not in to_remove]
            self.centroids = self.centroids[keep]
            self.counts = self.counts[keep]
            self.sums = self.sums[keep]
            self.vars = self.vars[keep]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.metric == "cosine":
            X = self._normalize(X)
        D = self._pairwise_dist(X, self.centroids)
        return D.argmin(axis=1)

    def get_state(self):
        return {
            "centroids": self.centroids.copy(),
            "counts": self.counts.copy(),
            "vars": self.vars.copy(),
            "total_seen": self.total_seen,
        }