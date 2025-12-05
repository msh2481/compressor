from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import datasets
import numpy as np


class KMeansCompressor:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.probabilities_ = None

    def fit(self, X: np.ndarray):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        unique, counts = np.unique(labels, return_counts=True)
        self.probabilities_ = np.zeros(self.n_clusters)
        self.probabilities_[unique] = counts / len(labels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        cluster_indices = np.random.choice(
            self.n_clusters, size=n_samples, p=self.probabilities_
        )
        return self.cluster_centers_[cluster_indices].flatten()


class NormalCompressor:
    def __init__(self, n_components: int | None, n_clusters: int):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=n_components)
        self.compressors_ = None

    def fit(self, X: np.ndarray):
        X_transformed = self.pca.fit_transform(X)
        n_features = X_transformed.shape[1]
        self.compressors_ = [
            KMeansCompressor(n_clusters=self.n_clusters) for _ in range(n_features)
        ]
        for i, compressor in enumerate(self.compressors_):
            compressor.fit(X_transformed[:, i])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_transformed = self.pca.transform(X)
        n_samples = X_transformed.shape[0]
        n_features = X_transformed.shape[1]
        result = np.zeros((n_samples, n_features), dtype=int)
        for i, compressor in enumerate(self.compressors_):
            result[:, i] = compressor.transform(X_transformed[:, i])
        return result

    def sample(self, n_samples: int = 1) -> np.ndarray:
        n_features = len(self.compressors_)
        samples = np.zeros((n_samples, n_features))
        for i, compressor in enumerate(self.compressors_):
            samples[:, i] = compressor.sample(n_samples)
        return self.pca.inverse_transform(samples)


class GMMCompressor:
    def __init__(self, n_components: int, n_pca_components: int | None, n_clusters: int):
        self.n_components = n_components
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        self.compressors_ = None
        self.probabilities_ = None

    def fit(self, X: np.ndarray):
        self.gmm.fit(X)
        labels = self.gmm.predict(X)
        unique, counts = np.unique(labels, return_counts=True)
        self.probabilities_ = np.zeros(self.n_components)
        self.probabilities_[unique] = counts / len(labels)
        self.compressors_ = [
            NormalCompressor(n_components=self.n_pca_components, n_clusters=self.n_clusters)
            for _ in range(self.n_components)
        ]
        for i, compressor in enumerate(self.compressors_):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                compressor.fit(X[cluster_mask])
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cluster_indices = self.gmm.predict(X)
        n_samples = X.shape[0]
        inside_coords = np.zeros((n_samples, len(self.compressors_[0].transform(X[:1]))))
        for i in range(self.n_components):
            cluster_mask = cluster_indices == i
            if np.any(cluster_mask):
                inside_coords[cluster_mask] = self.compressors_[i].transform(X[cluster_mask])
        return cluster_indices, inside_coords

    def sample(self, n_samples: int = 1) -> np.ndarray:
        cluster_indices = np.random.choice(
            self.n_components, size=n_samples, p=self.probabilities_
        )
        samples_list = []
        for i in range(self.n_components):
            cluster_mask = cluster_indices == i
            if np.any(cluster_mask):
                samples_list.append(self.compressors_[i].sample(np.sum(cluster_mask)))
        samples = np.vstack(samples_list)
        return samples[np.argsort(np.concatenate([np.where(cluster_indices == i)[0] for i in range(self.n_components)]))]


X, y = datasets.make_swiss_roll(n_samples=1500, random_state=0)

if show := False:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=y, s=50, alpha=0.8
    )
    plt.show()
    exit()

compressor = GMMCompressor(n_components=16, n_pca_components=3, n_clusters=10)
compressor.fit(X)
S = compressor.sample(1500)
S += np.random.randn(1500, 3) * 0.01

if show_samples := True:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=10, alpha=0.5)
    plt.show()
    exit()