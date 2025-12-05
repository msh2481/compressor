from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn import datasets
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

# X, y = datasets.make_swiss_roll(n_samples=1500, random_state=0)

X = np.random.uniform(0, 1, (1500, 3))
X[:, 2] = X[:, 0]
X[:, 0] *= 7
y = X[:, 0]

if show := False:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=y, s=50, alpha=0.8
    )
    plt.show()
    exit()

# F = RBFSampler(gamma=0.01, random_state=1, n_components=400).fit_transform(X)
# C = PCA(n_components=3).fit_transform(F)

gamma = 0.1
pw = 2.0
distances = np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
# distances = floyd_warshall(distances)
K = np.exp(-gamma * distances**pw)
C = KernelPCA(kernel="precomputed", n_components=3).fit_transform(K)

fig = plt.figure(figsize=(8, 6))
if C.shape[1] == 3:
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=y, s=50, alpha=0.8)
else:
    ax = fig.add_subplot(111)
    ax.scatter(C[:, 0], C[:, 1], c=y, s=50, alpha=0.8)
plt.show()