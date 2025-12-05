from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import SpectralEmbedding
from sklearn import datasets

X, y = datasets.make_swiss_roll(n_samples=1500, random_state=0)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)
# ax.scatter(
#     X[:, 0], X[:, 1], X[:, 2], c=y, s=50, alpha=0.8
# )
# plt.show()

# F = RBFSampler(gamma=0.01, random_state=1, n_components=400).fit_transform(X)
# C = PCA(n_components=3).fit_transform(F)

C = KernelPCA(kernel="rbf", gamma=0.01, n_components=2).fit_transform(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=y, s=50, alpha=0.8)
# ax = fig.add_subplot(111)
# ax.scatter(C[:, 0], C[:, 1], c=y, s=50, alpha=0.8)
plt.show()