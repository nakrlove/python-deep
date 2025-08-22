from sklearn.mixture import GaussianMixture

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = load_iris()

gmm = GaussianMixture(n_components = 3, random_state = 0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

gmm_cluster_labels

irisdf = pd.DataFrame(data = iris.data, columns = iris.feature_names)
irisdf['target'] = iris.target


from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 데이터 준비
X, y = load_iris(return_X_y=True)

# PCA로 2차원으로 축소
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 시각화
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Projection of Iris Dataset')
plt.show()