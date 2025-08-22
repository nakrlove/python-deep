from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 데이터 생성
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=42)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X)

# 시각화
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("DBSCAN (노이즈=-1)")
plt.show()