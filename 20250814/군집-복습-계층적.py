from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


############################ 계층적 군집 (Hierarchical Clustering) #############################
# 데이터 생성
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=42)

# 계층적 군집
hier = AgglomerativeClustering(n_clusters=3)
labels = hier.fit_predict(X)

# 시각화
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Hierarchical Clustering")
plt.show()