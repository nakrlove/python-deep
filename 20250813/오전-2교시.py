from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = load_iris()
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

irisdf = pd.DataFrame(data = iris.data, columns = iris.feature_names)
irisdf['target'] = iris.target

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components = 3, random_state = 0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

# 군집화 결과를 irisdf의 'gmm_cluster' 칼럼명으로 저장
irisdf['gmm_cluster'] = gmm_cluster_labels
irisdf['target'] = iris.target

# target 값에 따라 gmm_cluster 값이 어떻게 매핑됐는지 확인
iris_result = irisdf.groupby(['target'])['gmm_cluster'].value_counts()
iris_result


irisdf.groupby(['target','gmm_cluster']).agg('mean')
irisdf.groupby(['target','gmm_cluster'])['gmm_cluster'].value_counts()

kmeans = KMeans(n_clusters=3, max_iter = 300, random_state = 0).fit(iris.data)
kmeans_cluster_labels = kmeans.predict(iris.data)
irisdf['kmeans_cluster'] = kmeans_cluster_labels
iris_result = irisdf.groupby(['target'])['kmeans_cluster'].value_counts()
iris_result