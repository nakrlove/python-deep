#실루엣개수를 구해봅시다.
import numpy as np
import pandas as pd
# from sys import os
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

fruits = np.load("C:\\Users\\Admin\\study01\\20250811\\fruits_300.npy")
fruits = fruits.reshape(300,10000)
fruits


# fruits[0,0,:]
plt.imshow(fruits[0,0,:],cmap='gray')
plt.show()


km = KMeans(n_clusters=3)
km.fit(fruits)
km.n_clusters
km.cluster_centers_
len(km.labels_)

#실루엣 갯수 구하는 함수
silhouette_score(fruits,km.labels_)



# ==== load_iris ==========
iris = load_iris()
iris = iris['data']

km = KMeans(n_clusters=3)
km.fit(iris)
km.n_clusters
km.cluster_centers_
len(km.labels_)
#실루엣 갯수 구하는 함수
silhouette_score(iris,km.labels_)


plt.figure(figsize=(8,5))
plt.plot(k_range, inertias, marker='o', linestyle='-')