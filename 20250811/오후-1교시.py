import numpy as np
import pandas as pd
# from sys import os
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
# os.getcwd()
# fruits = np.load("C:\\Users\\Admin\\study01\\20250811\\fruits_300.npy")
fruits = np.load("C:\\Users\\Admin\\study01\\20250811\\fruits_300.npy")
fruits = fruits.reshape(300,10000)

km = KMeans(n_clusters=3)
km.fit(fruits)
km.n_clusters
km.cluster_centers_
len(km.labels_)

km.predict(fruits[[0],:])
km.predict(fruits[[1],:])
km.predict(fruits[[2],:])
km.predict(fruits[[3],:])
km.predict(fruits[[4],:])




from sklearn.datasets import load_iris

iris =  load_iris()
idf = iris['data']
km = KMeans(n_clusters=3)
km.fit(idf)
km.n_clusters
km.cluster_centers_
km.labels_

series = pd.Series(km.labels_)
series.value_counts()

np.unique(km.labels_,return_counts=True)



#====================================
dia = sns.load_dataset('diamonds')

dia.info()
dia = dia.iloc[:,[4,5,6,7,8,9]]

km = KMeans(n_clusters=3)
km.fit(dia)
km.labels_

km.predict([[62.2  ,55.0,   2757,  5.83 , 5.87 , 3.64]])



plt.figure(figsize=(8,5))
plt.plot(k_range, inertias, marker='o', linestyle='-')