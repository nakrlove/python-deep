from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns


X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)


X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

#에 대하여 KMeans를 사용하여 2개의 군집으로 나누고 시각화하는 코드