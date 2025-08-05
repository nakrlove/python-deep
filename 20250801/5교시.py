from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#시그모이드 Sigmoid 란?
#시그모이드 수식  0 < y < 1 사이
# 최대값,최소값이 있다
e = 2.718281828

x = np.linspace(-7,7,1000)
y = np.exp(x) / (1 + np.exp(x))
np.exp(2)

plt.scatter(x,y)
