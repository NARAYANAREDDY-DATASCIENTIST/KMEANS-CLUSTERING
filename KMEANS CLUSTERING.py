# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:34:10 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset

dataset=pd.read_csv('Mall_Customers.csv')

x=dataset.iloc[:,[3,4]].values

# FIND THE OPTIMAL NUMBERS OF CLUSTERS USING ELBOW METHOD

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('THE ELBOW METHOD')
plt.xlabel('NUMBER OF CLUSTERS')
plt.ylabel('WCSS')
plt.show()

# Fitting the kmeans modelto the dataset

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)

#VISUALISING THE KMEANS CLUSTERING

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='cyan',label='cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='magenta',label='cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('KMEANS CLUSTERS')
plt.xlabel('ANNUAL INCOME K$ ')
plt.ylabel('SPENDING SCORE 1-100')
plt.legend()
plt.show()





