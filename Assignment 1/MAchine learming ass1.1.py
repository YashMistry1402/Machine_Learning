#!/usr/bin/env python
# coding: utf-8

# In[28]:


from matplotlib import pyplot as plt


# In[29]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[32]:


from numpy import genfromtxt
my_data = genfromtxt('digitData3.csv', delimiter=',')
my_data


# In[43]:


mu = 0.0
sigma = 1.0

X = np.array(my_data)


empiricalMean   = np.mean(X)
empiricalStdDev = np.std(X)

print("empirical mean    = ", empiricalMean)
print("empirical std dev = ", empiricalStdDev)


# In[49]:


kmeans = KMeans(n_clusters=5) #n_clusters define the number of clusters to find
kmeans.fit(X) #


# In[50]:


centroids = kmeans.cluster_centers_
labels    = kmeans.labels_

print(centroids)
print(labels)


# In[52]:


colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


# Visualize the centroids
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=250, linewidths = 5, zorder = 10)
plt.show()


# In[55]:


Question 1.3
If we have an ARI value of 0.7 after a single run of K-means clustering with 'Kmeans++' initializaton 
for any data set then what will be the value of averaged ARI over 20 repeatations. Explain why? 


Ans: if we have the ARI value of 0.7and the average is on the low dimenision of the diagram, there are very
    high chances of the value to also increase if the ARI value is alos increased.
    after about 20 repetitions the value of the averaged ARI will also increase significantly.


# In[ ]:




