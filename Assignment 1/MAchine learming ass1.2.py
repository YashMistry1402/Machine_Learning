#!/usr/bin/env python
# coding: utf-8

# In[69]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# In[56]:


from numpy import genfromtxt
data = pd.read_csv('digitData3.csv', delimiter=',', header=None).values
print(data.shape)



# In[57]:


from sklearn.preprocessing import scale
Xnorm = scale(data)


# In[58]:



pca = PCA(n_components=5)
pca.fit(Xnorm)


# In[59]:


var= pca.explained_variance_ratio_
print(var)


# In[61]:


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")


# In[64]:


pca = PCA(n_components=4)
Zred = pca.fit_transform(Xnorm)
print(Zred.shape)


# In[65]:


Xrec = pca.inverse_transform(Zred)
print(Xrec.shape)


# In[66]:


rec_error = np.linalg.norm(Xnorm-Xrec, 'fro')/np.linalg.norm(Xnorm, 'fro')
print(rec_error)


# In[67]:


nSamples, nDims = Xnorm.shape

n_comp = range(1,nDims+1)
print(n_comp)


# In[68]:


rec_error = np.zeros(len(n_comp)+1)

for k in n_comp:
    pca = PCA(n_components=k)
    Zred = pca.fit_transform(Xnorm)
    Xrec = pca.inverse_transform(Zred)
    rec_error[k] = np.linalg.norm(Xnorm-Xrec, 'fro')/np.linalg.norm(Xnorm, 'fro')
    print("k={}, rec_error={}".format(k, rec_error[k]))

rec_error = rec_error[1:] 
plt.plot(n_comp,rec_error)
plt.xlabel('No of principal components (k)')
plt.ylabel('Reconstruction Error')


# In[ ]:




