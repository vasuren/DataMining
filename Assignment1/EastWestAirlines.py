# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:47:52 2016

@author: Vasu
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\EastWestAirlinesCluster.csv')
df.set_index('ID#')
df_norm = (df - df.mean())/df.std()
#df_norm = df.copy()
Z = linkage(df_norm, 'ward')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
b = dendrogram(Z, p=2, truncate_mode = 'lastp')
nclust = 2
cutree = cut_tree(Z, n_clusters=nclust)
df.loc[:,'ClusterID'] = cutree
dict1 = {}
for i in range(nclust):
    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
print (listmean)
##################Part b#######################
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\EastWestAirlinesCluster.csv')
df.set_index('ID#')
#df_norm = (df - df.mean())/df.std()
df_norm = df.copy()
Z = linkage(df_norm, 'ward')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
b = dendrogram(Z, p=2, truncate_mode = 'lastp')
nclust = 2
cutree = cut_tree(Z, n_clusters=nclust)
df.loc[:,'ClusterID'] = cutree
dict1 = {}
for i in range(nclust):
    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
print (listmean)
#######################Part d#################
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\EastWestAirlinesCluster.csv')
retain = np.random.randint(0,3999,3800)
df.set_index('ID#')
df_norm = (df - df.mean())/df.std()
retain = np.random.randint(0,3999,3800)
df_norm = df_norm.loc[retain,:]
#df_norm = df.copy()
Z = linkage(df_norm, 'ward')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
b = dendrogram(Z, p=2, truncate_mode = 'lastp')
nclust = 2
cutree = cut_tree(Z, n_clusters=nclust)
df = df.loc[retain,:]
df.loc[:,'ClusterID'] = cutree
dict1 = {}
for i in range(nclust):
    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
print (listmean)

###############K-Means Clustering##################
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=2, n_init=100, random_state=1,init='random',\
    precompute_distances = False, verbose=True)
    
KM.fit(df_norm)
listmean = [KM.cluster_centers_[i]*df.std() + df.mean() for i in range(nclust)]
