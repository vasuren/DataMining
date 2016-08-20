# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:17:01 2016

@author: Vasu
"""

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\winedata.csv')
df1 = df.copy()
del df1['Wine']
df1 = df1.set_index('ID')
pca = PCA()
df_scaled = StandardScaler().fit_transform(df1)
pca.fit(df_scaled)
# Plotting dendrogram using full data
Z = linkage(df_scaled,method = 'ward')
dendrogram(Z)
# Plotting dendrogram using first two princ comp
df_red = PCA(n_components=2).fit_transform(df_scaled)
Zred = linkage(df_red, method = 'ward')
dendrogram(Zred)

# Predicting clusters with cutree using full data
cutreefull = cut_tree(Z, n_clusters=3)
df_scaled.loc[:,'predicted'] = cutreefull

# Predicting clusters with cutree using first two princ comp
cutreered = cut_tree(Zred, n_clusters=3)
df_red = pd.DataFrame(df_red)
df_red.loc[:,'predicted'] = cutreered

#TODO Measuring the accuracy of predictions

#Taking two most significant chemicals
pca1 = PCA().fit(df_scaled) 
princcomp = pd.DataFrame(pca1.components_)
vari = pca1.explained_variance_
princcomp
princcomp.mul(vari, axis=1)

# Correlation matrix
corrmat = pd.DataFrame(np.corrcoef(df1.T))
eig_vals, eig_vecs = np.linalg.eig(corrmat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

princcomp = pd.DataFrame([eig_pairs[i][1] for i in range(len(eig_pairs))])
princcomp.index = df1.columns


# Plotting heatmap for correlation matrix

fig, ax = plt.subplots()
heatmap = ax.pcolor(corrmat,cmap = plt.cm.Greens)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(corrmat.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(corrmat.shape[1])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()
row_labels = df1.columns
column_labels = df1.columns
ax.set_xticklabels(row_labels, minor=False, rotation = 90)
ax.set_yticklabels(column_labels, minor=False)
print (fig)