# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:47:52 2016

@author: Vasu
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def patentanalyse(df,df_cite):
    df_cite = df_cite.groupby('cited')
    srs_citing_sic = df_cite.citing_sic.nunique()
    df_cite = df_cite.aggregate("count")
    df_cite.loc[:,'citing_sic'] = srs_citing_sic
    
    
    df_cite = df_cite.loc[:,['citing_sic','citing','citing_icl']]
    df_cite = df_cite.reset_index()
    df = pd.merge(left = df, right = df_cite, how = 'left', left_on = 'patent', right_on = 'cited')
    
    
    df_sic = df.groupby(['sic'])
    npat = df_sic.patent.aggregate("count")
    df_sic = df_sic.aggregate(np.nanmean)
    df_sic.loc[:,'Number of Patents'] = npat
    
    df_sic = df_sic.reset_index()
    df_sic_code = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Project\SIC_data.csv')
    df_sic = pd.merge(df_sic,df_sic_code,how = 'left',left_on = 'sic', right_on = 'SIC')
    
    cols_to_check_nan = df_sic.columns[:-7]
    df_sic = df_sic.dropna(subset=cols_to_check_nan)
    
    rdr_1 = df_sic.xrd_1/df_sic.sale_1
    rdr0 = df_sic.xrd0/df_sic.sale0
    rdr1 = df_sic.xrd1/df_sic.sale1
    df_sic.loc[:,'rdr_1'] = rdr_1
    df_sic.loc[:,'rdr0'] = rdr0
    df_sic.loc[:,'rdr1'] = rdr1
    
    
    cols_to_keep = df_sic.columns[-3:]
    df_sic_keep = df_sic.loc[:,cols_to_keep]
    df_norm = (df_sic_keep - df_sic_keep.mean())/df_sic_keep.std()
    df_sic.loc[:,cols_to_keep] = df_norm
    df_norm = df_norm.dropna()
    
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
    b = dendrogram(Z, p=3, truncate_mode = 'lastp')
    
    nclust = 3
    df_norm = (df_sic_keep - df_sic_keep.mean())/df_sic_keep.std()
    df_norm = df_norm.dropna()
    df_clust_centroid = pd.DataFrame()
    KM = KMeans(n_clusters=nclust, n_init=100, random_state=1,init='random',\
        precompute_distances = False, verbose=True)
    KM.fit(df_norm)
    df_clust_centroid = pd.DataFrame(KM.cluster_centers_,columns = cols_to_keep)
    clust_size = [sum(KM.labels_ == i) for i in range(nclust)]
    df_norm.loc[:,'Cluster_ID'] = KM.labels_
    df_sic = df_sic.loc[df_norm.index,:]
    df_sic.loc[:,'Cluster_ID'] = KM.labels_
    sales_1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'sale_1']) for i in range(nclust)]
    sales0 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'sale0']) for i in range(nclust)]
    sales1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'sale1']) for i in range(nclust)]
    
    income_1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'ni_1']) for i in range(nclust)]
    income0 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'ni0']) for i in range(nclust)]
    income1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'ni1']) for i in range(nclust)]

    margin_1 = [income_1[i]/sales_1[i] for i in range(len(income_1))]
    margin0 = [income0[i]/sales0[i] for i in range(len(income0))]
    margin1 = [income1[i]/sales1[i] for i in range(len(income1))]
    
    return [margin_1, margin0, margin1, df_sic, rdr_1, rdr0, rdr1]


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Project\Patents_2004.csv')
    df_cite = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Project\citations_2004.csv')
    resultlist = patentanalyse(df, df_cite)

#
#    citationratio = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'citing_sic']) for i in range(nclust)]
#    citation1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'citing']) for i in range(nclust)]
#    df_clust_centroid.loc[:,'Citation_Ratio'] = citationratio
#    clust_size = [sum(KM.labels_ == i) for i in range(nclust)]
#    df_clust_centroid.loc[:,'Cluster_Size'] = clust_size
#    citation1 = [np.mean(df_sic.loc[df_norm.Cluster_ID == i,'citing']) for i in range(nclust)]
#    df_clust_centroid.loc[:,'Citation'] = citation1
#    return 




#'''
#
#nclust = 2
#cutree = cut_tree(Z, n_clusters=nclust)
#df.loc[:,'ClusterID'] = cutree
#dict1 = {}
#for i in range(nclust):
#    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
#listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
#listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
#print (listmean)
###################Part b#######################
#from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
#import numpy as np
#import pandas as pd
#
#df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\EastWestAirlinesCluster.csv')
#df.set_index('ID#')
##df_norm = (df - df.mean())/df.std()
#df_norm = df.copy()
#Z = linkage(df_norm, 'ward')
#
#plt.figure(figsize=(25, 10))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
#dendrogram(
#    Z,
#    leaf_rotation=90.,  # rotates the x axis labels
#    leaf_font_size=8.,  # font size for the x axis labels
#)
#b = dendrogram(Z, p=2, truncate_mode = 'lastp')
#nclust = 2
#cutree = cut_tree(Z, n_clusters=nclust)
#df.loc[:,'ClusterID'] = cutree
#dict1 = {}
#for i in range(nclust):
#    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
#listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
#listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
#print (listmean)
########################Part d#################
#from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
#import numpy as np
#import pandas as pd
#df = pd.read_csv(r'C:\Users\Vasu\Business Analytics\Datamining1\Assignment\Assignment1\EastWestAirlinesCluster.csv')
#retain = np.random.randint(0,3999,3800)
#df.set_index('ID#')
#df_norm = (df - df.mean())/df.std()
#retain = np.random.randint(0,3999,3800)
#df_norm = df_norm.loc[retain,:]
##df_norm = df.copy()
#Z = linkage(df_norm, 'ward')
#
#plt.figure(figsize=(25, 10))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.xlabel('sample index')
#plt.ylabel('distance')
#dendrogram(
#    Z,
#    leaf_rotation=90.,  # rotates the x axis labels
#    leaf_font_size=8.,  # font size for the x axis labels
#)
#b = dendrogram(Z, p=2, truncate_mode = 'lastp')
#nclust = 2
#cutree = cut_tree(Z, n_clusters=nclust)
#df = df.loc[retain,:]
#df.loc[:,'ClusterID'] = cutree
#dict1 = {}
#for i in range(nclust):
#    dict1["df_{0}".format(i)] = df[df.loc[:,'ClusterID'] == i]
#listcount = [dict1["df_{0}".format(i)].count()[1] for i in range(nclust)]
#listmean = [dict1["df_{0}".format(i)].mean() for i in range(nclust)]
#print (listmean)
#
################K-Means Clustering##################
#from sklearn.cluster import KMeans
#KM = KMeans(n_clusters=2, n_init=100, random_state=1,init='random',\
#    precompute_distances = False, verbose=True)
#    
#KM.fit(df_norm)
#listmean = [KM.cluster_centers_[i]*df.std() + df.mean() for i in range(nclust)]