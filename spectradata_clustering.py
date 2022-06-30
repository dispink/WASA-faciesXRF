# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:08:24 2019

@author: An-Sheng
This is the workflow try to cluster the scanning points based on the spectra. 
Three clustering algorithms are tried: Kmean, HDBSCA, and agglomerative clustering (Ward).
HDBSCAN provides best clustering reslt that corresponds to the data distribbution in the first 3 PCs space.
Apart from the main cluster, the spectra reveal that the rest clusters represent the XRF scanning using Mo tube instead of Cr tube.
Therefore, only the data belong to the main cluster are remained.  
"""
################## Import Packages #################### 
import time
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import hdbscan
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
####################################################### 


###### Set the working directory ######
path = '\\\\10.110.16.10\\GeopolarLabor\\#Projekte\\WASA\\XRF\\data_separated'
os.chdir(path)

date = time.strftime('%Y%m%d', time.localtime())
#######################################


##### reduce data dimensions: PCA #####

spe_df = pd.read_csv('..\\data_composite\\WASA_all_xrf_spe_20190405.csv').iloc[:, 0:1024]

#### minimize the dataset ####
cols_sel = []

for col in range(len(spe_df.columns) - 2):
    if spe_df.iloc[:, col].max() != 0:
        cols_sel.append(col)
        
X = spe_df.iloc[:, cols_sel].values

del spe_df

##### PCA without clr-transformation #####
# no clr means the variance caused by machine, matrix properties won't be eliminated
# in other words, later cluster analysis takes those variance in account

# standardization
X_std = StandardScaler().fit_transform(X)                
del X

# PCA
pca = PCA(n_components = 'mle', svd_solver = 'auto')
pca.fit(X_std)
n_components = len(
        pca.explained_variance_ratio_[pca.explained_variance_ratio_ > 0.01]
        )
pca = PCA(n_components = n_components, svd_solver = 'auto')
PCs_df = pd.DataFrame(
       pca.fit_transform(X_std),
        columns = ['PC{}'.format(_) for _ in range(1, len(pca.explained_variance_) + 1)]
      )

print(
      '{:0.2f}% of variance is preserved.'.format(100 * pca.explained_variance_ratio_.sum())
      )

del X_std

PCs_df['spe_dir'] = pd.read_csv('..\\data_composite\\WASA_all_xrf_spe_20190405.csv', usecols = ['spe_dir'])
PCs_df.to_csv('pca_reduced_dimensions_{}.csv'.format(date), index = False)

### Ward's CA: scipy
## take the tutoral
## https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

plt.scatter(PCs_df.PC1, PCs_df.PC2)
plt.scatter(PCs_df.PC1, PCs_df.PC3)
#Z = hierarchy.linkage(PCs_df, 'ward')
# the data set is too huge... MemoryError



##### CA: sklearn KMeans #####

## compute the Silhouette score of all samples to determine the number of clusters
"""
The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and 
the mean nearest-cluster distance (b) for each sample. 
The Silhouette Coefficient for a sample is (b - a) / max(a, b). 
The best value is 1 and the worst value is -1.
"""
start = time.time()
silhouette_avg = []

for n_clusters in range(2, 10):
    cluster_labels = KMeans(n_clusters = n_clusters).fit_predict(PCs_df.iloc[:, 0:33])
    silhouette_avg.append(
            silhouette_score(PCs_df, cluster_labels)
            )
    if n_clusters == 2:
        end = time.time()
        print('one round of sklearn KMeans costs {:0.1f} minutes'.format((end-start)/60))
        
sil_df = pd.DataFrame({'silhouette_scores': silhouette_avg,
                       'n_clusters': range(2,10)})
    
fig, ax = plt.subplots(1, 1, figsize = (5, 4.98))
ax.plot(sil_df.n_clusters, sil_df.silhouette_scores)
ax.set(
       xlabel = 'number of clusters', 
       ylabel = 'Silhoeutte scores'
       )

fig.savefig(
        'plots/silhouette_{}.pdf'.format(date), 
        bbox_inches = 'tight',
        dpi = 300)


end = time.time()
dur = (end-start)/3600
print('The loop of Sklearn KMeans takes {:0.2f} hours'.format(dur))

"""
The Silhouette plot shows 4 clusters has the second high score (the first is 2).
so I choose 4 as the cluster amount in kmeans result.
But the drop between 2 & 4 clusters is very large, maybe kmeans is not the suitable algorithm.
"""

labels_Kmeans = KMeans(n_clusters = 4).fit_predict(PCs_df.iloc[:, 0:9])
end = time.time()
dur = (end-start)/60
print('Sklean clustering takes {:0.1f} minutes'.format(dur))




##### CA: HDBSCAN #####
start = time.time()

# I choose 5 as the minimum size of cluster because data points fewer than 5 (1 cm) are too rare to be a kind of sediment  
clusterer = hdbscan.HDBSCAN(min_cluster_size = 5)
clusterer.fit(PCs_df.iloc[:, 0:9])

end = time.time()
dur = (end-start)/60
print('HDBSCAN clustering takes {:0.1f} minutes'.format(dur))
# HDBSCAN clustering takes 0.05 hours

palette = sns.color_palette('deep', clusterer.labels_.max() + 1)
colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in clusterer.labels_]
fig, ax = plt.subplots(1, 2, figsize = (10, 5), sharey = 'row')
ax[0].scatter(PCs_df.PC2, PCs_df.PC1, c = colors, alpha = 0.25, s = 20)
ax[1].scatter(PCs_df.PC3, PCs_df.PC1, c = colors, alpha = 0.25, s = 20)
ax[0].set(xlabel = 'PC2', ylabel = 'PC1')
ax[1].set(xlabel = 'PC3')
ax[0].text(-40, 225, 'Clustering took {:.2f} mins'.format(dur), fontsize=14)

fig.suptitle('Clusters found by HDBSCAN', size = 'xx-large')
fig.subplots_adjust(wspace = 0.02)

fig.savefig('plots/ca_result_hdbscan.pdf'.format(date),
            bbox_inches = 'tight',
            dpi = 300)

cluster_labels = pd.Series(clusterer.labels_)
for label in np.sort(cluster_labels.unique()):
    print(
            label, cluster_labels[cluster_labels == label].count(), sep = '\t'
            )



##### CA: random sampling + ward's #####
## random sampling
    
start = time.time()

for frac in np.arange(0.1, 1.0, 0.1)[::-1]:
    data = PCs_df.iloc[:, 0:9].sample(frac = frac, random_state = 1)
    try:
        Z = hierarchy.linkage(data, 'ward')
    except MemoryError:
        print('Fraction {} is too much.'.format(frac))
    else:
        print('Fraction {} is now avaiable for ward\'s calculation'.format(frac))
        break

end = time.time()
dur = (end-start)/60
print('randomly ward\'s clustering takes {:0.1f} minutes'.format(dur))

## draw dendrogram

#hierarchy.set_link_color_palette(['C1', 'C2','C4', 'C5', 'C6'])     # set the pslette for the dendrogram
plt.figure(figsize=(15, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
hierarchy.dendrogram(
    Z,
    no_labels = True,
    color_threshold = 0     # don't separate the clusters yet
)
#plt.show()

#hierarchy.set_link_color_palette(None)      # reset the color palette to its default

plt.savefig(
        'plots/dendrogram_ward_{}.pdf'.format(date), 
        bbox_inches = 'tight',
        dpi = 300)

AgglomerativeClustering(n_clusters = 5).fit(data)
labels_ward = AgglomerativeClustering(n_clusters = 5).fit_predict(PCs_df.iloc[:, :9])