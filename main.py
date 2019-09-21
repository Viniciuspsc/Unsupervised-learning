import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics.cluster import contingency_matrix

print()
print('--------------------------------------')
print('Programa para clusterização de medidas')
print('--------------------------------------')
print()
# #############################################################################
# GERANDO OS DADOS PARA CLUSTERIZAR
matricula = 34706 # insira sua matricula aqui
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=1.9, random_state=matricula)

# #############################################################################
# CALCULANDO CLUSTERS
# Estima a largura de banda para o MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=200)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
# numero de clusters encontrado
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print('MeanShift clusters estimados: ', n_clusters_)

# erro da clusterizacao real
print('MeanShift Matriz de Contigência')
cm = contingency_matrix(labels_true=y, labels_pred=labels)
print(cm)


 
#############################################################################
# GRÁFICO DOS PONTOS ORIGINAIS
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(4), colors):
    my_members = y == k
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('0_Padrao.png')



#############################################################################
print()
print()
print()
print('--------------------------------------')
print('Kmeans:')
print()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='random', n_init=5, max_iter=100, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')

kmeans.fit(X)
labels_1 = kmeans.labels_
# numero de clusters encontrado
labels_unique_1 = np.unique(labels_1)
n_clusters_1 = len(labels_unique_1)
print('Kmeans clusters estimados: ', n_clusters_1)
# erro da clusterizacao real
print('Kmeans Matriz de Contigência')
cm_1 = contingency_matrix(labels_true=y, labels_pred=labels_1)
print(cm_1)

 #plotar grafico
for k  in range(0,1000):
  if labels_1[k] == -1:
      color = 'k'
  elif labels_1[k] == 0:
      color = 'r'
  elif  labels_1[k] == 1:
      color = 'g'
  elif  labels_1[k] == 2:
      color = 'b'
  elif  labels_1[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Kmeans')
plt.savefig('1_Kmeans.png')


#############################################################################
print()
print()
print()
print('--------------------------------------')
print('Spectral clustering:')
print()
from sklearn.cluster import SpectralClustering

Spectral = SpectralClustering(n_clusters=4, eigen_solver=None, random_state=None, n_init=10, gamma=0.55, affinity='rbf', n_neighbors=0, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None)

Spectral.fit(X)
labels_2 = Spectral.labels_
cluster_centers_2 = Spectral.affinity_matrix_
# numero de clusters encontrado
labels_unique_2 = np.unique(labels_2)
n_clusters_2 = len(labels_unique_2)
print('Spectral clusters estimados: ', n_clusters_2)
# erro da clusterizacao real
print('Spectral Matriz de Contigência')
cm_2 = contingency_matrix(labels_true=y, labels_pred=labels_2)
print(cm_2)

 #plotar grafico
for k  in range(0,1000):
  if labels_2[k] == -1:
      color = 'k'
  elif labels_2[k] == 0:
      color = 'r'
  elif  labels_2[k] == 1:
      color = 'g'
  elif  labels_2[k] == 2:
      color = 'b'
  elif  labels_2[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('2_Spectral.png')



#############################################################################
print()
print()
print()
print('--------------------------------------')
print('Agglomerative Clustering:')
print()
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, 5, mode='connectivity', include_self=True)

Agglomerative = AgglomerativeClustering(n_clusters=4, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func='deprecated', distance_threshold=None)

Agglomerative.fit(X)
labels_3 = Agglomerative.labels_
# numero de clusters encontrado
labels_unique_3 = np.unique(labels_3)
n_clusters_3 = len(labels_unique_3)
print('Agglomerative clusters estimados: ', n_clusters_3)
# erro da clusterizacao real
print('Agglomerative Matriz de Contigência')
cm_3 = contingency_matrix(labels_true=y, labels_pred=labels_3)
print(cm_3)

 #plotar grafico
for k  in range(0,1000):
  if labels_3[k] == -1:
      color = 'k'
  elif labels_3[k] == 0:
      color = 'r'
  elif  labels_3[k] == 1:
      color = 'g'
  elif  labels_3[k] == 2:
      color = 'b'
  elif  labels_3[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('3_Agglomerative.png')



#############################################################################
print()
print()
print()
print('--------------------------------------')
print('DBSCAN:')
print()
from sklearn.cluster import DBSCAN

DBSCAN = DBSCAN(eps=1, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)

DBSCAN.fit(X)
labels_4 = DBSCAN.labels_
# numero de clusters encontrado
labels_unique_4 = np.unique(labels_4)
if labels_unique_4[0] == -1:
    n_clusters_4 = len(labels_unique_4) - 1
    print('DBSCAN clusters estimados: ', n_clusters_4)
else:
    n_clusters_4 = len(labels_unique_4) 
    print('DBSCAN clusters estimados: ', n_clusters_4)

# erro da clusterizacao real
print('DBSCAN matriz de Contigência')
cm_4 = contingency_matrix(labels_true=y, labels_pred=labels_4)
print(cm_4)
print('A 1º coluna representa os pontos que o programa entendeu como ruído, considerando como "-1", pontos pretos')

#plotar grafico
for k  in range(0,1000):
  if labels_4[k] == -1:
      color = 'k'
  elif labels_4[k] == 0:
      color = 'r'
  elif  labels_4[k] == 1:
      color = 'g'
  elif  labels_4[k] == 2:
      color = 'b'
  elif  labels_4[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('4_DBSCAN.png')



#############################################################################
print()
print()
print()
print('--------------------------------------')
print('Gaussian mixture:')
print()
from sklearn import mixture

Gaussian = mixture.GaussianMixture(n_components=4, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=2, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)

Gaussian.fit(X)
a = Gaussian.predict(X)
# numero de clusters encontrado
labels_unique_5 = np.unique(a)
n_clusters_5 = len(labels_unique_5)
print('Gaussian mixture clusters estimados: ', n_clusters_5)
print('Gaussian mixture matriz de Contigência')
cm_5 = contingency_matrix(labels_true=y, labels_pred=a)
print(cm_5)

#plotar grafico
for k  in range(0,1000):
  if a[k] == -1:
      color = 'k'
  elif a[k] == 0:
      color = 'r'
  elif  a[k] == 1:
      color = 'g'
  elif  a[k] == 2:
      color = 'b'
  elif  a[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('5_gaussian.png')


#############################################################################
print()
print()
print()
print('--------------------------------------')
print('Birch:')
print()
from sklearn.cluster import Birch

Birch = Birch(threshold=0.5, branching_factor=25, n_clusters=4, compute_labels=True, copy=True)

Birch.fit(X)
labels_6 = Birch.labels_
# numero de clusters encontrado
labels_unique_6 = np.unique(labels_6)
n_clusters_6 = len(labels_unique_6)
print('Birch clusters estimados: ', n_clusters_6)
# erro da clusterizacao real
print('Birch Matriz de Contigência')
cm_6 = contingency_matrix(labels_true=y, labels_pred=labels_6)
print(cm_6)

#plotar grafico
for k  in range(0,1000):
  if labels_6[k] == -1:
      color = 'k'
  elif labels_6[k] == 0:
      color = 'r'
  elif  labels_6[k] == 1:
      color = 'g'
  elif  labels_6[k] == 2:
      color = 'b'
  elif  labels_6[k] == 3:
      color = 'c'
  plt.plot(X[k, 0], X[k, 1], color+'.')
plt.title('Pontos sorteados para clusterização')
plt.savefig('6_Birch.png')
