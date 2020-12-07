import numpy as np
from binarytree import Node
from itertools import permutations
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

A = [161, 112, 165]
B = [92, 163]
C = [71, 195]

num_unlock_vectors = 24
unlock_vectors = np.zeros((num_unlock_vectors, 4), dtype=int)
m = 0
for i, j in permutations(A, 2) :

    for k in B :

        for l in C :

            unlock_vectors[m] = [i, j, k, l]
            m += 1

num_table_vectors = num_unlock_vectors

cluster_ids = np.arange(num_table_vectors)
num_clusters = num_table_vectors

row_cluster_id = [i for i in range(num_table_vectors)]

i, j = np.meshgrid(cluster_ids, cluster_ids, indexing='ij')

table = np.linalg.norm(np.subtract(unlock_vectors[i], unlock_vectors[j]), axis=2)

linkage_matrix = []

while num_table_vectors > 1 :

    tril_rows, tril_cols = np.tril_indices_from(table, k=-1,)
    min_index = np.argmin(table[tril_rows, tril_cols])

    i1 = min(tril_rows[min_index], tril_cols[min_index])
    i2 = max(tril_rows[min_index], tril_cols[min_index])

    cluster_ids[cluster_ids == i2] = i1
    cluster_ids[cluster_ids >  i2] -= 1

    linkage_matrix.append([row_cluster_id[i1], row_cluster_id.pop(i2), table[i1, i2], np.count_nonzero(cluster_ids == i1)])

    row_cluster_id[i1] = num_clusters
    num_clusters += 1

    table[i1, :] = np.max(np.vstack((table[i1, :], table[i2, :])), axis=0)
    table[:, i1] = np.max(np.vstack((table[:, i1], table[:, i2])), axis=0)

    table = np.delete(table, i2, 0)
    table = np.delete(table, i2, 1)

    num_table_vectors -= 1

linkage_matrix = np.array(linkage_matrix)

plt.style.use('dark_background')

dendrogram(linkage_matrix, p=20, labels=unlock_vectors)

plt.title('Hierarchical Clustering of Unlock Combinations', fontdict={'fontsize':14})
plt.grid(which='major', axis='y', ls='--', color='grey')
plt.minorticks_on()
plt.grid(which='minor', ls='dotted', axis='y', color='green')
plt.show()

