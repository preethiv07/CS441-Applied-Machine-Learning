%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Input
df = pd.read_csv("../Clustering-lib/EuropeanJobs.dat", sep='\t', header=0)

feature_cols = ['Agr','Min','Man','PS','Con','SI','Fin','SPS','TC'] 
X = df[feature_cols].values
Y = df['Country'].tolist()

def single_linkage(X):
    """
    Produce a single-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                single_link (np.array): The single-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    single_link=hierarchy.linkage(X,optimal_ordering=True)
    
    return single_link

single_link = single_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Single Link")
dn_single = hierarchy.dendrogram(single_link, labels=Y)

def complete_linkage(X):
    """
    Produce a complete-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                comp_link (np.array): The complete-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    comp_link = hierarchy.linkage(X, method='complete' , optimal_ordering=True)
    
    return comp_link

complete_link = complete_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Complete Link")
dn_complete = hierarchy.dendrogram(complete_link,labels=Y)


def group_avg_linkage(X):
    """
    Produce an average-link agglomerative clustering.
    
        Parameters:
                X (np.array): A numpy array of the shape (N,d) where N is the number of samples and d is the number of features.
                
        Returns:
                avg_link (np.array): The average-link agglomerative clustering of X encoded as a linkage matrix.
    """
    
    # your code here
    avg_link = hierarchy.linkage(X, method='average' , optimal_ordering=True)
    
    return avg_link

average_link = group_avg_linkage(X)
plt.figure(figsize=(12,6), dpi=90)
plt.ylabel("Distance")
plt.title("Agglomerative Clustering of European Jobs - Group Average")
dn_average = hierarchy.dendrogram(average_link,labels=Y)

# Task 3: K means clustering
k_list = list(range(2,26))
k_inertias = []
k_scores = []
model_list = []
for k in k_list:
    model = KMeans(n_clusters=k, random_state=12345).fit(X)
    model_list.append(model)
    cluster_assignments = model.labels_
    score = silhouette_score(X, cluster_assignments, metric='euclidean')
    inertia = model.inertia_
    k_scores.append(score)
    k_inertias.append(inertia)


# Plot K Means
plt.figure(figsize=(8,4), dpi=120)
plt.title('The Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Square Distances')
_=plt.plot(k_list, k_inertias,'bo--')

plt.figure(figsize=(8,4), dpi=120)
plt.title('Silhouette Score vs. Number of clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
_=plt.plot(k_list, k_scores,'bo--')