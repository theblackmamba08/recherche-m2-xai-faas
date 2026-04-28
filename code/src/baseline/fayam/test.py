import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt
import plotly.express as px 

# --- Simuler des données ---
np.random.seed(0)
n_apps = 100
n_minutes = 60

# Chaque application a un profil aléatoire avec une tendance temporelle
data = np.array([[int(np.random.rand()*10) for _ in range(n_minutes)] for _ in range(n_apps)])
print(data)
# --- Standardiser par ligne (z-score par application) ---
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data.T).T  # standardiser chaque ligne
print(data_standardized)

# --- Calculer la matrice de distance de corrélation ---
distance_matrix = pairwise_distances(data_standardized, metric='correlation')
print(distance_matrix)

# --- Clustering avec HDBSCAN en utilisant la matrice de distances ---
clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5)
labels = clusterer.fit_predict(distance_matrix)
print(labels)

# --- t-SNE pour visualiser ---

perplexity = np.arange(5, 100, 5)
divergence = []

for i in perplexity:
    model = TSNE(n_components=2, perplexity=i, random_state=42, init='pca')
    reduced = model.fit_transform(distance_matrix)
    divergence.append(model.kl_divergence_)
fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
fig.show()

tsne = TSNE(n_components=2, metric="precomputed", perplexity=99, random_state=42, init='random')
embedding = tsne.fit_transform(distance_matrix)

# --- Visualisation ---
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=50)
plt.title("Visualisation des clusters d'applications (t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Cluster")
plt.show()