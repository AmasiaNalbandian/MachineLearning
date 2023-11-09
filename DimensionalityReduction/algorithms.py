from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, mean_squared_error, silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import kurtosis


def tsne_transform(data, n_components=2, perplexity=30.0, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(data)

def plot_cluster_3d(data, labels, cluster_centers, title): 
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], s=300, c='red', marker='X', label='Centroids')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.title(title)
        plt.legend()
        plt.show()

def plot_cluster_2d(data, labels, cluster_centers, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    if cluster_centers is not None and not isinstance(cluster_centers, str):  
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_clustering(data, labels, cluster_centers=None, title="Clustering", tsne_components=2, run_tsne=True):
    if run_tsne:
        data = tsne_transform(data, n_components=tsne_components)
    
    # Plotting the data after t-SNE transformation
    if tsne_components == 2:
        plot_cluster_2d(data, labels, cluster_centers, title)
    elif tsne_components == 3:
        plot_cluster_3d(data, labels, cluster_centers, title)


def run_kmeans(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=26)
    labels = kmeans.fit_predict(data)
    return kmeans, labels

def run_gmm(data, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=26)
    gmm.fit(data)
    labels = gmm.predict(data)
    return gmm, labels

def plot_explained_variance(pca, title):
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title(f'Visualization of {title}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Dimension Reduction algorithms       
def pca(x_train, n_components = 2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(x_train)
    eigenvalues = pca.explained_variance_
    print("Explained Variance per component:", pca.explained_variance_ratio_)
    print("Total Explained Variance (first n components):", sum(pca.explained_variance_ratio_))
    return pca, X_pca

def plot_eigenvalues(X_train_scaled, title):
    pca_full = PCA()
    pca_full.fit(X_train_scaled)

    # Eigenvalues (which are the same as the explained_variance of pca_full)
    eigenvalues = pca_full.explained_variance_
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2)
    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()


def ica(x_train, n_components=2):
    ica = FastICA(n_components=n_components)
    X_ica = ica.fit_transform(x_train)
    return X_ica

def evaluate_kurtosis(X, n_components_range):
    kurtosis_results = []
    for n_components in n_components_range:
        X_ica = ica(X, n_components)
        kurt_values = kurtosis(X_ica, axis=0, fisher=True)  # Fisher's definition (normal ==> 0.0)
        kurtosis_results.append(np.mean(np.abs(kurt_values)))  # Average of absolute kurtosis values
    return kurtosis_results

def apply_random_projection(X_train, X_test, n_components='auto'):
    rp = GaussianRandomProjection(n_components=n_components)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    return X_train_rp, X_test_rp


def visualize_silhouette_score(X, n_clusters, cluster_labels):
    """
    Visualize the silhouette scores for each sample in a clustering.

    Parameters:
    - X: Feature dataset, a NumPy array or a DataFrame.
    - n_clusters: The number of clusters.
    - cluster_labels: Cluster labels for each point.
    """
    # Create a subplot with 1 row and 1 column
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples gap between clusters

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


def calculate_silhouette_score(X, cluster_labels):
    """
    Calculate the mean silhouette score for all samples.

    Parameters:
    - X: Feature dataset, a NumPy array or a DataFrame.
    - cluster_labels: Cluster labels for each point.

    Returns:
    - silhouette_avg: The mean silhouette score.
    """
    silhouette_avg = silhouette_score(X, cluster_labels)
    return silhouette_avg

def calculate_reconstruction_error(data, n_components):
    pca = PCA(n_components=n_components)
    data_transformed = pca.fit_transform(data)
    data_inverse = pca.inverse_transform(data_transformed)
    reconstruction_error = np.mean(np.square(data - data_inverse))
    return reconstruction_error
