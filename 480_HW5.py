from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import cdist 
import numpy as np
import pandas as pd 
#import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.io as io


def elbow_method(x, y, X, gth, run_iris=False, run_pines=False):
    if run_iris:
        cost = []
        for i in range(2,6):
            km = KMeans(n_clusters=i, max_iter=500)
            km.fit(x)
            cost.append(km.inertia_)
        plt.plot(range(2,6), cost, color='b', linewidth='4')
        plt.xlabel("K value")
        plt.ylabel("Squared Error (cost)")
        plt.title("Elbow Method on IRIS data")
        plt.show()
    # Elbow method on Pines dataset    
    if run_pines:
        cost = []
        for i in range(2,16):
            km = KMeans(n_clusters=i, max_iter=500)
            km.fit(X)
            cost.append(km.inertia_)
        plt.plot(range(2,16), cost, color='b', linewidth='4')
        plt.xlabel("K value")
        plt.ylabel("Squared Error (cost)")
        plt.title("Elbow Method on Pines data")
        plt.show()


def classification_kmeans(x, y, X, gth, run_PCA=False, run_iris=False, run_pines=False):
    if run_PCA:
        if run_iris:
            pca = PCA(n_components=3)
            x = pca.fit_transform(x)
        if run_pines:
            pca = PCA(n_components=10)
            principleComponents = pca.fit_transform(X)
            x1 = X.transpose()
            X = np.matmul(x1, principleComponents)
            print("X:", principleComponents.shape)
            print("gth:", gth.shape)
            X = principleComponents
        print("Running with PCA dimensionality reduction")

    colors = np.array(['red', 'green', 'blue'])
    if run_iris:
        km = KMeans(n_clusters=4, random_state=111)
        km.fit(x)
        centroids = pd.DataFrame(km.cluster_centers_, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
        print("Centroids:\n", centroids)
        pred_y = pd.DataFrame(km.labels_, columns=['Target'])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Before and after for sepal length vs sepal width
        axes[0].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=50)
        axes[1].scatter(x['Sepal Length'], x['Sepal Width'], c=colors[pred_y['Target']], s=50)
        axes[1].scatter(centroids['Sepal Length'], centroids['Sepal Width'], c='k', s=70)
        axes[0].set_xlabel("Sepal Length", fontsize=14)
        axes[0].set_ylabel("Sepal Width", fontsize=14)
        axes[0].set_title("Before K-Means Clustering")
        axes[1].set_xlabel("Sepal Length", fontsize=14)
        axes[1].set_ylabel("Sepal Width", fontsize=14)
        axes[1].set_title("After K-Means Clustering")
        plt.show()
        # Before and after for petal length vs petal width
        axes[0].scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=50)
        axes[1].scatter(x['Petal Length'], x['Petal Width'], c=colors[pred_y['Target']], s=50)
        axes[1].scatter(centroids['Petal Length'], centroids['Petal Width'], c='k', s=70)
        axes[0].set_xlabel("Petal Length", fontsize=14)
        axes[0].set_ylabel("Petal Width", fontsize=14)
        axes[0].set_title("Before K-Means Clustering")
        axes[1].set_xlabel("Petal Length", fontsize=14)
        axes[1].set_ylabel("Petal Width", fontsize=14)
        axes[1].set_title("After K-Means Clustering")
        plt.show()


if __name__ == "__main__":
    # Get iris data
    iris = load_iris()
    x_iris = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y_iris = pd.DataFrame(iris.target, columns=['Target'])
    # Get Pines data
    #R_file = io.loadmat("data/indianR.mat")
    #gth = np.array(R_file['gth'])
    #X = np.array(R_file['X']).transpose()
    #R_rows = R_file['num_rows']
    #R_cols = R_file['num_cols']
    #R_bands = R_file['num_bands']
    #gth = np.reshape(gth, (int(R_rows)*int(R_cols)))
    # Ugly but works. Remove all the data where there isn't a ground truth
    #zeros = (gth == 0).nonzero()
    #for i in range(len(zeros)):
    #    zeros_new = (gth == 0).nonzero()
    #    delete_index = zeros_new[0]
    #    gth = np.delete(gth, delete_index)
    #    X = np.delete(X, delete_index, axis=0)
    # Run elbow method and plot output
    #elbow_method(x_iris, y_iris, X, gth, run_iris=True, run_pines=False)
    classification_kmeans(x_iris, y_iris, None, None, run_PCA=False, run_iris=True)
