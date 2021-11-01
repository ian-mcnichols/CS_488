import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def pines_analysis(R_filepath, display=False):
    R_file = io.loadmat(R_filepath)
    gth = np.array(R_file['gth'])
    X = np.array(R_file['X'])
    R_rows = R_file['num_rows']
    R_cols = R_file['num_cols']
    R_bands = R_file['num_bands']
    gth = np.reshape(gth, (int(R_rows)*int(R_cols)))
    bands, samples = X.shape
    gth_mat = io.loadmat('data/indian_gth.mat')
    gth_mat = {i:j for i, j in gth_mat.items() if i[0] != "_"}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})
    #R_data = pd.DataFrame(data=X, columns=gth)
    #R_data.to_csv('data/IndianPines.csv')
    #gt = np.reshape(gt, (21025))
    n = []
    ind = []
    for i in range(bands):
        n.append(i+1)
    for i in range(bands):
        ind.append('band' + str(n[i]))
    features = ind

    #X = np.reshape(X, (int(R_rows)*int(R_cols), int(R_bands)))
    scaler_model = MinMaxScaler()
    scaler_model.fit(X.astype(float))
    X = scaler_model.transform(X)
    print('gt shape:', gt.shape)
    print('X shape:', X.shape)
    target_names = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
                    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
                    "Buildings-Grass-Treess-Drives", "Stone-Steel-Towers"]

    # Starting PCA
    pca = PCA(n_components=10)
    principleComponents = pca.fit_transform(X)
    principleDf = pd.DataFrame(data=principleComponents,
                               columns = ['PC-' + str(i+1) for i in range(10)])
    finalDf1 = pd.concat([principleDf, gt], axis=1)

    x1 = X.transpose()
    X_pca = np.matmul(x1, principleComponents)
    x_pca_df = pd.DataFrame(data=X_pca, columns=['PC-' + str(i+1) for i in range(10)])
    X_pca_df = pd.concat([x_pca_df, gt], axis=1)

    # Starting LDA    
    X = X.transpose()
    lda = LinearDiscriminantAnalysis(n_components=10)
    linear_discriminants = lda.fit(X, np.ravel(gt)).transform(X)
    # Making dataframes from LDA
    linearDf = pd.DataFrame(data=linear_discriminants,
                            columns = ['LD-' + str(i+1) for i in range(10)])
    finalDf2 = pd.concat([linearDf, gt], axis=1)
    x2 = X.transpose()
    X_lda = np.matmul(x2, linear_discriminants)
    x_lda_df = pd.DataFrame(data=X_lda, columns=['LD-' + str(i+1) for i in range(10)])
    X_lda_df = pd.concat([x_lda_df, gt], axis=1)

    if display:
        class_num = [i+1 for i in range(15)]
        colors = ["navy", "turquoise", "mediumslateblue", "gray", "lime", "pink", "yellow",
                  "red", "black", "tomato", "mediumorchid", "darkolivegreen", "darkorange",
                  "crimson", "gold", "peru", "mediumslateblue"]
        markerm = ['o', 'o', 'o', 'o', 'o', 'o', 'o', '+', '+', '+', '+', '+', '+', '+', '*', '*']

        
        # Displaying Variance Ratio
        plt.figure()
        plt.bar([1,2,3,4,5,6,7,8,9,10], list(pca.explained_variance_ratio_*100),label="Principal Components", color="b")
        plt.legend()
        plt.xlabel('Principal Components')
        pc = []
        for i in range(10):
            pc.append('PC' + str(i+1))
        plt.xticks([1,2,3,4,5,6,7,8,9,10],pc,fontsize=8,rotation=30)
        plt.ylabel('Variance Ratio')
        plt.title('Variance Ratio of INDIAN PINES Dataset')
        plt.show()

        # Displaying PCA
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('PC-1', fontsize=15)
        ax.set_ylabel('PC-2', fontsize=15)
        ax.set_title('PCA on INDIAN PINES Dataset', fontsize=20)
        for target, color, m in zip(class_num, colors, markerm):
            indicesToKeep = X_pca_df['gth'] == target
            ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'],
                       X_pca_df.loc[indicesToKeep, 'PC-2'],
                       c=color, marker=m, s=9)
        ax.legend()
        ax.grid()
        plt.show()
        

        # Displaying LDA
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('LD-1', fontsize=15)
        ax.set_ylabel('LD-2', fontsize=15)
        ax.set_title('LDA on INDIAN PINES Dataset', fontsize=20)
        for color, i, target_name in zip(colors, class_num, class_num):
            ax.scatter(linear_discriminants[gth==i, 0], linear_discriminants[gth==i,1],
                        color=color, label=target_name)
        ax.legend()
        ax.grid()
        plt.show()

        plt.show()


def iris_analysis(display=False):
    # i) For PCA, plot the explained variance for all the PC’s in the dataset.(10 points)
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    if display:
        plt.figure()
        plt.bar([1,2],list(pca.explained_variance_ratio_*100), label='Principal Components', color='b')
        plt.legend()
        plt.xlabel("Principal Components")
        pc = []
        for i in range(2):
            pc.append('PC' + str(i+1))
        plt.xticks([1,2],pc,fontsize=8,rotation=25)
        plt.ylabel("Variance Ratio")
        plt.title("Variance Ratio of normal distributed data (IRIS)")

        plt.figure()
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2

        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(
                X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("PCA of IRIS dataset")

        plt.figure()
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(
                X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("LDA of IRIS dataset")

        plt.show()


def iris_classification():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    test_sizes = [.1, .2, .3, .4, .5]
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM-Poly', SVC(gamma='auto',kernel='poly')))
    models.append(('SVM-RBF', SVC(gamma='auto',kernel='rbf')))
    models.append(('NB', GaussianNB()))
    # Train and test without dimensionality reduction
    for name, model in models:
        names = []
        results = []
        for test_size in test_sizes:
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size,random_state=1,shuffle=True)
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
            results.append(cv_results)
            names.append(name)
        plt.boxplot(results, labels=test_sizes)
        plt.title("Algorithm Comparison on Iris dataset, Model = " + str(name))
        plt.show()
    # Run PCA dimensionality reduction
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    # Train and test models
    for name, model in models:
        names = []
        results = []
        for test_size in test_sizes:
            X_train, X_validation, Y_train, Y_validation = train_test_split(X_r, y, test_size=test_size,random_state=1,shuffle=True)
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
            results.append(cv_results)
            names.append(name)
        plt.boxplot(results, labels=test_sizes)
        plt.title("Algorithm Comparison on Iris dataset, Model = " + str(name))
        plt.show()


def main():
    filepath = "data/indianR.mat"

    #iris_analysis(display=False)
    #pines_analysis(filepath, display=True)
    iris_classification()
    #PCA_(Rdata, Rtruth)


if __name__ == "__main__":
    main()
