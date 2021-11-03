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
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline


def pines_analysis(R_filepath, display=False):
    # Load in Indian Pines data from file and reshape/make into dataframes
    R_file = io.loadmat(R_filepath)
    gth = np.array(R_file['gth'])
    X = np.array(R_file['X'])
    R_rows = R_file['num_rows']
    R_cols = R_file['num_cols']
    R_bands = R_file['num_bands']
    # Store ground truth data
    gth = np.reshape(gth, (int(R_rows)*int(R_cols)))
    gth_mat = io.loadmat('data/indian_gth.mat')
    gth_mat = {i:j for i, j in gth_mat.items() if i[0] != "_"}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})
    # Pre-process data
    scaler_model = MinMaxScaler()
    scaler_model.fit(X.astype(float))
    X = scaler_model.transform(X)
    # Give a heads up to the user about data information
    print('gt shape:', gt.shape)
    print('X shape:', X.shape)
    # Class types in order of the Pines dataset
    target_names = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
                    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
                    "Buildings-Grass-Treess-Drives", "Stone-Steel-Towers"]

    # Starting PCA
    pca = PCA(n_components=10)
    principleComponents = pca.fit_transform(X)
    # Creating dataframes from PCA output
    principleDf = pd.DataFrame(data=principleComponents,
                               columns = ['PC-' + str(i+1) for i in range(10)])
    finalDf1 = pd.concat([principleDf, gt], axis=1)
    # Reshaping PCA output for plotting, adding PC titles to each column
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
    # Reshaping output of LDA so it can be plotted, assigning LD's to each column of data
    x2 = X.transpose()
    X_lda = np.matmul(x2, linear_discriminants)
    x_lda_df = pd.DataFrame(data=X_lda, columns=['LD-' + str(i+1) for i in range(10)])
    X_lda_df = pd.concat([x_lda_df, gt], axis=1)

    if display:
        # Lists that will be re-used for each plot
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


def iris_analysis(display=False):
    # Load iris dataset into data and targets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Run PCA on the iris dataset
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    # Run LDA on the iris dataset
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    # Plot figures if prompted
    if display:
        # Show the first 2 PC's
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
        # Display 2D PCA results
        plt.figure()
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            # Show the PCA results for each datatype in the same color
            plt.scatter(
                X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("PCA of IRIS dataset")
        # Display 2D LDA results
        plt.figure()
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            # Show the LDA results for each datatype in the same color
            plt.scatter(
                X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("LDA of IRIS dataset")
        # Show all the plots to screen
        plt.show()


def iris_classification(run_pca=False, run_lda=True):
    if run_pca and run_lda:
        print("Can't run both lda and pca.")
        return
    # Load IRIS dataset, run PCA, and run combinations of all test sizes and model types
    # Display output to the user's screen

    # Load iris dataset into X and y variables
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    plot_data = {}
    # Check for PCA flag and run dimensionality reduction if needed
    if run_pca:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    if run_lda:
        lda = LinearDiscriminantAnalysis(n_components=2)
        X = lda.fit(X, y).transform(X)

    test_sizes = [.9, .8, .7, .6, .5]
    for test_size in test_sizes:
        # Split dataset based on test size
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size,random_state=1,shuffle=True)
        # Re-make each model for the current test size
        models = [('KNN', KNeighborsClassifier()), ('SVM-Poly', SVC(gamma='auto', kernel='poly')),
                  ('SVM-RBF', SVC(gamma='auto', kernel='rbf')), ('NB', GaussianNB())]
        # Lists to hold average accuracies for later
        classification_accuracy = []
        training_accuracy = []
        for name, model in models:
            # Create a pipeline to preprocess data and run the model
            pipeline = Pipeline([("scaler", MinMaxScaler()), ("classifier", model)])
            # Fit to the training dataset
            pipeline.fit(X_train, Y_train)
            # Create a confusion matrix
            disp = plot_confusion_matrix(pipeline, X_validation, Y_validation, cmap=plt.cm.Blues)
            true_positive = disp.confusion_matrix[1][1]
            false_negative = disp.confusion_matrix[1][0]
            true_negative = disp.confusion_matrix[0][0]
            false_positive = disp.confusion_matrix[0][1]
            # Calculate sensitivity and specificity from confusion matrix
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            # Calculate and store overall accuracy of the model
            classification_score = pipeline.score(X_validation, Y_validation)
            classification_accuracy.append(classification_score*100)
            # Calculate and store training accuracy of the model
            training_accuracy.append(pipeline.score(X_train, Y_train)*100)
            # Print model information to display
            data_line = "test size: " + str(test_size) + " model: " + str(name) + " training accuracy: " +\
                        str(pipeline.score(X_train, Y_train)*100) + " classification accuracy: " +\
                        str(classification_score*100)
            data_line += " sensitivity: " + str(sensitivity) + " specificity: " + str(specificity) + "\n"
            print(data_line)
        # Combine all the averages per test size
        plot_data[test_size] = [training_accuracy, classification_accuracy]
    # Display average overall accuracy and training accuracy to output for each test size
    training_accuracies = [plot_data[x][0] for x in plot_data]
    print("Training accuracies:", training_accuracies)
    classification_accuracies = [plot_data[x][1] for x in plot_data]
    print("Classification accuracies:", classification_accuracies)


def pines_classification(R_filepath, run_pca=False, run_lda=False):
    if run_pca and run_lda:
        print("Can't run both pca and lda")
        return
    # Runs combinations of model types/test sizes with or without PCA dimensionality reduction on pines dataset
    # Loading in the pines dataset
    R_file = io.loadmat(R_filepath)
    gth = np.array(R_file['gth'])
    X = np.array(R_file['X']).transpose()
    R_rows = R_file['num_rows']
    R_cols = R_file['num_cols']
    R_bands = R_file['num_bands']
    gth = np.reshape(gth, (int(R_rows)*int(R_cols)))

    if run_pca:
        # Run PCA on data and save back to the X variable
        pca = PCA(n_components=10)
        principleComponents = pca.fit_transform(X)
        x1 = X.transpose()
        X = np.matmul(x1, principleComponents)
        print("X:", principleComponents.shape)
        print("gth:", gth.shape)
        X = principleComponents
        print("Running with PCA dimensionality reduction")
    elif run_lda:
        lda = LinearDiscriminantAnalysis(n_components=10)
        linear_discriminants = lda.fit(X, np.ravel(gth)).transform(X)
        X = linear_discriminants
        print("X:", X.shape)
        print("gth:", gth.shape)
        print("Running with LDA reduction")
    else:
        # Display data info
        print("X:", X.shape)
        print("gth:", gth.shape)
        print("Running without dimensionality reduction")
    plot_data = {}
    test_sizes = [.9, .8, .7, .6, .5]
    for test_size in test_sizes:
        # Splitting data by test size
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, gth, test_size=test_size,
                                                                        random_state=1, shuffle=True)
        # Re-making each model to use on the current test size (cache size taken from CPU cache size to go faster)
        models = [('NB', GaussianNB()), ('KNN', KNeighborsClassifier()), ('SVM-Poly', SVC(gamma='auto', kernel='poly', cache_size=5200000)),
                  ('SVM-RBF', SVC(gamma='auto', kernel='rbf', cache_size=5200000))]
        classification_accuracy = []
        training_accuracy = []
        for name, model in models:
            # Create a pipeline to preprocess data and run the model
            pipeline = Pipeline([("scaler", MinMaxScaler()), ("classifier", model)])
            # Fit to the training dataset
            pipeline.fit(X_train, Y_train)
            # Make a confusion matrix
            disp = plot_confusion_matrix(pipeline, X_validation, Y_validation, cmap=plt.cm.Blues)
            # Grab values for sensitivity/specificty
            true_positive = disp.confusion_matrix[1][1]
            false_negative = disp.confusion_matrix[1][0]
            true_negative = disp.confusion_matrix[0][0]
            false_positive = disp.confusion_matrix[0][1]
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            # Overall model classification score vs training data score
            classification_score = pipeline.score(X_validation, Y_validation)
            train_score = pipeline.score(X_train, Y_train)
            # Add to a list to calculate the average later
            training_accuracy.append(train_score)
            classification_accuracy.append(classification_score*100)
            # Printing model information to output
            data_line = "test size: " + str(test_size) + " model: " + str(name) + " training accuracy: " + str(train_score*100) + " classification accuracy: " + str(classification_score*100)
            data_line += " sensitivity: " + str(sensitivity) + " specificity: " + str(specificity) + "\n"
            print(data_line)
        # Adding all model averages to a dictionary corresponding with test size
        # TODO: Fix it from being averages
        plot_data[test_size] = [training_accuracy, classification_accuracy]
    # Displaying overall accuracies for each test size
    test_sizes = [x*100 for x in test_sizes]
    training_accuracies = [plot_data[x][0] for x in plot_data]
    print("Training accuracies:", training_accuracies)
    classification_accuracies = [plot_data[x][1] for x in plot_data]
    print("Classification accuracies:", classification_accuracies)


def plot_classification(iris, pines):
    # Plots the training/overall accuracies of each model at each test-size
    test_sizes = [.5, .4, .3, .2, .1][::-1]
    if pines:
        # Saved data from PINES classification output (Hard-coded to avoid re-running models)
        # No PCA
        accuracies = []
        classification_accuracies = []
        training_accuracies = [56.51636225266362, 56.59532302814111, 56.64367737990079, 56.68995243757431, 56.72233379135398][::-1]
        plt.scatter(test_sizes, accuracies, label="Classification accuracy",c="blue")
        plt.scatter(test_sizes, training_accuracies, label="Training accuracy",c="red")
        plt.xlabel("% test size")
        plt.ylabel("Average accuracy")
        plt.title("Supervised Classification without Dimensionality Reduction on INDIAN PINES")
        plt.legend()
        plt.show()
        # After PCA
        pca_accuracies = [61.07914011224199, 61.278240190249704, 61.711318960050725, 61.86682520808561, 62.684260580123635][::-1]
        training_accuracies = [63.5773401826484, 63.77724930638129, 64.07555887748861, 64.19887039239001, 64.3444139097347][::-1]
        plt.scatter(test_sizes, pca_accuracies, label="Classification accuracy")
        plt.scatter(test_sizes, training_accuracies, label="Training accuracy")
        plt.xlabel("% test size")
        plt.ylabel("Average accuracy")
        plt.title("Supervised Classification with PCA Reduction on INDIAN PINES")
        plt.legend()
        plt.show()
    if iris:
        # Saved data from IRIS classification output (Hard-coded to avoid re-running models)
        # No PCA
        training_accuracies = [[[86.66666666666667, 53.333333333333336, 86.66666666666667, 100.0],
                               [96.66666666666667, 43.333333333333336, 76.66666666666667, 100.0],
                               [95.55555555555556, 53.333333333333336, 91.11111111111111, 95.55555555555556],
                               [96.66666666666667, 51.66666666666667, 95.0, 93.33333333333333],
                               [94.66666666666667, 66.66666666666666, 94.66666666666667, 93.33333333333333]],
                               [[80.0, 53.333333333333336, 73.33333333333333, 100.0],
                                [93.33333333333333, 43.333333333333336, 76.66666666666667, 96.66666666666667],
                                [91.11111111111111, 51.11111111111111, 77.77777777777779, 86.66666666666667],
                                [93.33333333333333, 43.333333333333336, 88.33333333333333, 86.66666666666667],
                                [94.66666666666667, 64.0, 88.0, 88.0]],
                               [[80.0, 46.666666666666664, 60.0, 93.33333333333333],
                                [90.0, 66.66666666666666, 76.66666666666667, 96.66666666666667],
                                [95.55555555555556, 71.11111111111111, 88.88888888888889, 95.55555555555556],
                                [95.0, 68.33333333333333, 91.66666666666666, 96.66666666666667],
                                [94.66666666666667, 68.0, 93.33333333333333, 96.0]]]
        classification_accuracies = [[[91.85185185185185, 54.074074074074076, 88.88888888888889, 94.81481481481482],
                                     [95.0, 33.33333333333333, 64.16666666666667, 93.33333333333333],
                                     [97.14285714285714, 56.19047619047619, 93.33333333333333, 96.19047619047619],
                                     [97.77777777777777, 48.888888888888886, 94.44444444444444, 96.66666666666667],
                                     [97.33333333333334, 70.66666666666667, 97.33333333333334, 96.0]],
                                     [[81.48148148148148, 51.11111111111111, 60.0, 91.85185185185185],
                                      [88.33333333333333, 33.33333333333333, 64.16666666666667, 88.33333333333333],
                                      [91.42857142857143, 55.23809523809524, 80.0, 97.14285714285714],
                                      [92.22222222222223, 40.0, 88.88888888888889, 93.33333333333333],
                                      [94.66666666666667, 65.33333333333333, 88.0, 93.33333333333333]],
                                     [[97.03703703703704, 46.666666666666664, 65.92592592592592, 97.77777777777777],
                                      [95.0, 59.166666666666664, 64.16666666666667, 98.33333333333333],
                                      [97.14285714285714, 63.8095238095238, 96.19047619047619, 99.04761904761905],
                                      [97.77777777777777, 64.44444444444444, 93.33333333333333, 98.88888888888889],
                                      [97.33333333333334, 64.0, 94.66666666666667, 100.0]]]

        for i, accuracies in \
                enumerate(zip(classification_accuracies, training_accuracies)):
            classification_accuracies_itr = accuracies[0]
            training_accuracies_itr = accuracies[1]
            KNN_accuracies = [x[0] for x in classification_accuracies_itr]
            KNN_train_accuracies = [x[0] for x in training_accuracies_itr]
            SVM_poly_accuracies = [x[1] for x in classification_accuracies_itr]
            SVM_poly_train_accuracies = [x[1] for x in training_accuracies_itr]
            SVM_rbf_accuracies = [x[2] for x in classification_accuracies_itr]
            SVM_rbf_train_accuracies = [x[2] for x in training_accuracies_itr]
            GNB_accuracies = [x[3] for x in classification_accuracies_itr]
            GNB_train_accuracies = [x[3] for x in training_accuracies_itr]
            plt.plot(test_sizes, KNN_accuracies, label="KNN Classification Accuracies")
            plt.plot(test_sizes, SVM_poly_accuracies,label="SVM-Poly Classification Accuracies")
            plt.plot(test_sizes, SVM_rbf_accuracies,label="SVM-rbf Classification Accuracies")
            plt.plot(test_sizes, GNB_accuracies,label="Gaussian Classification Accuracies")
            plt.xlabel("% train size")
            plt.ylabel("Average Classification Accuracies")
            if i == 0:
                plt.title("Supervised Classification without Dimensionality Reduction on IRIS")
            elif i == 1:
                plt.title("Supervised Classification with PCA Reduction on IRIS")
            elif i == 2:
                plt.title("Supervised Classification with LDA Reduction on IRIS")
            plt.legend()
            plt.show()
            plt.figure()
            plt.plot(test_sizes, GNB_train_accuracies, label="Gaussian Training Accuracies")
            plt.plot(test_sizes, KNN_train_accuracies, label="KNN Training Accuracies")
            plt.plot(test_sizes, SVM_poly_train_accuracies, label="SVM-Poly Training Accuracies")
            plt.plot(test_sizes, SVM_rbf_train_accuracies, label="SVM-rbf Training Accuracies")
            plt.xlabel("% train size")
            plt.ylabel("Average Training Accuracies")
            if i == 0:
                plt.title("Supervised Classification without Dimensionality Reduction on IRIS")
            elif i == 1:
                plt.title("Supervised Classification with PCA Reduction on IRIS")
            elif i == 2:
                plt.title("Supervised Classification with LDA Reduction on IRIS")
            plt.legend()
            plt.show()


def main():
    # Driver for pines and iris data analysis/classification
    filepath = "data/indianR.mat"
    #iris_analysis(display=True)
    #pines_analysis(filepath, display=True)
    #iris_classification(run_pca=True, run_lda=False)
    #iris_classification(run_pca=False, run_lda=True)
    #iris_classification(run_pca=False, run_lda=False)
    pines_classification(filepath, run_pca=False, run_lda=False)
    pines_classification(filepath, run_pca=True, run_lda=False)
    pines_classification(filepath, run_pca=False, run_lda=True)
    #plot_classification(iris=True, pines=False)


if __name__ == "__main__":
    main()
