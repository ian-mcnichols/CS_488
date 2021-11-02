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
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline


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


def iris_classification(run_pca=True):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    plot_data = {}
    if run_pca:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    test_sizes = [.1, .2, .3, .4, .5]
    for test_size in test_sizes:
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_size,random_state=1,shuffle=True)
        models = [('KNN', KNeighborsClassifier()), ('SVM-Poly', SVC(gamma='auto', kernel='poly')),
                  ('SVM-RBF', SVC(gamma='auto', kernel='rbf')), ('NB', GaussianNB())]
        names = []
        classification_accuracy = []
        training_accuracy = []
        for name, model in models:
            model.fit(X_train, Y_train)
            classification_accuracy.append(model.score(X_validation, Y_validation)*100)

            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
            training_accuracy.append(cv_results.mean()*100)
            names.append(name)
        plot_data[test_size] = [np.mean(training_accuracy), np.mean(classification_accuracy)]
    print("plot_data:", plot_data)
    test_sizes = [x*100 for x in test_sizes]
    training_accuracies = [plot_data[x][0] for x in plot_data]
    print("Training accuracies:", training_accuracies)
    classification_accuracies = [plot_data[x][1] for x in plot_data]
    print("Classification accuracies:", classification_accuracies)
    plt.scatter(test_sizes,training_accuracies,c='blue',label="Training accuracy")
    plt.scatter(test_sizes,classification_accuracies,c='green',label="Classification accuracy")
    plt.xlabel("% test size")
    plt.ylabel("Average accuracy")
    if run_pca:
        plt.title("Supervised Classification with PCA Reduction on IRIS")
    else:
        plt.title("Supervised Classification without Dimensionality Reduction on IRIS")
    plt.legend()
    plt.show()


def pines_classification(R_filepath, run_pca=False):
    R_file = io.loadmat(R_filepath)
    gth = np.array(R_file['gth'])
    X = np.array(R_file['X']).transpose()
    R_rows = R_file['num_rows']
    R_cols = R_file['num_cols']
    R_bands = R_file['num_bands']
    gth = np.reshape(gth, (int(R_rows)*int(R_cols)))
    bands, samples = X.shape
    gth_mat = io.loadmat('data/indian_gth.mat')
    gth_mat = {i:j for i, j in gth_mat.items() if i[0] != "_"}
    gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})

    if run_pca:
        #data_file = open("pca_pines_classification.txt", "w")
        pca = PCA(n_components=10)
        principleComponents = pca.fit_transform(X)
        x1 = X.transpose()
        X = np.matmul(x1, principleComponents)
        #X = X.transpose()
        print("X:", principleComponents.shape)
        print("gth:", gth.shape)
        X = principleComponents
        print("Running with dimensionality reduction")
    else:
        #data_file = open("pines_classification.txt", "w")
        print("X:", X.shape)
        print("gth:", gth.shape)
        print("Running without dimensionality reduction")
    plot_data = {}
    test_sizes = [.5, .4, .3, .2, .1]
    test_sizes = [.3]
    for test_size in test_sizes:
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, gth, test_size=test_size,
                                                                        random_state=1, shuffle=True)
        models = [('NB', GaussianNB()), ('KNN', KNeighborsClassifier()), ('SVM-Poly', SVC(gamma='auto', kernel='poly', cache_size=5200000)),
                  ('SVM-RBF', SVC(gamma='auto', kernel='rbf', cache_size=5200000))]
        #models = [('KNN', KNeighborsClassifier()), ('NB', GaussianNB())]
        names = []
        classification_accuracy = []
        training_accuracy = []
        for name, model in models:
            pipeline = Pipeline([("scaler", MinMaxScaler()), ("classifier", model)])
            pipeline.fit(X_train, Y_train)
            disp = plot_confusion_matrix(pipeline, X_validation, Y_validation, cmap=plt.cm.Blues)
            true_positive = disp.confusion_matrix[1][1]
            false_negative = disp.confusion_matrix[1][0]
            true_negative = disp.confusion_matrix[0][0]
            false_positive = disp.confusion_matrix[0][1]
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            classification_score = pipeline.score(X_validation, Y_validation)
            classification_accuracy.append(classification_score*100)

            #kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            #cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
            #training_accuracy.append(cv_results.mean()*100)
            #names.append(name)

            #data_line = "test size: " + str(test_size) + " model: " + str(name) + " training accuracy: " + str(cv_results.mean()*100) + " classification accuracy: " + str(classification_score*100) 
            #data_line += " sensitivity: " + str(sensitivity) + " specificity: " + str(specificity) + "\n"
            data_line = "test size: " + str(test_size) + " model: " + str(name) +  " classification accuracy: " + str(classification_score*100) 
            data_line += " sensitivity: " + str(sensitivity) + " specificity: " + str(specificity) + "\n"
            
            print(data_line)
            #data_file.writelines("test size: " + str(test_size) + " model: " + str(name) + " training accuracy: " + str(cv_results.mean()*100) + " classification accuracy: " + str(classification_score*100) + "\n")
        plot_data[test_size] = [np.mean(training_accuracy), np.mean(classification_accuracy)]
    #data_file.close()
    print("plot_data:", plot_data)
    test_sizes = [x*100 for x in test_sizes]
    training_accuracies = [plot_data[x][0] for x in plot_data]
    print("Training accuracies:", training_accuracies)
    classification_accuracies = [plot_data[x][1] for x in plot_data]
    print("Classification accuracies:", classification_accuracies)
    plt.scatter(test_sizes,training_accuracies,c='blue',label="Training accuracy")
    plt.scatter(test_sizes,classification_accuracies,c='green',label="Classification accuracy")
    plt.xlabel("% test size")
    plt.ylabel("Average accuracy")
    if run_pca:
        plt.title("Supervised Classification with PCA Reduction on INDIAN PINES")
    else:
        plt.title("Supervised Classification without Dimensionality Reduction on INDIAN PINES")
    plt.legend()


def main():
    filepath = "data/indianR.mat"
    #iris_analysis(display=False)
    #pines_analysis(filepath, display=True)
    #iris_classification(run_pca=True)
    #iris_classification(run_pca=False)
    pines_classification(filepath, run_pca=False)
    #PCA_(Rdata, Rtruth)


if __name__ == "__main__":
    main()

'''
#test size: 0.5 model: KNN training accuracy: 75.14261268464217 classification accuracy: 71.89194330828498 sensitivity: 0.5714285714285714 specificity: 0.9997753313862053

#test size: 0.5 model: NB training accuracy: 31.097994667399863 classification accuracy: 30.590697231998476 sensitivity: 1.0 specificity: 0.9340909090909091

#test size: 0.4 model: KNN training accuracy: 76.09183087404533 classification accuracy: 72.94887039239 sensitivity: 0.5555555555555556 specificity: 0.9994410285075461

#test size: 0.4 model: NB training accuracy: 30.84422847562684 classification accuracy: 30.071343638525565 sensitivity: 1.0 specificity: 0.9290030211480362

#test size: 0.3 model: KNN training accuracy: 76.4899377087459 classification accuracy: 72.98668357641091 sensitivity: 0.75 specificity: 0.9996290801186943

#test size: 0.3 model: NB training accuracy: 30.47476760854787 classification accuracy: 30.183893468611288 sensitivity: 1.0 specificity: 0.9306122448979591

#test size: 0.2 model: KNN training accuracy: 77.06896551724137 classification accuracy: 73.36504161712247 sensitivity: 0.7142857142857143 specificity: 0.9994419642857143

#test size: 0.2 model: NB training accuracy: 30.06539833531511 classification accuracy: 30.10701545778835 sensitivity: 1.0 specificity: 0.946843853820598

#test size: 0.1 model: KNN training accuracy: 77.21168117991175 classification accuracy: 73.22872087494055 sensitivity: 0.5 specificity: 0.9967141292442497

#test size: 0.1 model: NB training accuracy: 30.165969762862847 classification accuracy: 30.147408464098906 sensitivity: 1.0 specificity: 0.9254658385093167

test size: 0.5 model: NB classification accuracy: 64.24426900028536 sensitivity: 0.9545454545454546 specificity: 0.9977814753189129

test size: 0.5 model: KNN classification accuracy: 69.39979073528013 sensitivity: 0.5 specificity: 1.0

test size: 0.5 model: SVM-Poly classification accuracy: 51.5266812517835 sensitivity: 0.0 specificity: 1.0

test size: 0.5 model: SVM-RBF classification accuracy: 59.18386759250451 sensitivity: 0.0 specificity: 1.0

test size: 0.4 model: NB classification accuracy: 64.29250891795482 sensitivity: 0.9444444444444444 specificity: 0.9986216402481047

test size: 0.4 model: KNN classification accuracy: 69.80975029726515 sensitivity: 0.5555555555555556 specificity: 0.9997138769670959

test size: 0.4 model: SVM-Poly classification accuracy: 51.73602853745541 sensitivity: 0.0 specificity: 1.0

test size: 0.4 model: SVM-RBF classification accuracy: 59.29845422116527 sensitivity: 0.0 specificity: 1.0

test size: 0.3 model: NB classification accuracy: 64.61636017755231 sensitivity: 0.9090909090909091 specificity: 0.9986232216613126

test size: 0.3 model: KNN classification accuracy: 70.57704502219404 sensitivity: 0.7777777777777778 specificity: 0.9996204933586338

test size: 0.3 model: SVM-Poly classification accuracy: 51.8706404565631 sensitivity: 0.0 specificity: 1.0

test size: 0.3 model: SVM-RBF classification accuracy: 59.84464172479391 sensitivity: 0.0 specificity: 1.0

test size: 0.2 model: NB classification accuracy: 64.13793103448275 sensitivity: 0.8888888888888888 specificity: 1.0

test size: 0.2 model: KNN classification accuracy: 71.70035671819262 sensitivity: 0.75 specificity: 1.0

test size: 0.2 model: SVM-Poly classification accuracy: 51.7717003567182 sensitivity: 0.0 specificity: 1.0

test size: 0.2 model: SVM-RBF classification accuracy: 59.80975029726516 sensitivity: 0.0 specificity: 1.0

test size: 0.1 model: NB classification accuracy: 64.76462196861627 sensitivity: 0.8 specificity: 1.0

test size: 0.1 model: KNN classification accuracy: 72.42035187826914 sensitivity: 0.6 specificity: 1.0

test size: 0.1 model: SVM-Poly classification accuracy: 52.543984783642415 sensitivity: 0.0 specificity: 1.0

test size: 0.1 model: SVM-RBF classification accuracy: 61.00808368996672 sensitivity: 0.0 specificity: 1.0
'''
