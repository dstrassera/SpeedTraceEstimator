from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class Classification():

    def __init__(self):
        # data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # model
        self.clf = None

    def getX_test(self):
        return self.X_test

    def gety_test(self):
        return self.y_test

    def loadData(self, data, test_ratio=0.20):
        # load data set
        dataset = fetch_openml(data, version=1)  # "mnist_784"
        X, y = dataset["data"], dataset["target"]
        print('Dataset size X:', X.shape)
        print('Dataset size y:', y.shape)

        # split data in training and test
        # n_split = round(X.shape[0]*split_ratio)
        # self.X_train, self.X_test, self.y_train, self.x_test = X[:n_split], X[n_split:], y[:n_split], y[n_split:]
        # Split data into train and test subsets
        self.X_train, self.X_test, self.y_train, self.x_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)

    def trainModel(self, model=1):
        # training model
        if model == 1:
            self.clf = make_pipeline(StandardScaler(), KNeighborsClassifier())  # knn is a multilabel classifier
            modelType = 'KNN'
        elif model == 2:
            self.clf = make_pipeline(StandardScaler(), tree.DecisionTreeClassifier())
            modelType = 'DecTree'
        elif model == 3:
            self.clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10))
            modelType = 'RandForest'
        elif model == 4:
            self.clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))
            # self.clf = make_pipeline(StandardScaler(), svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo'))
            # self.clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'))
            modelType = 'Support Vectore Machine'
        # Use pipeline to Standardize data
        self.clf.fit(self.X_train, self.y_train)
        print('Model Trained: ', modelType)

    def predict(self, sample, lable, bPrint=0):
        # plot a sample
        some_digit = sample  # X[36000]
        some_digit_label = lable  # y[36000]
        prediction = self.clf.predict([some_digit])
        if(bPrint == 1):
            some_digit_image = some_digit.reshape(28, 28)
            plt.imshow(some_digit_image, cmap='gray')
            plt.show()
            print('sample label:', some_digit_label)
            # test the model on a sample
            print('sample prediction: ', prediction)
        return prediction

    def crossVal(self, k_fold=3, bConfmat=0):
        # cross validation (K-fold)
        k_fold = 3
        # cross val
        y_train_pred = cross_val_predict(self.clf, self.X_train, self.y_train, cv=k_fold)
        print(f"Classification report for classifier {self.clf}:\n"
              f"{metrics.classification_report(self.y_train, y_train_pred)}\n")
        # confusion matrix
        if bConfmat == 1:
            conf_mx = confusion_matrix(self.y_train, y_train_pred)
            # print(conf_mx)
            plt.matshow(conf_mx, cmap='gray')
            plt.show()

            row_sums = conf_mx.sum(axis=1, keepdims=True)
            norm_conf_mx = np.round(conf_mx / row_sums, 3)
            plt.matshow(norm_conf_mx, cmap='gray')
            # print(norm_conf_mx)
            plt.show()
        # cvs = cross_val_score(self.clf, self.X_train, self.y_train, cv=k_fold, scoring='accuracy')
        # print(cvs)

if __name__ == '__main__':
    cl = Classification()
    cl.loadData("mnist_784", test_ratio=0.80)
    cl.trainModel(model=4)
    cl.crossVal()
    # listSample = [6000, 1600, 2600, 3600, 4600]
    # for sample in listSample:
    #     smp = cl.getX_test()[sample]
    #     print(np.shape(smp))
    #     cl.predict(smp, 9, bPrint=1)
