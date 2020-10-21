import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time


def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout.

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO fill int
    start_time = time.time()
    X = xFeat.values
    Y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize)
    md = model.fit(X_train,y_train)
    yPredictTrain = md.predict_proba(X_train)
    yPredictTest = md.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_train,yPredictTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(y_test,yPredictTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    end_time = time.time()
    timeElapsed = end_time - start_time
    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):

    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    count = 0
    # TODO FILL IN

    kfold = KFold(k,True,1)
    for train,test in kfold.split(xFeat,y):
        md = model.fit(xFeat.iloc[train], y.iloc[train])
        yPredictTest = md.predict_proba(xFeat.iloc[test])
        yPredictTrain = md.predict_proba(xFeat.iloc[train])
        fpr, tpr, thresholds = metrics.roc_curve(y.iloc[train], yPredictTrain[:, 1])
        trainAuc += metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], yPredictTest[:, 1])
        testAuc += metrics.auc(fpr, tpr)
        count += 1
    trainAuc =trainAuc/count
    testAuc = testAuc/count

    return trainAuc, testAuc




def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create the classifier


    knnAcc =[]
    for i in range(1,30):
        knnClass = KNeighborsClassifier(i,metric='minkowski',p=2)
        # use 10-fold validation
        aucTrain1, aucVal1= kfold_cv(knnClass, xTrain, yTrain, 10)
        knnAcc.append(aucVal1)
    print(knnAcc)

    dtDF = pd.DataFrame(columns=['maxDepth', 'minleaf', 'ValAUC'])
    for i in (2,10):
        for j in (2,10):
            dtClass = DecisionTreeClassifier(max_depth=i,min_samples_leaf=j)
            aucTrain2, aucVal2 = kfold_cv(dtClass, xTrain, yTrain, 10)
            dtDF['maxDepth']=i
            dtDF['minleaf'] =j
            dtDF['ValAUC'] = aucVal2
    print(dtDF)


if __name__ == "__main__":
    main()