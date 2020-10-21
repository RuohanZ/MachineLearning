"""
CS 334 Machine Learning
Ruohan Zhang
"""


import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.xTrain = xFeat
        self.yTrain = y


        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO

        xTest  = pd.DataFrame(xFeat)
        xTrain = pd.DataFrame(self.xTrain)
        yDist = pd.DataFrame(self.yTrain) #Dataframe that xTran's contain label and distance from xTest
        for x2 in xTest.iterrows():
            distances = []
            numOfOne = 0
            numOfZero = 0
            #Calculate the distances
            for x1 in xTrain.iterrows():
                dist = np.linalg.norm(x1[1]-x2[1])
                distances.append(dist)
            #Find the nearest k neighbours
            yDist['dist']= distances
            yDist=yDist.sort_values('dist')
            for j in range(self.k):
                if yDist.iloc[j,0]== 1:
                    numOfOne += 1
                else:
                    numOfZero += 1
            if numOfOne >= numOfZero:
                yHat.append(1)
            else:
                yHat.append(0)
            yDist=yDist.sort_index()
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    correct = 0;
    for i,y in enumerate(yHat):
        if y == yTrue[i]:
            correct += 1
    acc = correct / len(yHat)
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
