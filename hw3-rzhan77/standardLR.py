import argparse
import numpy as np
import pandas as pd
import time
from numpy.linalg import inv

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        timeElapse = 0
        # TODO: DO SOMETHING
        start_time = time.time()
        #add a vector of 1 to xTrain and xTest for intercept term
        int = np.ones(shape=yTrain.shape)
        xTrain = np.concatenate((int, xTrain), axis=1)
        int = np.ones(shape=yTest.shape)
        xTest = np.concatenate((int,xTest),axis=1)
        #calculate coefficients using closed-form solution
        self.beta = inv(xTrain.transpose().dot(xTrain)).dot(xTrain.transpose()).dot(yTrain)
        #beta and xTrain/Test to 1*10 ans 10*n
        self.beta = self.beta.transpose()
        xTrain = xTrain.transpose()
        xTest = xTest.transpose()
        train_mse = self.mse(xTrain,yTrain)
        test_mse = self.mse(xTest,yTest)
        end_time = time.time()
        timeElapse = end_time - start_time
        #update trainStates
        dic = {'time': timeElapse,
               'train-mse': train_mse,
               'test-mse': test_mse}
        trainStats.update({0: dic})
        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
