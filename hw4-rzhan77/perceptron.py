import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron, 

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        self.w = np.zeros(1+xFeat.shape[1])
        stats = {} # epoch:error
        error = 0
        # TODO implement this
        for i in range(1,self.mEpoch+1):
            error = 0
            xFeat,y = shuffle(xFeat,y)
            for xi, label in zip(xFeat,y):
                update = label - self.predict(xi)
                self.w[1:] += update * xi
                self.w[0] += update
                error += int(update != 0.0)
            stats.update({i:error})
            if error == 0:
                break
    
            


        return stats

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
            Predicted response per sample
        """
        #w[0]: weight for bias, w[1:]:weight for features
        yHat = []
        wx = np.dot(xFeat,self.w[1:])+self.w[0]
        yHat = np.where(wx >= 0.0,1,0)
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    count = 0
    i = 0
    for i in range(len(yHat)):
        if(yHat[i] != yTrue[i]):
            count+= 1

    return count


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    """
    parser.add_argument("X",
                        help="filename for features dataset")
    parser.add_argument("Y",
                        help="filename for labels associated with data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    x = file_to_numpy(args.X)
    y = file_to_numpy(args.Y)
    #split trani test data: (7:3)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)
    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))
    #print(yHat[:20])

    #indices largest 15 postive weight
    max_idx = (-model.w).argsort()[:15]
    print("Indices of 15 most postive")
    print(max_idx)
    #print(model.w[max_idx])

    #indices smallest 15 negative weight
    min_idx = (model.w).argsort()[:15]
    print("Indices of 15 most negative")
    print(min_idx)
    #print(model.w[min_idx])

    


if __name__ == "__main__":
    main()