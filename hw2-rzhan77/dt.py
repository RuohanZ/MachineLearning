import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        #tree representation: {question:[less/{question:[ , ],more/{questio:[ , ]}
        self.tree={}

    def train(self, xFeat, y=None):
        """
        Train the decision tree model.

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
        trainData = xFeat.join(y)
        tree = self.decisionTree(trainData,self.maxDepth,self.minLeafSample)
        self.tree = tree;
        return self


    def decisionTree(self,data, maxDepth, minLeafSample, depth=0):
        #change pd dataframe to numpy array to np.unique()
        if depth == 0:
            data =data.values
        else:
            data = data
        # base
        if (isPure(data)) or (depth == maxDepth) or (len(data) < minLeafSample):
            ans = classify(data)
            return ans
        # recursive
        else:
            depth += 1
            potential_splits = potentialSplits(data)
            if self.criterion == 'gini':
                col, splitPoints = giniBestSplit(data, potential_splits)
            else:
                col, splitPoints = entropyBestSplit(data, potential_splits)
            lessSet, moreSet = split(data, col, splitPoints)
            question = "{} <= {}".format(col, splitPoints)
            subTree = {question: []}
            # grow leaves
            lessAns = self.decisionTree(lessSet, maxDepth,minLeafSample,depth)
            moreAns = self.decisionTree(moreSet, maxDepth,minLeafSample,depth)
            subTree[question].append(lessAns)
            subTree[question].append(moreAns)
            return subTree

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
        tree = self.tree
        xText = xFeat.values
        for i in range(xText.shape[0]):
            x = xText[i,:]
            answer = self.singlePredict(x,tree)
            yHat.append(answer)
        return yHat

    def singlePredict(self,data,tree):
        # tree representation: {question:[less/{question:[ , ],more/{questio:[ , ]}
        #question: colIndex <= splitPoint
        question = list(tree.keys())[0]
        colIndex,operater,splitPoint = question.split(" ")
        if data[int(colIndex)] <= float(splitPoint):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
        #base when answer no nolonger a dictionary
        if not isinstance(answer,dict):
            return answer
        #recursive
        else:
            resTree = answer
            return self.singlePredict(data,resTree)

# Helper function for growing tree
# functions for split
def potentialSplits(data):
    # splits {colIndex,[average values of every two unique values]}
    splits ={}
    colNum = data.shape[1]
    for colIndex in range(colNum -1):
        splits[colIndex] = []
        uniqueValues = np.unique(data[:,colIndex])
        #calculate average of every adjacent unique values exclude before first
        for i in range(len(uniqueValues)):
            if i != 0:
                pre = uniqueValues[i-1]
                cur = uniqueValues[i]
                split = (pre + cur)/2
                splits[colIndex].append(split)
    return splits

def split(data, colIndex, splitPoint ):
        colValues = data[:,colIndex]
        lessSet = data[colValues <= splitPoint]
        moreSet = data[colValues > splitPoint]
        return lessSet,moreSet

def dtEntropy(lessSet,moreSet):
        #lessSet entropy
        labelCol1 = lessSet[:,-1]
        values,counts1 = np.unique(labelCol1,return_counts=True)
        entropy1 = entropy(counts1,base = None)
        #moreSet entropy
        labelCol2 = moreSet[:,-1]
        values,counts2 = np.unique(labelCol2,return_counts=True)
        entropy2 = entropy(counts2,base = None)
        #entropy of the split
        N = len(lessSet)+len(moreSet)
        entropyAll = len(lessSet)/N * entropy1 + len(moreSet)/N * entropy2
        return entropyAll

def entropyBestSplit(data,potentialSplits):
        smallest = 9999
        for colIndex in potentialSplits:
            for splitPoint in potentialSplits[colIndex]:
                lessSet,moreSet= split(data,colIndex,splitPoint)
                splitEntropy = dtEntropy(lessSet,moreSet)
                if splitEntropy <= smallest:
                    smallest = splitEntropy
                    bestCol = colIndex
                    bestSP = splitPoint
        return bestCol,bestSP

def gini(lessSet,moreSet):
        #lessSet gini
        labelCol1 = lessSet[:, -1]
        values, counts1 = np.unique(labelCol1, return_counts=True)
        gini1 = 1-sum(counts1/counts1.sum())
        #moreSet gini
        labelCol2 = moreSet[:,-1]
        values,counts2 = np.unique(labelCol2,return_counts=True)
        gini2 = 1-sum(counts2/counts2.sum())
        N = len(lessSet) + len(moreSet)
        giniAll = len(lessSet) / N * gini1 + len(moreSet) / N * gini2
        return giniAll

def giniBestSplit(data, potentialSplits):
        smallest = 9999
        for colIndex in potentialSplits:
            for splitPoint in potentialSplits[colIndex]:
                lessSet, moreSet = split(data, colIndex, splitPoint)
                splitGini = gini(lessSet, moreSet)
                if splitGini <= smallest:
                    smallest = splitGini
                    bestCol = colIndex
                    bestSP = splitPoint
        return bestCol, bestSP
# functions to check leaf purity
def classify(data):
        labelCol = data[:,-1]
        values,counts = np.unique(labelCol,return_counts=True)
        major = counts.argmax()
        result = values[major]
        return result

def isPure(data):
        labelCol = data[:, -1]
        values, counts = np.unique(labelCol, return_counts=True)
        if len(values)==1:
            return True
        else:
            return False





def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    """
    #for 3d plot
    plotDF = pd.DataFrame(columns=['maxdepth', 'minleaf', 'giniTrain', 'giniTest','entropyTrain','entropyTest'])
    for i in range(1,10):
        for j in range(1,10):
            dt3 = DecisionTree('gini', i, j)
            trainAcc3, testAcc3 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
            dt4 = DecisionTree('entropy',i,j)
            trainAcc4,testAcc4 = dt_train_test(dt4,xTrain,yTrain,xTest,yTest)
            plotDF = plotDF.append({'maxdepth': i, 'minleaf': j, 'giniTrain': trainAcc3,'giniTest': testAcc3,'entropyTrain': trainAcc4,'entropyTest': testAcc4}, ignore_index=True)
    print(plotDF)
    one = plt.figure().gca(projection='3d')
    one.scatter(plotDF['maxdepth'], plotDF['minleaf'], plotDF['entropyTrain'])
    one.set_xlabel('maxdepth')
    one.set_ylabel('minleaf')
    one.set_zlabel('entropyTrain')
    two = plt.figure().gca(projection='3d')
    two.scatter(plotDF['maxdepth'], plotDF['minleaf'], plotDF['entropyTest'])
    two.set_xlabel('maxdepth')
    two.set_ylabel('minleaf')
    two.set_zlabel('entropyTest')
    plt.show()
    """
if __name__ == "__main__":
    main()
