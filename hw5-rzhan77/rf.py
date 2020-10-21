import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    forest = None
    treeFeat = None    #feature selected for each tree

    def __init__(self, nest=0, maxFeat=0, criterion='gini', maxDepth=0, minLeafSample=0):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.forest = []
        self.treeFeat = [] 
        

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        oob = []
        for i in range(self.nest):
            df = xFeat.sample(n=self.maxFeat, axis=1)
            self.treeFeat.append(list(df.columns))
            X,Y = shuffle(df,y)
            xBoot, xOOB, yBoot, yOOB = train_test_split(X, Y, test_size=0.3, random_state=0)
            tree = DecisionTreeClassifier(criterion=self.criterion,max_depth=self.maxDepth,min_samples_leaf=self.minLeafSample)
            tree.fit(xBoot,yBoot)
            self.forest.append(tree)
            yhat = tree.predict(xOOB)
            oobError = 1-accuracy_score(yOOB,yhat)
            oob.append(oobError)
        
        oob_average = sum(oob)/len(oob)
            


        return oob_average

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
        yHat = []
        df_predict = {}
        for i in range(len(self.forest)):
            col_name= "tree_{}".format(i)
            X = xFeat[self.treeFeat[i]]
            yhat = self.forest[i].predict(X)
            df_predict[col_name] = yhat

        df_predict = pd.DataFrame(df_predict)
        yHat = df_predict.mode(axis=1)[0]

        return yHat


#def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
#    df = pd.read_csv(filename)
#    return df.to_numpy()




def main():
    """
    Main file to run from the command line.
    """
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
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    np.random.seed(args.seed)  
    #preprocess data
    scaler = MinMaxScaler()
    scaler.fit(xTrain)
    Xtrain = pd.DataFrame(scaler.transform(xTrain))
    Xtest = pd.DataFrame(scaler.transform(xTest))
    
    maxFeat=[1,2,3,4,5,6,7,8,9,10,11]
    nest = [10,50,100,200,500,800,1000]
    param_predict = []

    for n in nest:
        for m in maxFeat:
            rf = RandomForest(nest = 1,maxFeat=11,criterion='gini',maxDepth=3,minLeafSample=5)
            trainStats = rf.train(Xtrain,yTrain)
            yHat = rf.predict(Xtest)
            acc = accuracy_score(yHat,yTest)
            dic = {'nest': n,
                    'maxFeat': m,
                    'OOBerror': trainStats,
                    'Test Acc': acc}
            param_predict.append(dic)
    param = pd.DataFrame(param_predict)
    #print(param)
    print(param.iloc[param['OOBerror'].argmin()])
    print(param.iloc[param['Test Acc'].argmax()])


    """
    #plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(param['nest'], param['maxFeat'], param['OOBerror'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xlabel("nest")
    ax.set_ylabel("maxFeat")
    ax.set_zlabel("OOBerror")
    ax.set_title("OBBerror for diff neat and #feature")
    plt.show()
    
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.scatter(param['nest'], param['maxFeat'], param['Test Acc'], cmap=plt.cm.viridis, linewidth=0.2)
    ax1.set_xlabel("nest")
    ax1.set_ylabel("maxFeat")
    ax1.set_zlabel("Test Acc")
    ax1.set_title("Test Accuracy for diff neat and #feature")
    plt.show()
    """




if __name__ == "__main__":
    main()