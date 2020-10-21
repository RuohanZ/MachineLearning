import argparse
import numpy as np
import pandas as pd
import time
#I use this to shuffle two np array together only
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr #learning rate
        self.bs = bs #batch
        self.mEpoch = epoch #max iterate

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        timeElapse = 0
        # TODO: DO SGD
        #add a vector of 1 to xTrain and xTest for intercept term
        ints = np.ones(shape=yTrain.shape)
        xTrain = np.concatenate((ints, xTrain), axis=1)
        ints = np.ones(shape=yTest.shape)
        xTest = np.concatenate((ints,xTest),axis=1)
        #randomly initialize beta, shape of beta 1*d
        self.beta = np.random.rand(1,xTrain.shape[1])
        #initialize dictionary
        start_time1 = time.time()
        x_train = xTrain.transpose()
        x_test = xTest.transpose()
        train_mse = self.mse(x_train,yTrain)
        test_mse = self.mse(x_test,yTest)
        end_time1 = time.time()
        timeElapse1 = end_time1 - start_time1
        dic = {'time': timeElapse1,
               'train-mse': train_mse,
               'test-mse': test_mse}
        trainStats.update({0: dic})
        
        epoch = 1
        b = 1
        plotTrain = []
        plotTest = []
        plotTime = []
        start_time2 = time.time()
        while epoch < self.mEpoch+1:
            #random shufflre taining data and break into N/bs batches
            X, Y= shuffle(xTrain,yTrain)
            Bx = np.array_split(X, int(len(X)/self.bs))
            By = np.array_split(Y, int(len(X)/self.bs))
            while b < len(Bx):
                x_feat = Bx[b].transpose()
                y_hat = self.predict(x_feat)
                gradient = (np.subtract(By[b], y_hat)).transpose().dot(Bx[b])
                gradient = gradient.dot(1/len(Bx[b]))
                #print(self.beta.shape) -- 1,10
                #print(gradient.shape)-- 1,10
                self.beta = np.add(self.beta,gradient.dot(self.lr))
                #train_prdict the train and test data and update dictory
                x_Train = Bx[b].transpose()
                train_mse1 = self.mse(x_Train,By[b])
                test_mse1 = self.mse(x_test,yTest)
                end_time2 = time.time()
                timeElapse2 = end_time2 - start_time2
                itr = epoch*b
                dic = {'time': timeElapse2,
                        'train-mse': train_mse1,
                        'test-mse': test_mse1}
                trainStats.update({itr:dic})
                b+=1 
            b=1
            epoch += 1
            end_time3 = time.time()
            timeElapse3 = end_time3 - start_time2
            xPlot = xTrain.transpose()
            plotTrain.append(self.mse(xPlot,yTrain))
            plotTest.append(self.mse(x_test,yTest))
            plotTime.append(timeElapse3)
            
        #Q4
        plt.plot(plotTime,plotTest)
        #plt.plot(plotTime,plotTrain)

        #Add Close form point
        #plt.plot(0.007259845733642578, 0.3590447177262758, 'ro')
        plt.plot(0.007259845733642578, 0.30271971818002, 'ro')

        plt.ylabel('Test MSE')
        #plt.ylabel('Train MSE')
        plt.xlabel('Time')
        plt.title(self.bs)
        plt.show()

        '''
        # Q3 graph
        plt.plot(plotTrain)
        plt.ylabel('Train MSE')
        plt.show()
        plt.plot(plotTest)
        plt.ylabel('Test MSE')
        plt.show()
        '''
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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)


    

    # setting the seed for deterministic behavior
    np.random.seed(args.seed) 
    #for Q3b, train_test_split to get a radom 40% of xTrain and yTrain
    #a, x_Q3b, b, y_Q3b= train_test_split(xTrain, yTrain, test_size=0.4)
    model = SgdLR(args.lr, args.bs, args.epoch)
    #trainStats = model.train_predict(x_Q3b, y_Q3b, xTest, yTest)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)
    

if __name__ == "__main__":
    main()

