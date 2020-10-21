import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

def file_to_numpy(filename):
    
    df = pd.read_csv(filename)
    return df.to_numpy()

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
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)
    #preprocess data
    scaler = MinMaxScaler()
    scaler.fit(xTrain)
    Xtrain = scaler.transform(xTrain)
    Xtest = scaler.transform(xTest)

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    param_grid = {'xgb__n_estimators': [1000],#[100,200,500,800,1000]
                'xgb__max_depth': [3,4,5,6], 
                'xgb__learning_rate': [0.1,0.5,1,2]
                }
    xgb = XGBClassifier(random_state=0)
    pipe = Pipeline([("scaler",MinMaxScaler()), ("xgb",xgb)])
    grid_xgb = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
    grid_xgb.fit(Xtrain, yTrain.ravel())
    print("Best parameters: {}".format(grid_xgb.best_params_))
    print("Best train accuracy: {:.6f}".format(grid_xgb.best_score_))

    #print("Test set score: {:.3f}".format(grid_xgb.score(Xtest,yTest.ravel())))
    xgb1 = XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=0)
    xgb1.fit(Xtrain,yTrain)
    yHat = xgb1.predict(Xtest)
    print("(100,5,0.1)Accuracy: {:.3f}".format(accuracy_score(yTest,yHat)))

    xgb2 = XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.1,random_state=0)
    xgb2.fit(Xtrain,yTrain)
    yHat = xgb2.predict(Xtest)
    print("(500,6,0.1)Accuracy: {:.3f}".format(accuracy_score(yTest,yHat)))








if __name__ == "__main__":
    main()