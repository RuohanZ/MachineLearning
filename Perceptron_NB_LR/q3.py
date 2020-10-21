import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

def calc_mistakes(yHat, yTrue):
    count = 0
    i = 0
    for i in range(len(yHat)):
        if(yHat[i] != yTrue[i]):
            count+= 1

    return count

def file_to_numpy(filename):
    
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("X",
                        help="filename of the dataset")
    parser.add_argument("--Y",
                        default="label.csv",
                        help="filename of the lable file")
    args = parser.parse_args()
    x = file_to_numpy(args.X)
    y = file_to_numpy(args.Y)

    #split trani test data: (7:3)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

    yTrain = yTrain.ravel()

    #multinominalNB
    nb_M = MultinomialNB()
    nb_M.fit(xTrain,yTrain)
    yHat_M = nb_M.predict(xTest)
    error_M = calc_mistakes(yHat_M,yTest)
    print("MultinominalNB mistake:")
    print(error_M)
    #ComplementNB
    nb_C = ComplementNB()
    nb_C.fit(xTrain,yTrain)
    yHat_C = nb_C.predict(xTest)
    error_C = calc_mistakes(yHat_C,yTest)
    print("ComplementNB mistake:")
    print(error_C)
    #BernoulliNB
    nb_B = BernoulliNB()
    nb_B.fit(xTrain,yTrain)
    yHat_B = nb_B.predict(xTest)
    error_B = calc_mistakes(yHat_B,yTest)
    print("BernoulliNB mistake:")
    print(error_B)

    #logistic regression
    lr = LogisticRegression()
    lr.fit(xTrain,yTrain)
    yHat_l = lr.predict(xTest)
    error_l = calc_mistakes(yHat_l,yTest)
    print("Logistic regression mistake:")
    print(error_l)


if __name__ == "__main__":
    main()

