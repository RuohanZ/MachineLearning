import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn import preprocessing


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    #split the date and time; drop date; change time to 24h scale (e.g. 02:30 = 2.5)
    timeStamps = df.iloc[:,0].values
    times = []
    day = []
    month = []
    for time in timeStamps:
        split = time.split(" ")
        date= split[0].split("/")
        timestamp = split[1].split(":")
        month.append(int(date[0]))
        day.append(int(date[1]))
        times.append(int(timestamp[0])+int(timestamp[1])/60)   
    df['month'] = month
    df['day'] = day
    df['time'] = times
    df = df.drop(columns=['date'])
    return df


def select_features(df,y=None):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data
    y: target column of feature df
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO
    df['label'] = y
    # Use Pearson Correlation
    cor = df.corr(method ='pearson')
    '''
    #draw heatmap
    ax = sns.heatmap(cor, annot=True, annot_kws={"size": 5},cmap=plt.cm.Reds,xticklabels=True, yticklabels=True)
    print(ax.get_ylim())
    ax.set_ylim(29.0,0)
    plt.show()
    '''
    # drop features poorly correlated to label, cor_label < 0.1
    cor_label = abs(cor["label"])
    irrelevant_features = cor_label[cor_label<0.1].index
    for i in range(len(irrelevant_features)):
        cor = cor.drop(columns=[irrelevant_features[i]])
        cor = cor.drop(index=[irrelevant_features[i]])
        df = df.drop(columns=[irrelevant_features[i]])
    #drop one of the features highly correlated to one another
    #select highly correlated features, cor_feature > 0.7
    feature_drop = []
    for col in cor.columns:
        cor_label = {}
        cor_feature = abs(cor[col])
        related_features = cor_feature[cor_feature>0.7].index
        if len(related_features) ==1:
            continue
        else:
            for feature in related_features:
                cor_label.update({feature:abs(cor.loc["label",feature])})
        
        #select half of related_feature with smaller cor_label, save features names in feature_drop
        for i in range(int(len(cor_label)/2)):
            smallest = min(cor_label, key=lambda k: cor_label[k])
            if smallest not in feature_drop:
                feature_drop.append(smallest)
            del cor_label[smallest]

    #drop features in feature_drop
    for  i in range(len(feature_drop)):
        cor = cor.drop(columns=[feature_drop[i]])
        df = df.drop(columns=[feature_drop[i]])
    #drop the label column
    df = df.drop(columns=['label'])
    
    #return all dropped features names for test data feature selection
    feature_drop = feature_drop +list(irrelevant_features)

    return df , feature_drop


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    stdScale = preprocessing.StandardScaler().fit(trainDF[trainDF.columns])
    trainDF[trainDF.columns] = stdScale.transform(trainDF[trainDF.columns])
    testDF[testDF.columns] = stdScale.transform(testDF[testDF.columns])
    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--yTrain",
                        default="eng_yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    parser.add_argument("--yTest",
                        default="eng_yTest.csv",
                        help="filename for labels associated with training data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    yTrain = pd.read_csv(args.yTrain)
    yTest = pd.read_csv(args.yTest)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain, feature_drop= select_features(xNewTrain,yTrain)
    #xNewTest = select_features(xNewTest,yTest)
    # select test feature according to train's feature
    for  i in range(len(feature_drop)):
        xNewTest = xNewTest.drop(columns=[feature_drop[i]])
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
