import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

def file_to_numpy(filename):
    
    df = pd.read_csv(filename)
    return df.to_numpy()

def logisticRegr_roc(xTrain,yTrain,xTest,yTest):
    lr = LogisticRegression(C=1e42)
    lr.fit(xTrain,yTrain)
    yproba = lr.predict_proba(xTest)[::,1]
    fpr, tpr, _ = roc_curve(yTest,  yproba)
    auc = roc_auc_score(yTest, yproba)
   
    return fpr,tpr,auc


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
    # normalize
    normed_xTrain = normalize(xTrain, axis=0)
    normed_xTest = normalize(xTest, axis=0)

    #PCA
    pca = PCA(0.95)
    pca.fit(normed_xTrain)
    #print(abs( pca.components_ ))
    #print(sum(pca.explained_variance_ratio_))
    pca_xTrain = pca.transform(normed_xTrain)
    pca_xTest = pca.transform(normed_xTest)

    #NMF
    nmf = NMF(n_components = 6)
    nmf.fit(xTrain)
    nmf_xTrain = nmf.transform(xTrain)
    nmf_xTest = nmf.transform(xTest)
    



    roc_table = pd.DataFrame(columns=['dataset', 'fpr','tpr','auc'])

    fpr,tpr,auc = logisticRegr_roc(normed_xTrain,yTrain,normed_xTest,yTest)
    roc_table = roc_table.append({'dataset':'Normalize',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    fpr,tpr,auc = logisticRegr_roc(pca_xTrain,yTrain,pca_xTest,yTest)
    roc_table = roc_table.append({'dataset':'PCA',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    fpr,tpr,auc = logisticRegr_roc(nmf_xTrain,yTrain,nmf_xTest,yTest)
    roc_table = roc_table.append({'dataset':'NMF',
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    roc_table.set_index('dataset', inplace=True)

    print(roc_table)
    

    #plot ROC 
    fig = plt.figure(figsize=(8,6))

    for i in roc_table.index:
        plt.plot(roc_table.loc[i]['fpr'], 
                roc_table.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, roc_table.loc[i]['auc']))
    
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()








if __name__ == "__main__":
    main()