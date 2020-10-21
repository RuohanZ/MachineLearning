import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split




def model_assessment(filename):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """
    vocabList,pd_label = build_vocab_map(filename)
    pd_binary = construct_binary(vocabList,filename)
    pd_count = construct_count(vocabList,filename)
    pd_tfidf = construct_tfidf(vocabList,filename)
    
    label = pd_label.values

    #prepare train-teat data for 3 dataset
    #binary
    binary = pd_binary.values
    xTrain_b, xTest_b, yTrain_b, yTest_b = train_test_split(binary, label, test_size=0.3, random_state=0)
    yTrain_b = yTrain_b.ravel()
    #count
    count = pd_binary.values
    xTrain_c, xTest_c, yTrain_c, yTest_c = train_test_split(count, label, test_size=0.3, random_state=0)
    yTrain_c = yTrain_c.ravel()
    #tfidf
    tfidf = pd_tfidf.values
    xTrain_t, xTest_t, yTrain_t, yTest_t = train_test_split(tfidf, label, test_size=0.3, random_state=0)
    yTrain_t = yTrain_t.ravel()

    #fit in models and calculate auc
    #perceptron
    p = Perceptron()
    error_p_b = model_error(p,xTrain_b,yTrain_b,xTest_b,yTest_b)
    error_p_c = model_error(p,xTrain_c,yTrain_c,xTest_c,yTest_c)
    error_p_t = model_error(p,xTrain_t,yTrain_t,xTest_t,yTest_t)
    #multinominalNB
    m = MultinomialNB()
    error_m_b = model_error(m,xTrain_b,yTrain_b,xTest_b,yTest_b)
    error_m_c = model_error(m,xTrain_c,yTrain_c,xTest_c,yTest_c)
    error_m_t = model_error(m,xTrain_t,yTrain_t,xTest_t,yTest_t)
    #logistic regression
    l = LogisticRegression()
    error_l_b = model_error(l,xTrain_b,yTrain_b,xTest_b,yTest_b)
    error_l_c = model_error(l,xTrain_c,yTrain_c,xTest_c,yTest_c)
    error_l_t = model_error(l,xTrain_t,yTrain_t,xTest_t,yTest_t)

    print("Binary dataset #Mistake:")
    print("Perceptron") 
    print(error_p_b)
    print("MultinominalNB") 
    print(error_m_b)
    print("Logistic regression") 
    print(error_l_b)

    print("Count dataset #Mistake:")
    print("Perceptron") 
    print(error_p_c)
    print("MultinominalNB") 
    print(error_m_c)
    print("Logistic regression") 
    print(error_l_c)

    print("TFIDF dataset #Mistake:")
    print("Perceptron") 
    print(error_p_t)
    print("MultinominalNB") 
    print(error_m_t)
    print("Logistic regression") 
    print(error_l_t)
    
    """
    #Q2(c) 
    vocab = np.array(vocabList)
    max_ind = np.array([1551,20,1550,74,630,1379,138,910,413,156,154,409,145,80,426])
    max_voab = vocab[max_ind]
    print("15 most postive weight word")
    print(max_voab)
    min_ind = np.array([294,357,697,276,344,925,436,2000,1538,1563,1889,1318,2524,217,690])
    min_voab = vocab[min_ind]
    print("15 most negative weight word")
    print(min_voab)
    """

    return None

def model_error(model,xTrain,yTrain,xTest,yTest):
    model.fit(xTrain,yTrain)
    yHat = model.predict(xTest)
    error = calc_mistakes(yHat,yTest)
    return error

def calc_mistakes(yHat, yTrue):
    count = 0
    i = 0
    for i in range(len(yHat)):
        if(yHat[i] != yTrue[i]):
            count+= 1

    return count

def build_vocab_map(filename):
    cnt = 0 #for debug
    words = []
    label = []
    words_dic = {} #{words : #row/email}
    words_dic_30 = {}
    with open(filename) as data:
        for line in data:
            cnt += 1
            #get unique words in a row (separated by " " and pop label 1 & 0)
            words = line.strip().split(' ')
            label.append(words.pop(0))
            row_words = np.unique(np.array(words))
            #update word_dic with new word in this row
            for word in row_words:
                if word in words_dic:
                    words_dic[word] += 1
                else:
                    words_dic[word] = 1
    #print(words_dic)
    #print(cnt)
    #words_dic for words appears in at least 30 emails
    words_dic_30 = {k:v for k,v in words_dic.items() if v > 29}
    pd_label = pd.DataFrame(label,columns=["Label"])
    pd_label.to_csv("label.csv",index=False)
    #return the words in words_dic_30 and label array
    return list(words_dic_30.keys()),pd_label


def construct_binary(vocab,filename):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    #cnt = 0 #for debug
    arr_email =[] #array of dic
    with open(filename) as data:
        for line in data:
            #create dictionary, keys = Label + vocab, update dataframe. 
            email = dict.fromkeys(vocab,0)
            #cnt += 1
            #print(cnt)
            words =line.strip().split(' ')
            words.pop(0)
            words = np.unique(np.array(words))
            for word in words:
                if word in email:
                    email[word] = 1
                else:
                    continue
            arr_email.append(email)
    pd_email = pd.DataFrame(arr_email)
    #print(pd_email)
    pd_email.to_csv("binary.csv", index=False)
    return pd_email


def construct_count(vocab,filename):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    #cnt = 0 #for debug
    arr_email =[] #array of dic
    with open(filename) as data:
        for line in data:
            #create dictionary, keys = Label + vocab, update dataframe. 
            email = dict.fromkeys(vocab,0)
            #cnt += 1
            #print(cnt)
            words =line.strip().split(' ')
            words.pop(0)
            words = np.array(words)
            for word in words:
                if word in email:
                    email[word] += 1
                else:
                    continue
            arr_email.append(email)
    pd_email = pd.DataFrame(arr_email)
    #print(pd_email)
    pd_email.to_csv("count.csv", index=False)
    return pd_email



def construct_tfidf(vocab,filename):
    """
    Construct the email datasets based on
    the TF-IDF representation of the email.
    """
    emails = []
    with open(filename) as data:
        for line in data:
            emails.append(line)
    

    tf = TfidfVectorizer(analyzer='word', vocabulary=vocab)
    tfidf_matrix =  tf.fit_transform(emails)
    feature_names = tf.get_feature_names()#words in vocab
    pd_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns = feature_names)
    pd_tfidf.to_csv("tfidf.csv", index=False)
    
    return pd_tfidf


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    model_assessment(args.data)
   
    
    
 
    



if __name__ == "__main__":
    main()
