from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


class LinearRegression(ABC):
    """
    Base Linear Regression class from which all 
    linear regression algorithm implementations are
    subclasses. Can not be instantiated.
    """
    beta = None      # Coefficients

    @abstractmethod
    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        Train the linear regression and predict the values

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : dictionary
            key refers to the batch number
            value is another dictionary with time elapsed and mse
        """
        pass

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
        # TODO
        yHat = (self.beta.dot(xFeat)).transpose()
        return yHat

    def mse(self, xFeat, y):
        """
        """
        yHat = self.predict(xFeat)
        return mean_squared_error(y, yHat)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
