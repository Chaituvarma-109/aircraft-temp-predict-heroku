from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd


class Model:
    """
    This class builds the linear regression model for the Aircraft Maintenance Prediction.
    """

    def __init__(self, file):
        """
        Initializer for Arguments.
        Args:
            file: Name of the file to predict
        """
        self.file = file
        self._file_df = ''
        self._y = ''
        self._X = ''
        self._reg = ''
        self.read_file()

    def read_file(self):
        """
        Reads the csv file.
        Returns:
            returns the pandas dataframe from the csv file
        """
        self._file_df = pd.read_csv(self.file)
        self.separation_of_features()

    def separation_of_features(self):
        """
        Separates the Independent and Dependent features.
        Returns:
            returns nothing.
        """
        self._y = self._file_df[['Air temperature [K]']]
        self._X = self._file_df[
            ['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF',
             'OSF', 'RNF']]
        self.ml_model()

    @staticmethod
    def adjusted_r_square(x, y, score):
        """
        Calculates the adj r-squared score for linear regression.
        Args:
            x: X_test data sample
            y: y_test data sample
            score: r-squared score

        Returns:
            returns the Adj r-squared score
        """
        r2 = score
        n = x.shape[0]
        p = x.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2

    def ml_model(self):
        """
        Applies Linear Regression Algorithm using scikit learn.
        Returns:
            returns the r-square and adj r-squared scores calculated from regression model.
        """
        X_train, X_test, y_train, y_test = train_test_split(self._X.values, self._y, test_size=0.15, random_state=49)
        self._reg = LinearRegression().fit(X_train, y_train)
        reg_score = self._reg.score(X_test, y_test)
        adj_r2 = self.adjusted_r_square(X_test, y_test, reg_score)
        return reg_score, adj_r2

    def prediction(self, pred_inp):
        """
        Predicts the air temperature based on trained model.
        Args:
            pred_inp: a 2D list with integers or floats act as input for the Model.

        Returns:
            returns the predicted air temperature.
        """
        res = self._reg.predict(pred_inp)
        return res
