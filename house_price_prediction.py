from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model.base import BaseEstimator
from tqdm import tqdm
import numpy as np
import pandas as pd

class housePricePredictor(BaseEstimator):
    """Regressor implementing the k-nearest neighbors that avoids time leakage
    Parameters
    ----------
    k : int, optional (default = 4)
        Number of neighbors to use
    weights : str, optional (default = 'uniform')
        weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
    p : integer, optional (default = 2)
        When p = 1, this is equivalent to using manhattan_distance (l1)
        and euclidean_distance (l2) for p = 2.
    """
    def __init__(self, k=4, weights='distance', p=2):
        self.k = k
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        """Fit k-nearest neighbors model of provided data
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training sample features.
        y : array-like, shape = [n_samples]
            Training target values
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        """Predict the target for the provided data
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test sample features.
        Returns
        -------
        y_pred : array of float, shape = [n_samples]
            Predicted target values
        """
        idx = np.argsort(X_test[:, 2])[::-1]
        X_test = X_test[idx]

        knn = KNeighborsRegressor(self.k, p=self.p, weights=self.weights)
        y_pred = []
        for i in tqdm(range(len(X_test))):
            close_date_i = X_test[i][2]
            index_train = [j for j, xj in enumerate(self.X_train) if xj[2] < close_date_i]
            self.X_train = self.X_train[index_train]
            self.y_train = self.y_train[index_train]
            if len(self.y_train) == 0:
                y_pred.append(np.mean(y_pred))
            else:
                if len(self.y_train) < self.k:
                    knn.n_neighbors = len(self.y_train)
                knn.fit(self.X_train[:, :2], self.y_train)
                y_pred.append(knn.predict([X_test[i][:2]]).item())
        y_pred = np.array(y_pred)
        return y_pred[np.argsort(idx)]

    def getMRAE(self, y_pred, y_test):
        """Calculate the Median Relative Absolute Error

        Parameters
        ----------
        y_pred : array-like, shape (n_query, n_features)
            Predicted target values
        y_test : array-like, shape (n_query, n_features)
            True target values
        Returns
        -------
        MRAE: float
            Median Relative Absolute Error of the model
        """
        return np.median(np.abs(y_pred-y_test)/y_test)

    def score(self, X, y):
        """Returns the MRAE on the given test data and target values.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True target values for X.
        Returns
        -------
        score : float
            MRAE of self.predict(X) wrt. y.
        """
        return self.getMRAE(self.predict(X), y)
def loadDataset(filename):
    """parse csv files into features, and close prices
    Parameters
    ----------
    filename : str
        path to file
    Return
    ----------
    X : array-like, shape (n_samples, n_features)
        Test samples.
    prices: array of float, shape = [n_samples]
        Target values.
    """
    print('### Loading the Housing Prices Data ###\n')
    df = pd.read_csv(filename)
    df['close_date'] = pd.to_datetime(df['close_date'])
    prices = df.pop('close_price').values
    print('### Done ###\n')
    return df.values, prices

if __name__ == '__main__':
    X, y = loadDataset('data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('''######################\n
        Split Housing prices data into training and test Sets \n
        train size = %s \n test size = %s \n
        #####################''' % (len(y_train), len(y_test)))
    regressor = housePricePredictor(k=4, weights='distance', p=2)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    MARE = regressor.getMRAE(y_pred, y_test)
    print('The Median Relative Absolute Error of our KNN Regressor is: %s' % MARE)

#######  Grid Search for best k #######
#     from sklearn.grid_search import GridSearchCV
#     parameters = [{'k': np.arange(1,10)}]
#     clf = GridSearchCV(housePricePredictor(), parameters, n_jobs=-1)
#     clf.fit(X_train, y_train)
#     clf.best_params_

res = pd.DataFrame(np.column_stack((X_test, y_test, y_pred)), columns=['latitude', 'longitude', 'close_date', 'close_price', 'pred_price'])
res.to_csv('predict_res.csv', index=False)
