
import numpy as np
from typing import NoReturn


class LinearRegression:
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem

    Attributes
    ----------
    fitted_ : bool
        Indicates if estimator has been fitted. Set to True in ``self.fit`` function

    include_intercept_: bool
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LinearRegression.fit` function.
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        self.include_intercept_ = include_intercept
        self.coefs_ = None
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            # Add a column of ones for the intercept
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        self.coefs_ = np.linalg.pinv(X) @ y
        self.fitted_ = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction.")

        if self.include_intercept_:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        return X @ self.coefs_

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under **mean squared error (MSE) loss function**

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        predicted = self.predict(X)
        return np.mean((y - predicted) ** 2)

