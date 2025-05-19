from abc import ABC, abstractmethod
from typing import NoReturn
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Lasso as SklearnLasso
import numpy as np
from base_estimator import BaseEstimator


class LinearRegression(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.model = SklearnLR()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

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
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)


class Lasso(BaseEstimator):
    def __init__(self, alpha: float = 1.0, include_intercept: bool = True):
        super().__init__()
        self.include_intercept_ = include_intercept
        self.model = SklearnLasso(alpha=alpha, fit_intercept=include_intercept, max_iter=10000)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

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
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True):
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.lam_ = lam
        self.include_intercept_ = include_intercept
        self.coefs_ = None


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

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
        n, d = X.shape
        if self.include_intercept_:
            # augment with intercept
            X_aug = np.hstack([np.ones((n, 1)), X])  # (n, d+1)
            # regularization matrix
            P = np.eye(d + 1)
            P[0, 0] = 0  # do not regularize intercept
        else:
            X_aug = X
            P = np.eye(d)
        # closed-form solution: (X^T X + lambda*P)^{-1} X^T y
        A = X_aug.T @ X_aug + self.lam_ * P
        b = X_aug.T @ y
        self.coefs_ = np.linalg.solve(A, b)

    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        if self.coefs_ is None:
            raise ValueError("Model is not fitted yet")
        if self.include_intercept_:
            intercept = self.coefs_[0]
            w = self.coefs_[1:]
            return X @ w + intercept
        else:
            return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

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
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

