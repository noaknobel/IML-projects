import copy
from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    ids = np.random.permutation(n_samples)
    folds = np.array_split(ids, cv)

    train_scores = []
    val_scores = []
    # for each fold
    for i in range(cv):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(cv) if j != i])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # fresh copy of estimator for each fold
        model = copy.deepcopy(estimator)
        model.fit(X_train, y_train)

        # compute losses
        train_loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)
        train_scores.append(train_loss)
        val_scores.append(val_loss)

    return float(np.mean(train_scores)), float(np.mean(val_scores))
