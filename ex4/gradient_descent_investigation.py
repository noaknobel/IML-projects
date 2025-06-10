import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from cross_validate import cross_validate

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange, scaleanchor="y", scaleratio=1),
                                      yaxis=dict(range=yrange),
                                      height=600,
                                      width=600,
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def callback(val, weight, **kwargs):
        values.append(val)
        weights.append(weight)

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for name, Module in {"L1": L1, "L2": L2}.items():
        values_per_eta = {}
        best_eta = None
        best_loss = float('inf')
        for eta in etas:
            module = Module(np.copy(init))
            lr = FixedLR(eta)
            callback, values, weights = get_gd_state_recorder_callback()

            gd = GradientDescent(learning_rate=lr, callback=callback)
            gd.fit(module, None, None)  # X, y are unused in L1/L2

            min_loss = min(values)
            min_loss = min(values)
            if min_loss < best_loss:
                best_loss = min_loss
                best_eta = eta

            descent_path = np.stack(weights)
            values_per_eta[eta] = values
            fig = plot_descent_path(Module, descent_path, title=f"{name}, η={eta}")
            fig.write_html(f"{name}_eta_{eta}.html")

            fig = go.Figure(layout=go.Layout(xaxis=dict(title="GD Iteration"),
                                             yaxis=dict(title="Norm"),

                                             title=f"{name} GD Convergence For Different Learning Rates"))
            for eta, val in values_per_eta.items():
                fig.add_trace(go.Scatter(x=list(range(len(val))), y=val, mode="lines", name=f"η={eta}"))
            fig.write_html(f"gd_{name}_fixed_rate_convergence.html")

        print(f"Lowest loss for {name}: {best_loss} (eta={best_eta})")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    #Fit logistic regression
    from logistic_regression import LogisticRegression
    model = LogisticRegression(penalty="none", include_intercept=True)
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Predict probabilities
    y_score = model.predict_proba(X_train.to_numpy())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression (no regularization)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("logistic_roc.png")
    plt.close()

    model.alpha_ = thresholds[np.argmax(tpr - fpr)]
    print(f"Optimal alpha: {model.alpha_ :}")
    print(f"Test error with alpha = {model.loss(X_test.values, y_test.values):}")

    # Fitting regularized logistic regression, while choosing lambda using cross-validaiton
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_lambda = None
    lowest_val_error = float('inf')

    def zero_one_loss(y_true, y_pred):
        return np.mean(y_true != y_pred)

    for lam in lambdas:
        model = LogisticRegression(
            penalty="l1",
            lam=lam,
            alpha=0.5,
            include_intercept=True,
            solver=GradientDescent(
                learning_rate=FixedLR(1e-4),
                max_iter=20000,
                out_type="last"
            )
        )
        train_err, val_err = cross_validate(
            estimator=model,
            X=X_train.to_numpy(),
            y=y_train.to_numpy(),
            scoring=zero_one_loss,
            cv=5
        )

        if val_err < lowest_val_error:
            lowest_val_error = val_err
            best_lambda = lam

    print(f"Best lambda from CV: {best_lambda:.3f}")

    # Retrain on entire train set using best lambda
    best_model = LogisticRegression(
        penalty="l1",
        lam=best_lambda,
        alpha=0.5,
        include_intercept=True,
        solver=GradientDescent(
            learning_rate=FixedLR(1e-4),
            max_iter=20000,
            out_type="last"
        )
    )
    best_model.fit(X_train.to_numpy(), y_train.to_numpy())
    test_error = best_model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Test error with best lambda = {best_lambda:}: {test_error:}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
