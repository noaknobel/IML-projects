import os
import pickle

import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
import plotly.express as px

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    # (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Train AdaBoost with 250 stumps
    # n_learners = 250
    # ab_model = AdaBoost(DecisionStump, iterations=n_learners)
    # ab_model.fit(train_X, train_y)

    cache_file = f"ab_cache_noise{noise}_train{train_size}_test{test_size}.pkl"

    if os.path.exists(cache_file):
        # Load data *and* model in one go
        with open(cache_file, "rb") as f:
            train_X, train_y, test_X, test_y, ab_model = pickle.load(f)
    else:
        # 1) generate data
        train_X, train_y = generate_data(train_size, noise)
        test_X, test_y = generate_data(test_size, noise)

        # 2) train model
        ab_model = AdaBoost(DecisionStump, iterations=n_learners)
        ab_model.fit(train_X, train_y)

        # 3) cache everything
        with open(cache_file, "wb") as f:
            pickle.dump((train_X, train_y, test_X, test_y, ab_model), f)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # question_1(ab_model, n_learners, test_X, test_y, train_X, train_y)

    # Question 2: Plotting decision surfaces
    question_2(ab_model, test_X, test_y, train_X)

    # # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    #
    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()


def question_2(ab_model, test_X, test_y, train_X):
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"T = {t}" for t in T],
                        horizontal_spacing=0.05, vertical_spacing=0.07)
    for i, t in enumerate(T):
        r, c = divmod(i, 2)
        row, col = r + 1, c + 1

        fig.add_traces([
            decision_surface(
                lambda X, T=t: ab_model.partial_predict(X, T),
                lims[0], lims[1],
                showscale=False
            ),
            go.Scatter(
                x=test_X[:, 0], y=test_X[:, 1],
                mode="markers", showlegend=False,
                marker=dict(
                    color=test_y,
                    symbol=[class_symbols[int((y_i + 1) // 2)] for y_i in test_y],
                    colorscale=px.colors.diverging.RdBu,
                    line=dict(color="black", width=1),
                    size=6
                )
            )
        ], rows=row, cols=col)

        # set axis ranges
        fig.update_xaxes(range=lims[0], row=row, col=col)
        fig.update_yaxes(range=lims[1], row=row, col=col)
    fig.update_layout(
        title_text="AdaBoost Partial Decision Surfaces (noise=0)",
        height=800, width=800,
    )
    fig.write_html("q2_adaboost_surfaces.html", include_plotlyjs="cdn", full_html=True)


def question_1(ab_model, n_learners, test_X, test_y, train_X, train_y):
    # Compute training & test errors for t = 1…250
    ts = np.arange(1, n_learners + 1)
    train_err = [ab_model.partial_loss(train_X, train_y, t) for t in ts]
    test_err = [ab_model.partial_loss(test_X, test_y, t) for t in ts]
    # Plot both curves
    plt.figure(figsize=(8, 5))
    plt.plot(ts, train_err, label="Train Error")
    plt.plot(ts, test_err, label="Test Error")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("Misclassification Error")
    plt.title("AdaBoost Train vs Test Error (noise=0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ada_train_test_error.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0.0)
